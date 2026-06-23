"""
Route coverage for kalman_filter.py.
Tests every branch in the KF for all state/measurement type combinations:
  State "0" (range) + Meas "0" (range)
  State "1" (vx/vy/x/y) + Meas "1" (azimuth)
  State "1" (vx/vy/x/y) + Meas "2" (azimuth+range)
  State "1" (vx/vy/x/y) + Meas "3" (azimuth+range+doppler)
  State "1" (vx/vy/x/y) + Meas "4" (rangex+rangey+doppler)
Also covers:
  - FileNotFoundError guards on read_measurements / read_sensor_motion
  - custom config_path constructor argument
  - invalid config raises jsonschema.ValidationError
  - singular S fallback regularisation (linalg.inv raises on first call)
  - single vs multi measurement R matrix
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import jsonschema

from tests.conftest import src_cwd, REPO_ROOT  # noqa

from kalman_filter import KalmanFilter


_CA_STATES = ["ax", "ay", "vx", "vy", "x", "y"]


def _make_kf(src_cwd_fixture, states="1", measurements="4"):
    """Construct a KalmanFilter configured for the given state/meas type."""
    kf = KalmanFilter()
    kf.inputFile["scenario_parameters"]["states"] = states
    kf.inputFile["scenario_parameters"]["measurements"] = measurements
    if states == "0":
        kf.inputFile["scenario_parameters"]["num_states"] = 1
    elif states == "2":
        kf.inputFile["scenario_parameters"]["num_states"] = 6
        # Ensure CA stateType and initial guesses are present regardless of
        # what input.json contains on disk.
        kf.inputFile["scenario_parameters"]["stateTypes"]["2"] = _CA_STATES
        for key in ("ax", "ay"):
            kf.inputFile["scenario_parameters"]["initial_state_guess"].setdefault(key, 0.0)
            kf.inputFile["scenario_parameters"]["initial_covariance_guess"].setdefault(key, 10.0)
    else:
        kf.inputFile["scenario_parameters"]["num_states"] = 4
    kf.setup()
    return kf


# =========================================================================
# init helpers
# =========================================================================

class RoutesKFInit:

    def route_kf_init(self, src_cwd):
        kf = KalmanFilter()
        assert kf.status is False
        assert "scenario_parameters" in kf.inputFile

    def route_kf_setup(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        assert kf.measurements.shape[0] == 3   # rangex, rangey, doppler
        assert kf.measurements.shape[1] == kf.steps

    def route_kf_init_measurement_vectors(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        z, zhat = kf.init_measurment_vectors()
        assert z.shape == zhat.shape

    def route_kf_init_state_vector(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        x = kf.init_state_vector()
        assert x.shape == (4, kf.steps)

    def route_kf_init_covariance_matrix(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        P = kf.init_state_covariance_matrix()
        assert P.shape == (4, 4, kf.steps)

    def route_kf_get_initial_guess_xyz(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        x0, P0 = kf.get_initial_guess()
        assert len(x0) == 4
        assert P0.shape == (4, 4)

    def route_kf_get_initial_guess_range(self, src_cwd):
        kf = _make_kf(src_cwd, "0", "0")
        x0, P0 = kf.get_initial_guess()
        assert len(x0) == 1


# =========================================================================
# R matrix — single vs multi measurement
# =========================================================================

class RoutesKFRMatrix:

    def route_r_single_measurement(self, src_cwd):
        """meas=azimuth → single meas → R is a (1, 1) matrix."""
        kf = _make_kf(src_cwd, "1", "1")
        R = kf.get_measurement_covariance_matrix()
        assert isinstance(R, np.ndarray) and R.shape == (1, 1)

    def route_r_multi_measurement(self, src_cwd):
        """meas=rangex+rangey+doppler → multi → R is (3, 3) matrix."""
        kf = _make_kf(src_cwd, "1", "4")
        R = kf.get_measurement_covariance_matrix()
        assert isinstance(R, np.ndarray) and R.ndim == 2 and R.shape[0] == 3

    def route_r_azimuth_range(self, src_cwd):
        """meas=azimuth+range → 2-element → R is (2, 2) matrix, azimuth in radians."""
        kf = _make_kf(src_cwd, "1", "2")
        R = kf.get_measurement_covariance_matrix()
        assert isinstance(R, np.ndarray) and R.shape == (2, 2)

    def route_r_range_only(self, src_cwd):
        """meas=range → single → R is a (1, 1) matrix."""
        kf = _make_kf(src_cwd, "0", "0")
        R = kf.get_measurement_covariance_matrix()
        assert isinstance(R, np.ndarray) and R.shape == (1, 1)


# =========================================================================
# Transition matrix (A)
# =========================================================================

class RoutesKFTransitionMatrix:

    def route_A_xyz_state(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        A = kf.get_transition_matrix(1, 1.0)
        assert A.shape == (4, 4)
        assert A[2, 0] == pytest.approx(1.0)  # dt in lower-left block

    def route_A_range_state(self, src_cwd):
        kf = _make_kf(src_cwd, "0", "0")
        A = kf.get_transition_matrix(1, 1.0)
        assert A.shape == (1, 1)
        assert A[0, 0] == pytest.approx(1.0)

    def route_A_unknown_state_raises(self, src_cwd):
        """Else branch: unrecognised state type raises ValueError."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.inputFile["scenario_parameters"]["stateTypes"]["1"] = ["unknown"]
        with pytest.raises(ValueError, match="get_transition_matrix"):
            kf.get_transition_matrix(1, 1.0)


# =========================================================================
# Process noise (Q) — all branches
# =========================================================================

class RoutesKFProcessNoise:

    def route_Q_xyz_azimuth(self, src_cwd):
        """meas=azimuth → sub-branch A gamUdk."""
        kf = _make_kf(src_cwd, "1", "1")
        Q = kf.get_process_covariance_matrix(1, 1.0)
        assert Q.shape == (4, 4)

    def route_Q_xyz_azimuth_range(self, src_cwd):
        """meas=azimuth+range → sub-branch A gamUdk."""
        kf = _make_kf(src_cwd, "1", "2")
        Q = kf.get_process_covariance_matrix(1, 1.0)
        assert Q.shape == (4, 4)

    def route_Q_xyz_rangex_rangey_doppler(self, src_cwd):
        """meas=rangex+rangey+doppler → sub-branch A gamUdk."""
        kf = _make_kf(src_cwd, "1", "4")
        Q = kf.get_process_covariance_matrix(1, 1.0)
        assert Q.shape == (4, 4)

    def route_Q_xyz_azimuth_range_doppler(self, src_cwd):
        """meas=azimuth+range+doppler → sub-branch B gamUdk (different gamUdk)."""
        kf = _make_kf(src_cwd, "1", "3")
        Q = kf.get_process_covariance_matrix(1, 1.0)
        assert Q.shape == (4, 4)

    def route_Q_range(self, src_cwd):
        """state=range → Q = [[1]]."""
        kf = _make_kf(src_cwd, "0", "0")
        Q = kf.get_process_covariance_matrix(1, 1.0)
        assert Q.shape == (1, 1)
        assert Q[0, 0] == pytest.approx(1.0)

    def route_Q_unknown_state_raises(self, src_cwd):
        """Else branch: unrecognised state type raises ValueError."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.inputFile["scenario_parameters"]["stateTypes"]["1"] = ["unknown"]
        with pytest.raises(ValueError, match="get_process_covariance_matrix"):
            kf.get_process_covariance_matrix(1, 1.0)


# =========================================================================
# System input vector (u)
# =========================================================================

class RoutesKFInputVector:

    def route_u_xyz_non_range_meas(self, src_cwd):
        """states=xyz → u is a 4-element zero vector (sensor motion in measurement model)."""
        kf = _make_kf(src_cwd, "1", "4")
        u = kf.get_system_input_vector(1, 1.0)
        assert u.shape == (4,)
        np.testing.assert_array_equal(u, np.zeros(4))

    def route_u_range_state(self, src_cwd):
        """states=range → u = [0]."""
        kf = _make_kf(src_cwd, "0", "0")
        u = kf.get_system_input_vector(1, 1.0)
        assert u.shape == (1,)
        assert u[0] == pytest.approx(0.0)


# =========================================================================
# Measurement vectors & H matrix — all 5 measurement type routes
# =========================================================================

class RoutesKFMeasurementMatrix:

    def _run_one_step(self, kf):
        z, zhat = kf.init_measurment_vectors()
        x_est = kf.init_state_vector()
        P = kf.init_state_covariance_matrix()
        x_est[:, 0], P[:, :, 0] = kf.get_initial_guess()
        return kf.get_measurment_vectors_matrices(1, x_est, zhat, 1.0)

    def route_H_rangex_rangey_doppler(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        H, zhat = self._run_one_step(kf)
        assert H.shape == (3, 4)

    def route_H_azimuth_range_doppler(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "3")
        H, zhat = self._run_one_step(kf)
        assert H.shape == (3, 4)

    def route_H_azimuth_range(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "2")
        H, zhat = self._run_one_step(kf)
        assert H.shape == (2, 4)

    def route_H_azimuth_only(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "1")
        H, zhat = self._run_one_step(kf)
        assert H.shape == (1, 4)

    def route_H_range_only(self, src_cwd):
        kf = _make_kf(src_cwd, "0", "0")
        H, zhat = self._run_one_step(kf)
        assert H.shape == (1, 1)
        assert H[0, 0] == pytest.approx(1.0)

    def route_H_unknown_state_meas_raises(self, src_cwd):
        """Else branch: invalid state/meas combination raises ValueError."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.inputFile["scenario_parameters"]["stateTypes"]["1"] = ["unknown"]
        x_est = np.zeros((4, 10))
        zhat = np.zeros((3, 10))
        with pytest.raises(ValueError, match="get_measurment_vectors_matrices"):
            kf.get_measurment_vectors_matrices(1, x_est, zhat, 1.0)


# =========================================================================
# Full Kalman filter run — all 5 route combinations
# =========================================================================

class RoutesKFFullRun:

    def route_kf_run_rangex_rangey_doppler(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        result = kf.kalman_filter()
        assert "x" in result.columns
        assert "y" in result.columns

    def route_kf_run_azimuth_range_doppler(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "3")
        result = kf.kalman_filter()
        assert "x" in result.columns

    def route_kf_run_azimuth_range(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "2")
        result = kf.kalman_filter()
        assert "x" in result.columns

    def route_kf_run_azimuth_only(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "1")
        result = kf.kalman_filter()
        assert "x" in result.columns

    def route_kf_run_range_only(self, src_cwd):
        kf = _make_kf(src_cwd, "0", "0")
        result = kf.kalman_filter()
        assert "range" in result.columns
        assert "P_range" in result.columns

    def route_kf_main(self, src_cwd):
        kf = _make_kf(src_cwd, "1", "4")
        result = kf.main()
        assert kf.status is True
        assert result is not None


# =========================================================================
# Regularisation branch (singular S → except triggers)
# =========================================================================

class RoutesKFRegularisation:

    def route_regularisation_branch(self, src_cwd, monkeypatch):
        """Force linalg.inv to raise on first call to cover the except branch."""
        import numpy.linalg as la

        call_count = {"n": 0}
        real_inv = la.inv

        def mock_inv(x):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise la.LinAlgError("forced singular for test")
            return real_inv(x)

        monkeypatch.setattr(la, "inv", mock_inv)

        kf = _make_kf(src_cwd, "0", "0")
        # Run only 2 steps so the mock triggers exactly on step k=1
        kf.inputFile["simulation_parameters"]["steps"] = 2
        kf.steps = 2
        result = kf.kalman_filter()
        assert result is not None
        assert call_count["n"] >= 2  # first raises, second succeeds


# =========================================================================
# FileNotFoundError guards — read_measurements and read_sensor_motion
# =========================================================================

class RoutesKFFileGuards:

    def route_read_measurements_file_not_found(self, src_cwd, monkeypatch):
        """read_measurements raises FileNotFoundError with helpful message."""
        kf = KalmanFilter()
        kf.states = kf.inputFile["scenario_parameters"]["num_states"]
        kf.steps = kf.inputFile["simulation_parameters"]["steps"]

        def _raise(path, *args, **kwargs):
            raise FileNotFoundError(f"No such file: {path}")

        monkeypatch.setattr(pd, "read_feather", _raise)
        with pytest.raises(FileNotFoundError, match="Run"):
            kf.read_measurements()

    def route_read_sensor_motion_file_not_found(self, src_cwd, monkeypatch):
        """read_sensor_motion raises FileNotFoundError with helpful message."""
        kf = KalmanFilter()
        kf.states = kf.inputFile["scenario_parameters"]["num_states"]
        kf.steps = kf.inputFile["simulation_parameters"]["steps"]

        # Provide real measurements so read_sensor_motion is reached
        real_meas = pd.read_feather(os.path.join(REPO_ROOT, "output", "measurements.ftr"))
        call_count = {"n": 0}

        def _selective_raise(path, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_meas
            raise FileNotFoundError(f"No such file: {path}")

        monkeypatch.setattr(pd, "read_feather", _selective_raise)
        with pytest.raises(FileNotFoundError, match="Run"):
            kf.read_measurements()
            kf.read_sensor_motion()


# =========================================================================
# Custom config_path and invalid config validation
# =========================================================================

class RoutesKFConfigPath:

    def route_kf_custom_config_path(self):
        """config_path constructor argument accepts an explicit file path."""
        path = os.path.join(REPO_ROOT, "input", "input.json")
        kf = KalmanFilter(config_path=path)
        assert kf.status is False
        assert "simulation_parameters" in kf.inputFile

    def route_kf_invalid_config_raises(self, tmp_path):
        """A config file that fails schema validation raises jsonschema.ValidationError."""
        bad_cfg = tmp_path / "bad.json"
        bad_cfg.write_text(json.dumps({}))
        with pytest.raises(jsonschema.ValidationError):
            KalmanFilter(config_path=str(bad_cfg))


# =========================================================================
# System input vector — world-frame state means u is always zero
# =========================================================================

class RoutesKFCVVelocityCompensation:
    """Verify that the CV control-input vector is zero for all sensor motions.

    All states are world-frame quantities; ego-motion is handled entirely in
    the measurement model (zhat_rangex/y subtract sensor position, zhat_doppler
    subtracts sensor velocity).  The prediction step therefore needs no additive
    compensation regardless of sensor acceleration or velocity.
    """

    def route_u_cv_all_zeros_for_any_sensor_velocity(self, src_cwd):
        """u is zero for CV even when sensor has a non-zero constant velocity."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.sensor = kf.sensor.copy()
        kf.sensor["vx"] = 5.0   # constant speed — no acceleration
        kf.sensor["vy"] = 3.0
        u = kf.get_system_input_vector(1, 1.0)
        assert u.shape == (4,)
        np.testing.assert_array_equal(u, np.zeros(4))

    def route_u_cv_all_zeros_for_accelerating_sensor(self, src_cwd):
        """u is zero for CV even when sensor velocity changes between steps."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.sensor = kf.sensor.copy()
        # Linearly increasing sensor vx simulates constant sensor acceleration.
        kf.sensor["vx"] = kf.sensor["vx"] + np.arange(len(kf.sensor)) * 0.5
        u = kf.get_system_input_vector(1, 1.0)
        np.testing.assert_array_equal(u, np.zeros(4))

    def route_u_cv_zero_when_constant_velocity(self, src_cwd):
        """All four u channels are zero, not just the velocity channels."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.sensor = kf.sensor.copy()
        kf.sensor["vx"] = 2.0
        kf.sensor["vy"] = 0.5
        u = kf.get_system_input_vector(1, 1.0)
        np.testing.assert_array_equal(u, np.zeros(4))


# =========================================================================
# Doppler measurement model fix — zhat and H Jacobian use sensor-relative V
# =========================================================================

class RoutesKFDopplerMeasModel:
    """Verify that the doppler predicted measurement and Jacobian account for
    sensor velocity, so the filter estimates world-frame target velocity.
    """

    def _one_step_with_sensor_vel(self, src_cwd, sensor_vx, sensor_vy,
                                   state_vx=1.0, state_vy=0.5,
                                   state_x=200.0, state_y=100.0):
        """Return (H, zhat, Vx_rel_expected, Rxy_expected) for a single step."""
        kf = _make_kf(src_cwd, "1", "4")
        kf.sensor = kf.sensor.copy()
        kf.sensor["vx"] = float(sensor_vx)
        kf.sensor["vy"] = float(sensor_vy)
        # Pin sensor position to zero across all rows so that the geometry is
        # deterministic regardless of which row index the filter reads (k vs k-1).
        kf.sensor["x"] = 0.0
        kf.sensor["y"] = 0.0
        z, zhat = kf.init_measurment_vectors()
        x_est = kf.init_state_vector()
        P = kf.init_state_covariance_matrix()
        x_est[:, 0], P[:, :, 0] = kf.get_initial_guess()
        # Override initial state with known values for deterministic geometry.
        x_est[0, 0] = state_vx   # vx
        x_est[1, 0] = state_vy   # vy
        x_est[2, 0] = state_x    # x
        x_est[3, 0] = state_y    # y
        H, zhat = kf.get_measurment_vectors_matrices(1, x_est, zhat, 1.0)
        return H, zhat, kf

    def route_zhat_doppler_stationary_sensor(self, src_cwd):
        """With sensor_vx=0, zhat_doppler equals the full velocity projection."""
        H, zhat, kf = self._one_step_with_sensor_vel(
            src_cwd, sensor_vx=0.0, sensor_vy=0.0,
            state_vx=2.0, state_vy=1.0, state_x=300.0, state_y=100.0,
        )
        # Sensor position pinned to 0 by _one_step_with_sensor_vel.
        Rx = 300.0; Ry = 100.0
        Rxy = float(np.sqrt(Rx**2 + Ry**2))
        # With sensor velocity = 0: Vx_rel = 2.0, Vy_rel = 1.0
        expected = (Rx * 2.0 + Ry * 1.0) / Rxy
        assert zhat[2, 1] == pytest.approx(expected, rel=1e-6)

    def route_zhat_doppler_subtracts_sensor_velocity(self, src_cwd):
        """zhat_doppler uses Vx-sensor_vx, not Vx alone."""
        sensor_vx, sensor_vy = 5.0, 3.0
        H_rel, zhat_rel, kf = self._one_step_with_sensor_vel(
            src_cwd, sensor_vx=sensor_vx, sensor_vy=sensor_vy,
            state_vx=2.0, state_vy=1.0, state_x=300.0, state_y=100.0,
        )
        # With sensor_vx=5: Vx_rel = 2-5 = -3; expect zhat < case sensor_vx=0
        H_zero, zhat_zero, _ = self._one_step_with_sensor_vel(
            src_cwd, sensor_vx=0.0, sensor_vy=0.0,
            state_vx=2.0, state_vy=1.0, state_x=300.0, state_y=100.0,
        )
        # The doppler zhat must differ by the sensor velocity projection.
        # Sensor position pinned to 0 by _one_step_with_sensor_vel.
        Rx = 300.0; Ry = 100.0
        Rxy = float(np.sqrt(Rx**2 + Ry**2))
        sensor_proj = (Rx * sensor_vx + Ry * sensor_vy) / Rxy
        assert zhat_rel[2, 1] == pytest.approx(zhat_zero[2, 1] - sensor_proj, rel=1e-6)

    def route_H_doppler_jacobian_t7_uses_relative_velocity(self, src_cwd):
        """H[2, 2] (∂doppler/∂x) uses Vx_rel/Rxy term, not Vx/Rxy."""
        sensor_vx = 8.0
        state_vx = 2.0
        H, zhat, kf = self._one_step_with_sensor_vel(
            src_cwd, sensor_vx=sensor_vx, sensor_vy=0.0,
            state_vx=state_vx, state_vy=0.0, state_x=300.0, state_y=100.0,
        )
        # Sensor position pinned to 0 by _one_step_with_sensor_vel.
        Rx = 300.0; Ry = 100.0
        Rxy = float(np.sqrt(Rx**2 + Ry**2))
        Vx_rel = state_vx - sensor_vx   # = -6
        Vy_rel = 0.0 - 0.0              # = 0
        t = (Rx * Vx_rel + Ry * Vy_rel) / Rxy**2
        t7_expected = Vx_rel / Rxy + (-(Rx / Rxy) * t)
        # H[2,2] = t7 (∂doppler/∂x, position column index 2 for CV)
        assert H[2, 2] == pytest.approx(t7_expected, rel=1e-6)

    def route_H_velocity_columns_unchanged(self, src_cwd):
        """H[2,0] and H[2,1] (∂doppler/∂vx, ∂doppler/∂vy) are Rx/Rxy, Ry/Rxy regardless of sensor_vx."""
        sensor_vx = 10.0
        H, zhat, kf = self._one_step_with_sensor_vel(
            src_cwd, sensor_vx=sensor_vx, sensor_vy=0.0,
            state_vx=2.0, state_vy=1.0, state_x=300.0, state_y=100.0,
        )
        # Sensor position pinned to 0 by _one_step_with_sensor_vel.
        Rx = 300.0; Ry = 100.0
        Rxy = float(np.sqrt(Rx**2 + Ry**2))
        # t5 = Rx/Rxy, t6 = Ry/Rxy — independent of sensor velocity
        assert H[2, 0] == pytest.approx(Rx / Rxy, rel=1e-6)  # ∂/∂vx
        assert H[2, 1] == pytest.approx(Ry / Rxy, rel=1e-6)  # ∂/∂vy


# =========================================================================
# CA state model (states == ["ax","ay","vx","vy","x","y"])
# =========================================================================

class RoutesKFCAState:

    def _run_one_step(self, kf):
        z, zhat = kf.init_measurment_vectors()
        x_est = kf.init_state_vector()
        P = kf.init_state_covariance_matrix()
        x_est[:, 0], P[:, :, 0] = kf.get_initial_guess()
        return kf.get_measurment_vectors_matrices(1, x_est, zhat, 1.0)

    # --- Transition matrix --------------------------------------------------

    def route_A_ca_state(self, src_cwd):
        """CA state → 6×6 transition matrix with correct dt and dt²/2 entries."""
        kf = _make_kf(src_cwd, "2", "4")
        dt = 2.0
        A = kf.get_transition_matrix(1, dt)
        assert A.shape == (6, 6)
        assert A[2, 0] == pytest.approx(dt)              # vx ← ax·dt
        assert A[3, 1] == pytest.approx(dt)              # vy ← ay·dt
        assert A[4, 2] == pytest.approx(dt)              # x  ← vx·dt
        assert A[5, 3] == pytest.approx(dt)              # y  ← vy·dt
        assert A[4, 0] == pytest.approx(0.5 * dt ** 2)  # x  ← ax·dt²/2
        assert A[5, 1] == pytest.approx(0.5 * dt ** 2)  # y  ← ay·dt²/2
        assert A[0, 0] == pytest.approx(1.0)             # ax persists
        assert A[4, 3] == pytest.approx(0.0)             # x  ← ay = 0 (decoupled)

    # --- Process noise Q ----------------------------------------------------

    def route_Q_ca_state_symmetric_psd(self, src_cwd):
        """CA Q is 6×6, symmetric, and positive semi-definite for several dt values."""
        kf = _make_kf(src_cwd, "2", "4")
        for dt in (0.1, 1.0, 2.0, 5.0):
            Q = kf.get_process_covariance_matrix(1, dt)
            assert Q.shape == (6, 6)
            assert np.allclose(Q, Q.T), f"Q not symmetric at dt={dt}"
            assert np.all(np.linalg.eigvalsh(Q) >= -1e-10), f"Q not PSD at dt={dt}"

    def route_Q_ca_state_scaling(self, src_cwd):
        """CA Q entries scale with powers of dt as expected from Van Loan derivation."""
        kf = _make_kf(src_cwd, "2", "4")
        q = kf.inputFile["scenario_parameters"]["process_noise"]
        dt = 1.0
        Q = kf.get_process_covariance_matrix(1, dt)
        # Q[0,0] = q*dt  (ax variance row)
        assert Q[0, 0] == pytest.approx(q * dt)
        # Q[2,0] = q*dt²/2  (vx-ax cross term)
        assert Q[2, 0] == pytest.approx(q * dt**2 / 2)
        # Q[4,0] = q*dt³/6  (x-ax cross term)
        assert Q[4, 0] == pytest.approx(q * dt**3 / 6)
        # Q[4,4] = q*dt⁵/20  (x variance)
        assert Q[4, 4] == pytest.approx(q * dt**5 / 20)
        # x and y channels are independent: Q[4,5] = 0
        assert Q[4, 5] == pytest.approx(0.0)

    # --- System input u -----------------------------------------------------

    def route_u_ca_state_shape_and_values(self, src_cwd):
        """CA u is 6-element zero vector — sensor motion is in the measurement model."""
        kf = _make_kf(src_cwd, "2", "4")
        u = kf.get_system_input_vector(1, 1.0)
        assert u.shape == (6,)
        np.testing.assert_array_equal(u, np.zeros(6))

    # --- H matrix: all measurement types ------------------------------------

    def route_H_ca_rangex_rangey_doppler(self, src_cwd):
        """CA + meas 4 → H (3,6); rangex/y rows touch only x/y cols (4,5)."""
        kf = _make_kf(src_cwd, "2", "4")
        H, _ = self._run_one_step(kf)
        assert H.shape == (3, 6)
        assert H[0, 4] == pytest.approx(1.0)   # rangex → x col
        assert H[1, 5] == pytest.approx(1.0)   # rangey → y col
        assert H[0, 0] == pytest.approx(0.0)   # rangex → ax col = 0
        assert H[0, 2] == pytest.approx(0.0)   # rangex → vx col = 0

    def route_H_ca_azimuth_range_doppler(self, src_cwd):
        """CA + meas 3 → H (3,6); az/range rows have zero in accel/vel cols."""
        kf = _make_kf(src_cwd, "2", "3")
        H, _ = self._run_one_step(kf)
        assert H.shape == (3, 6)
        assert H[0, 0] == pytest.approx(0.0)   # az → ax = 0
        assert H[0, 2] == pytest.approx(0.0)   # az → vx = 0
        assert H[1, 0] == pytest.approx(0.0)   # range → ax = 0
        assert H[1, 2] == pytest.approx(0.0)   # range → vx = 0
        # doppler row touches vx/vy (cols 2,3) and x/y (cols 4,5), not ax/ay
        assert H[2, 0] == pytest.approx(0.0)
        assert H[2, 1] == pytest.approx(0.0)

    def route_H_ca_azimuth_range(self, src_cwd):
        """CA + meas 2 → H (2,6); both rows have zero accel/vel columns."""
        kf = _make_kf(src_cwd, "2", "2")
        H, _ = self._run_one_step(kf)
        assert H.shape == (2, 6)
        assert H[0, 0] == pytest.approx(0.0)
        assert H[0, 2] == pytest.approx(0.0)

    def route_H_ca_azimuth_only(self, src_cwd):
        """CA + meas 1 → H (1,6); azimuth row touches only x/y cols."""
        kf = _make_kf(src_cwd, "2", "1")
        H, _ = self._run_one_step(kf)
        assert H.shape == (1, 6)
        assert H[0, 0] == pytest.approx(0.0)   # az → ax = 0
        assert H[0, 2] == pytest.approx(0.0)   # az → vx = 0

    # --- Full filter runs ---------------------------------------------------

    def route_kf_run_ca_meas4(self, src_cwd):
        """CA state + rangex/rangey/doppler: filter runs and returns all 6 states."""
        kf = _make_kf(src_cwd, "2", "4")
        kf.inputFile["simulation_parameters"]["steps"] = 5
        kf.steps = 5
        result = kf.kalman_filter()
        for col in ("ax", "ay", "vx", "vy", "x", "y"):
            assert col in result.columns
        assert "P_ax" in result.columns

    def route_kf_run_ca_meas3(self, src_cwd):
        """CA state + azimuth/range/doppler: filter completes without error."""
        kf = _make_kf(src_cwd, "2", "3")
        kf.inputFile["simulation_parameters"]["steps"] = 5
        kf.steps = 5
        result = kf.kalman_filter()
        assert "ax" in result.columns

    def route_kf_run_ca_meas2(self, src_cwd):
        """CA state + azimuth/range: filter completes without error."""
        kf = _make_kf(src_cwd, "2", "2")
        kf.inputFile["simulation_parameters"]["steps"] = 5
        kf.steps = 5
        result = kf.kalman_filter()
        assert "ax" in result.columns

    def route_kf_run_ca_meas1(self, src_cwd):
        """CA state + azimuth: filter completes without error."""
        kf = _make_kf(src_cwd, "2", "1")
        kf.inputFile["simulation_parameters"]["steps"] = 5
        kf.steps = 5
        result = kf.kalman_filter()
        assert "ax" in result.columns

    def route_kf_run_ca_output_columns(self, src_cwd):
        """kalman_filter() with CA state returns dynamic columns, not hardcoded CV set."""
        kf = _make_kf(src_cwd, "2", "4")
        kf.inputFile["simulation_parameters"]["steps"] = 3
        kf.steps = 3
        result = kf.kalman_filter()
        # CV columns still present
        for col in ("vx", "vy", "x", "y"):
            assert col in result.columns
        # CA-only columns present
        for col in ("ax", "ay", "P_ax", "P_ay"):
            assert col in result.columns
        # Total rows match steps
        assert len(result) == 3


# =========================================================================
# process_noise > 1.0 warning
# =========================================================================

class RoutesKFProcessNoiseWarning:

    def route_process_noise_warning(self, src_cwd, caplog):
        """process_noise > 1.0 emits a WARNING before the filter loop."""
        import logging
        kf = _make_kf(src_cwd, "0", "0")
        kf.inputFile["scenario_parameters"]["process_noise"] = 2.0
        kf.inputFile["simulation_parameters"]["steps"] = 2
        kf.steps = 2
        with caplog.at_level(logging.WARNING, logger="kalman_filter"):
            kf.kalman_filter()
        assert any("safe bound" in r.message for r in caplog.records)

    def route_no_warning_below_threshold(self, src_cwd, caplog):
        """process_noise <= 1.0 does NOT emit the safe-bound warning."""
        import logging
        kf = _make_kf(src_cwd, "0", "0")
        kf.inputFile["scenario_parameters"]["process_noise"] = 0.5
        kf.inputFile["simulation_parameters"]["steps"] = 2
        kf.steps = 2
        with caplog.at_level(logging.WARNING, logger="kalman_filter"):
            kf.kalman_filter()
        assert not any("safe bound" in r.message for r in caplog.records)

