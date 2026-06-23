"""Linear Kalman filter for target tracking.

Supports two state configurations (``["vx","vy","x","y"]`` and ``["range"]``)
and five measurement types, as specified in ``input/input.json``.
"""
import os
import logging
import json
from typing import Any

import numpy as np
import pandas as pd
from numpy import arctan2, sqrt
from numpy import zeros, radians, linalg, array, diag, eye

from config_schema import validate_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(_SRC_DIR, "..", "input")
_OUTPUT_DIR = os.path.join(_SRC_DIR, "..", "output")


class KalmanFilter:
    """Linear Kalman filter wrapping configuration, I/O, and the filter loop.

    Configuration is read from ``input/input.json`` on construction (or from
    *config_path* if supplied).  Call :meth:`main` to run the full pipeline;
    individual sub-steps are exposed as methods for unit testing.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Load ``input.json`` and set the initial filter status flag.

        @param config_path: Optional explicit path to a configuration JSON file.
            When *None* (default) the canonical ``input/input.json`` is used.
        @type config_path: str | None
        """
        path = config_path if config_path is not None else os.path.join(_INPUT_DIR, "input.json")
        with open(path, "r") as file:
            self.inputFile: dict[str, Any] = json.loads(file.read())
        validate_config(self.inputFile)
        self.status: bool = False

    def setup(self) -> None:
        """Read configuration scalars, measurement data, and sensor motion.

        Sets ``self.states``, ``self.steps``, and calls
        :meth:`read_measurements` and :meth:`read_sensor_motion`.
        """
        self.states = self.inputFile["scenario_parameters"]["num_states"]
        self.steps = self.inputFile["simulation_parameters"]["steps"]
        self.read_measurements()
        self.read_sensor_motion()

    def read_measurements(self) -> None:
        """Load the measurement Feather file and select configured columns.

        The selected columns are stored as a NumPy array in
        ``self.measurements`` with shape ``(num_meas, steps)``.

        @raises FileNotFoundError: If the measurements Feather file does not exist.
            Run the simulation (ObjectMotionSimulator) first to generate it.
        """
        meas_path = os.path.join(_OUTPUT_DIR, "measurements.ftr")
        try:
            self.measurements = pd.read_feather(meas_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Measurement file not found: {meas_path!r}. "
                "Run the object motion simulator first to generate output files."
            ) from exc
        meas = self.inputFile["scenario_parameters"]["measurementsTypes"][
            self.inputFile["scenario_parameters"]["measurements"]
        ]
        self.measurements = self.measurements[meas].to_numpy().T

    def read_sensor_motion(self) -> None:
        """Load the sensor trajectory Feather file into ``self.sensor``.

        @raises FileNotFoundError: If the sensor trajectory file does not exist.
        """
        sensor_path = os.path.join(_OUTPUT_DIR, "sensor_trajectory.ftr")
        try:
            self.sensor = pd.read_feather(sensor_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Sensor trajectory file not found: {sensor_path!r}. "
                "Run the object motion simulator first."
            ) from exc

    def init_measurment_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Allocate measurement and predicted-measurement arrays.

        @return: Tuple ``(z, zhat)`` — both have the same shape as
            ``self.measurements``.
        @rtype: tuple[np.ndarray, np.ndarray]
        """
        z = self.measurements
        zhat = zeros(z.shape)
        return z, zhat

    def init_state_vector(self) -> np.ndarray:
        """Allocate the state estimate array.

        @return: Zero array of shape ``(num_states, steps)``.
        @rtype: np.ndarray
        """
        x = zeros((self.states, self.steps))
        return x

    def init_state_covariance_matrix(self) -> np.ndarray:
        """Allocate the state covariance history array.

        @return: Zero array of shape ``(num_states, num_states, steps)``.
        @rtype: np.ndarray
        """
        P = zeros((self.states, self.states, self.steps))
        return P

    def get_initial_guess(self) -> tuple[list[float], np.ndarray]:
        """Build the initial state vector and covariance matrix from config.

        @return: Tuple ``(guessStates, guessStateCovariance)`` where
            ``guessStates`` is a list of scalar initial values and
            ``guessStateCovariance`` is a diagonal covariance matrix.
        @rtype: tuple[list[float], np.ndarray]
        """
        states = self.inputFile["scenario_parameters"]["stateTypes"][
            self.inputFile["scenario_parameters"]["states"]
        ]
        guessStates = [self.inputFile["scenario_parameters"]["initial_state_guess"][i] for i in states]
        guessStateCovariance = diag(
            [(self.inputFile["scenario_parameters"]["initial_covariance_guess"][i] ** 2) / 3 for i in states]
        )
        return guessStates, guessStateCovariance

    def get_measurement_covariance_matrix(self) -> np.ndarray:
        """Construct the measurement noise covariance matrix *R*.

        Azimuth noise is converted from degrees to radians before squaring.
        Always returns a 2-D diagonal matrix regardless of the number of
        measurements, so the result is safe for matrix multiplication.

        @return: Diagonal measurement noise covariance matrix.
        @rtype: np.ndarray
        """
        noises = self.inputFile["scenario_parameters"]["meas_noise_guess"]
        meas = self.inputFile["scenario_parameters"]["measurementsTypes"][
            self.inputFile["scenario_parameters"]["measurements"]
        ]
        n_ = [
            3 * (noises[k] ** 2) if k != "azimuth" else 3 * (radians(noises[k]) ** 2)
            for k in meas
        ]
        return diag(n_)

    def get_transition_matrix(self, k: int, dt: float) -> np.ndarray:
        """Return the state transition matrix *A* for the current step.

        @param k: Current time index (unused; retained for interface consistency).
        @type k: int
        @param dt: Time-step duration in seconds.
        @type dt: float
        @return: State transition matrix of appropriate dimension.
        @rtype: np.ndarray
        """
        states = self.inputFile["scenario_parameters"]["stateTypes"][
            self.inputFile["scenario_parameters"]["states"]
        ]
        if states == ["vx", "vy", "x", "y"]:
            A = array([
                [1,  0,  0, 0],
                [0,  1,  0, 0],
                [dt, 0,  1, 0],
                [0,  dt, 0, 1],
            ])
        elif states == ["ax", "ay", "vx", "vy", "x", "y"]:
            dt2 = 0.5 * dt ** 2
            A = array([
                [1,  0,  0,  0,  0,  0],
                [0,  1,  0,  0,  0,  0],
                [dt, 0,  1,  0,  0,  0],
                [0,  dt, 0,  1,  0,  0],
                [dt2, 0, dt, 0,  1,  0],
                [0, dt2,  0, dt, 0,  1],
            ])
        elif states == ["range"]:
            A = array([[1]])
        else:
            raise ValueError(f"get_transition_matrix: unsupported state type {states!r}")
        return A

    def get_process_covariance_matrix(self, k: int, dt: float) -> np.ndarray:
        """Compute the process noise covariance matrix *Q* for the current step.

        @param k: Current time index (unused; retained for interface consistency).
        @type k: int
        @param dt: Time-step duration in seconds.
        @type dt: float
        @return: Process noise covariance matrix.
        @rtype: np.ndarray
        """
        states = self.inputFile["scenario_parameters"]["stateTypes"][
            self.inputFile["scenario_parameters"]["states"]
        ]
        meas = self.inputFile["scenario_parameters"]["measurementsTypes"][
            self.inputFile["scenario_parameters"]["measurements"]
        ]

        process_noise = self.inputFile["scenario_parameters"]["process_noise"]

        if states == ["vx", "vy", "x", "y"]:
            if meas in (["azimuth"], ["azimuth", "range"], ["rangex", "rangey", "doppler"]):
                gamUdk = array([
                    [dt,              0],
                    [0,              dt],
                    [0.5 * (dt ** 2), 0],
                    [0,  0.5 * (dt ** 2)],
                ])
            elif meas == ["azimuth", "range", "doppler"]:
                gamUdk = array([
                    [0,   0],
                    [0,   0],
                    [-dt, 0],
                    [0, -dt],
                ])
            Q = array([
                [(dt ** 3) / 3, (dt ** 2) / 2],
                [(dt ** 2) / 2,            dt],
            ])
            Q = Q * process_noise
            Q = gamUdk @ Q @ gamUdk.T
        elif states == ["ax", "ay", "vx", "vy", "x", "y"]:
            # Discrete Q for CA model driven by jerk noise (process_noise = jerk PSD).
            # Derived from Van Loan's method; x and y channels are independent.
            dt2, dt3, dt4, dt5 = dt**2, dt**3, dt**4, dt**5
            Q = process_noise * array([
                [dt,      0,      dt2/2,  0,      dt3/6,  0     ],
                [0,       dt,     0,      dt2/2,  0,      dt3/6 ],
                [dt2/2,   0,      dt3/3,  0,      dt4/8,  0     ],
                [0,       dt2/2,  0,      dt3/3,  0,      dt4/8 ],
                [dt3/6,   0,      dt4/8,  0,      dt5/20, 0     ],
                [0,       dt3/6,  0,      dt4/8,  0,      dt5/20],
            ])
        elif states == ["range"]:
            Q = array([[1]])
        else:
            raise ValueError(f"get_process_covariance_matrix: unsupported state type {states!r}")
        return Q

    def get_system_input_vector(self, k: int, dt: float) -> np.ndarray:
        """Return the zero control-input vector.

        All filter states are world-frame quantities.  Sensor ego-motion is
        fully accounted for in the measurement model: ``zhat_rangex/y``
        subtracts sensor position and ``zhat_doppler`` subtracts sensor
        velocity (see :meth:`get_measurment_vectors_matrices`).  No additive
        prediction-step correction is therefore needed.

        @param k: Current time index (unused; retained for interface consistency).
        @type k: int
        @param dt: Time-step duration in seconds (unused; retained for interface
            consistency).
        @type dt: float
        @return: Zero control input vector of appropriate length.
        @rtype: np.ndarray
        """
        states = self.inputFile["scenario_parameters"]["stateTypes"][
            self.inputFile["scenario_parameters"]["states"]
        ]
        if states == ["vx", "vy", "x", "y"]:
            return zeros(4)
        elif states == ["ax", "ay", "vx", "vy", "x", "y"]:
            return zeros(6)
        else:
            return array([0.0])

    def get_measurment_vectors_matrices(
        self,
        k: int,
        x_est: np.ndarray,
        zhat: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the Jacobian measurement matrix *H* and predicted measurement.

        Linearises the nonlinear observation model around the prior estimate.

        @param k: Current time index (must be >= 1).
        @type k: int
        @param x_est: Full state estimate array of shape ``(num_states, steps)``.
        @type x_est: np.ndarray
        @param zhat: Predicted measurement array; column *k* is updated in place.
        @type zhat: np.ndarray
        @param dt: Time-step duration in seconds (unused; retained for interface).
        @type dt: float
        @return: Tuple ``(H, zhat)`` — linearised observation matrix and the
            updated predicted-measurement array.
        @rtype: tuple[np.ndarray, np.ndarray]
        """
        states = self.inputFile["scenario_parameters"]["stateTypes"][
            self.inputFile["scenario_parameters"]["states"]
        ]
        meas = self.inputFile["scenario_parameters"]["measurementsTypes"][
            self.inputFile["scenario_parameters"]["measurements"]
        ]

        if states in (["vx", "vy", "x", "y"], ["ax", "ay", "vx", "vy", "x", "y"]):
            is_ca = states == ["ax", "ay", "vx", "vy", "x", "y"]
            # State indices differ between CV and CA:
            #   CV:  vx=0  vy=1  x=2  y=3
            #   CA:  ax=0  ay=1  vx=2  vy=3  x=4  y=5
            vi_x, vi_y = (2, 3) if is_ca else (0, 1)
            pi_x, pi_y = (4, 5) if is_ca else (2, 3)
            n = 6 if is_ca else 4

            Vx = x_est[vi_x, k - 1]
            Vy = x_est[vi_y, k - 1]
            # Use sensor state at index k (the time the measurement z[:,k] was
            # taken), not k-1.  Using the stale k-1 position/velocity would
            # introduce a spurious bias proportional to sensor speed in the
            # rangex/rangey innovation and a mismatch in the doppler innovation.
            Rx = x_est[pi_x, k - 1] - self.sensor["x"].iloc[k]
            Ry = x_est[pi_y, k - 1] - self.sensor["y"].iloc[k]
            az_ = arctan2(Ry, Rx)
            Rxy_ = max(float(sqrt(Rx ** 2 + Ry ** 2)), 1e-9)

            # Sensor velocity at the measurement time k — subtracted so that
            # zhat_doppler predicts the radial relative velocity that the
            # simulation (and a real radar) actually measures.
            sensor_vx_k = float(self.sensor["vx"].iloc[k])
            sensor_vy_k = float(self.sensor["vy"].iloc[k])
            Vx_rel = Vx - sensor_vx_k   # target velocity relative to sensor (world-frame vx minus sensor vx)
            Vy_rel = Vy - sensor_vy_k

            Vrelrad = (Rx * Vx_rel + Ry * Vy_rel) / Rxy_
            t1 = -Ry / (Rxy_ ** 2)
            t2 = Rx / (Rxy_ ** 2)
            t3 = Rx / Rxy_
            t4 = Ry / Rxy_
            # t5/t6: ∂Vrelrad/∂(vx,vy) = Rx/Rxy_, Ry/Rxy_  (sensor_vx is constant w.r.t. state)
            t5 = t3
            t6 = t4
            # t7/t8: ∂Vrelrad/∂(x,y)  — use Vx_rel/Vy_rel so linearisation matches corrected zhat
            t = (Rx * Vx_rel + Ry * Vy_rel) / (Rxy_ ** 2)
            t7 = Vx_rel / Rxy_ + (-t3 * t)
            t8 = Vy_rel / Rxy_ + (-t4 * t)

            # Build sparse H rows — zeros everywhere except the active state columns.
            def _row(*cols_vals):
                r = [0.0] * n
                for idx, val in cols_vals:
                    r[idx] = val
                return r

            if meas == ["rangex", "rangey", "doppler"]:
                H = array([
                    _row((pi_x, 1)),
                    _row((pi_y, 1)),
                    _row((vi_x, t5), (vi_y, t6), (pi_x, t7), (pi_y, t8)),
                ])
                zhat[0, k] = Rx
                zhat[1, k] = Ry
                zhat[2, k] = Vrelrad
            elif meas == ["azimuth", "range", "doppler"]:
                H = array([
                    _row((pi_x, t1), (pi_y, t2)),
                    _row((pi_x, t3), (pi_y, t4)),
                    _row((vi_x, t5), (vi_y, t6), (pi_x, t7), (pi_y, t8)),
                ])
                zhat[0, k] = az_
                zhat[1, k] = Rxy_
                zhat[2, k] = Vrelrad
            elif meas == ["azimuth", "range"]:
                H = array([
                    _row((pi_x, t1), (pi_y, t2)),
                    _row((pi_x, t3), (pi_y, t4)),
                ])
                zhat[0, k] = az_
                zhat[1, k] = Rxy_
            elif meas == ["azimuth"]:
                H = array([_row((pi_x, t1), (pi_y, t2))])
                zhat[0, k] = az_
        elif states == ["range"] and meas == ["range"]:
            H = array([[1]])
            zhat[0, k] = x_est[0, k - 1]
        else:
            raise ValueError(
                f"get_measurment_vectors_matrices: unsupported state/meas combination "
                f"states={states!r} meas={meas!r}"
            )
        return H, zhat

    def kalman_filter(self) -> pd.DataFrame:
        """Execute the linear Kalman filter loop over all time steps.

        Static matrices *A*, *Q*, *R* are pre-computed once before the loop
        (performance improvement for large step counts).

        @return: Estimate DataFrame with one row per time step.
        @rtype: pd.DataFrame
        """
        z, zhat = self.init_measurment_vectors()
        x_est = self.init_state_vector()
        P = self.init_state_covariance_matrix()

        x_est[:, 0], P[:, :, 0] = self.get_initial_guess()

        dt = float(self.inputFile["simulation_parameters"]["time_step"])
        process_noise = float(self.inputFile["scenario_parameters"]["process_noise"])
        if process_noise > 1.0:
            logger.warning(
                "process_noise q=%.4g exceeds safe bound; filter diverges above q≈1.6. "
                "Keep q < 1.0 for stable operation.",
                process_noise,
            )

        # Pre-compute static matrices outside the loop
        A = self.get_transition_matrix(1, dt)
        Q = self.get_process_covariance_matrix(1, dt)
        R = self.get_measurement_covariance_matrix()

        timestamps = [0.0]
        time = 0.0
        log_interval = max(1, self.steps // 10)

        for k in range(1, self.steps):
            time += dt
            timestamps.append(time)

            if k % log_interval == 0:
                logger.info("KF step %d/%d (%.0f%%)", k, self.steps, 100 * k / self.steps)

            u = self.get_system_input_vector(k, dt)
            H, zhat = self.get_measurment_vectors_matrices(k, x_est, zhat, dt)

            x_pred = A @ x_est[:, k - 1] + u
            P_pred = A @ P[:, :, k - 1] @ A.T + Q

            nu = z[:, k] - zhat[:, k]
            S = R + H @ P_pred @ H.T

            try:
                Sinv = linalg.inv(S)
            except linalg.LinAlgError:
                logger.warning("KF step %d: singular S — fallback regularisation", k)
                S = S + eye(S.shape[0]) * 1e-4
                Sinv = linalg.inv(S)

            K = P_pred @ H.T @ Sinv
            x_est[:, k] = x_pred + K @ nu
            # Joseph form P⁺ = (I−KH)P⁻(I−KH)ᵀ + KRKᵀ is symmetric PSD by
            # construction, preventing the covariance from going negative due
            # to floating-point rounding — critical for the 6-state CA model.
            I_KH = eye(self.states) - K @ H
            P[:, :, k] = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

        states = self.inputFile["scenario_parameters"]["stateTypes"][
            self.inputFile["scenario_parameters"]["states"]
        ]
        df = pd.DataFrame(columns=["timestamps"] + list(states))
        df["timestamps"] = timestamps

        for i, s in enumerate(states):
            df[s] = list(x_est[i, :].flatten())
            df["P_" + s] = list(P[i, i, :].flatten())
        return df

    def main(self) -> pd.DataFrame:
        """Run the full setup then filter pipeline.

        @return: Estimate DataFrame; see :meth:`kalman_filter`.
        @rtype: pd.DataFrame
        """
        self.setup()
        estimates = self.kalman_filter()
        self.status = True
        return estimates
