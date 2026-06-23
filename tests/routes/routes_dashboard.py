"""
Route coverage for dashboard.py.
Tests every figure-building function and every Dash callback function
for all branching paths (state types, measurement types, UI conditions).
Also tests new operational features:
  - _check_environment passes / raises FileNotFoundError
  - _validate_startup_config passes / logs error silently
  - _load_disk_cache: nonexistent, valid, corrupt
  - _save_disk_cache: success, silent failure
  - run_simulation graceful exception handling returns 6 empty figures
Uses synthetic DataFrames from conftest rather than running the real simulation.
"""
import os
import json
import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import dash

from tests.conftest import (
    src_cwd, make_sensor_df, make_target_df,
    make_estimates_xyz, make_estimates_range, make_measurements_df,
    load_input_json, REPO_ROOT,
)

import dashboard as db


# =========================================================================
# Shared helpers
# =========================================================================

def _cfg(states="1", measurements="4", target_speed=1.0):
    cfg = load_input_json()
    cfg["scenario_parameters"]["states"] = states
    cfg["scenario_parameters"]["measurements"] = measurements
    cfg["initial_states"]["target"]["speed"] = target_speed
    return cfg


SE = make_sensor_df()
ST = make_target_df()
EST_XYZ = make_estimates_xyz()
EST_RNG = make_estimates_range()
MEAS = make_measurements_df()
NOISE = {"range": 0.05, "azimuth": 0.1, "doppler": 0.03}


# =========================================================================
# get_trajectory_fig — 3 branches
# =========================================================================

class RoutesTrajectoryFig:

    def route_trajectory_fig_xyz_state(self):
        """Branch: states != range → 2D x/y trajectory."""
        fig = db.get_trajectory_fig(SE, ST, EST_XYZ, _cfg("1", "4"), MEAS)
        assert fig is not None
        assert len(fig.data) > 0

    def route_trajectory_fig_range_state_moving_target(self):
        """Branch: states == range, target speed != 0."""
        fig = db.get_trajectory_fig(SE, ST, EST_XYZ, _cfg("0", "0", target_speed=1.0), MEAS)
        assert fig is not None

    def route_trajectory_fig_range_state_static_target(self):
        """Branch: states == range, target speed == 0."""
        fig = db.get_trajectory_fig(SE, ST, EST_XYZ, _cfg("0", "0", target_speed=0.0), MEAS)
        assert fig is not None

    def route_trajectory_fig_empty_dataframe_returns_blank_figure(self):
        """Guard: empty DataFrame (slider at 0) must not raise IndexError."""
        empty = pd.DataFrame(columns=ST.columns)
        fig = db.get_trajectory_fig(empty, empty, empty, _cfg("1", "4"), MEAS)
        assert fig is not None
        assert len(fig.data) == 0


# =========================================================================
# get_error_dist_fig — 2 branches
# =========================================================================

class RoutesErrorDistFig:

    def route_error_dist_xyz_state(self):
        """Branch: states == [vx,vy,x,y]."""
        fig = db.get_error_dist_fig(SE, ST, EST_XYZ, _cfg("1", "4"), NOISE, MEAS)
        assert fig is not None
        assert len(fig.data) >= 2

    def route_error_dist_range_state(self):
        """Branch: states == [range]."""
        fig = db.get_error_dist_fig(SE, ST, EST_RNG, _cfg("0", "0"), NOISE, MEAS)
        assert fig is not None
        assert len(fig.data) >= 1

    def route_error_dist_unknown_state_returns_empty_with_warning(self):
        """Else branch: unrecognised state type logs warning, returns empty figure."""
        cfg = _cfg("1", "4")
        cfg["scenario_parameters"]["stateTypes"]["1"] = ["unknown"]
        fig = db.get_error_dist_fig(SE, ST, EST_XYZ, cfg, NOISE, MEAS)
        assert fig is not None
        assert len(fig.data) == 0


# =========================================================================
# get_error_vel_fig — 2 branches
# =========================================================================

class RoutesErrorVelFig:

    def route_error_vel_xyz_state(self):
        """Branch: states == [vx,vy,x,y] → velocity error traces added."""
        fig = db.get_error_vel_fig(SE, ST, EST_XYZ, _cfg("1", "4"), NOISE, MEAS)
        assert fig is not None
        assert len(fig.data) >= 2

    def route_error_vel_range_state(self):
        """Branch: states == [range] → no traces added, still returns figure."""
        fig = db.get_error_vel_fig(SE, ST, EST_RNG, _cfg("0", "0"), NOISE, MEAS)
        assert fig is not None

    def route_error_vel_unknown_state_returns_empty_with_warning(self):
        """Else branch: unrecognised state type logs warning, returns empty figure."""
        cfg = _cfg("1", "4")
        cfg["scenario_parameters"]["stateTypes"]["1"] = ["unknown"]
        fig = db.get_error_vel_fig(SE, ST, EST_XYZ, cfg, NOISE, MEAS)
        assert fig is not None
        assert len(fig.data) == 0


# =========================================================================
# get_range_measurement, get_range_only_measurement,
# get_azimuth_measurement, get_doppler_measurement
# =========================================================================

class RoutesMeasurementFigs:

    def route_range_measurement(self):
        fig = db.go.Figure()
        result = db.get_range_measurement(fig, SE, ST, EST_XYZ, _cfg("1", "4"), MEAS)
        assert result is not None

    def route_range_only_measurement(self):
        fig = db.go.Figure()
        result = db.get_range_only_measurement(fig, SE, ST, EST_RNG, _cfg("0", "0"), MEAS)
        assert result is not None

    def route_azimuth_measurement(self):
        fig = db.go.Figure()
        result = db.get_azimuth_measurement(fig, SE, ST, EST_XYZ, _cfg("1", "1"), MEAS)
        assert result is not None

    def route_doppler_measurement(self):
        fig = db.go.Figure()
        result = db.get_doppler_measurement(fig, SE, ST, EST_XYZ, _cfg("1", "4"), MEAS)
        assert result is not None

    def route_measurement_figs_meas4(self):
        """meas=rangex/rangey/doppler → calls range+azimuth+doppler subfigs."""
        figs = db.get_measurement_figs(SE, ST, EST_XYZ, _cfg("1", "4"), MEAS)
        assert len(figs) == 3

    def route_measurement_figs_meas3(self):
        """meas=azimuth+range+doppler."""
        figs = db.get_measurement_figs(SE, ST, EST_XYZ, _cfg("1", "3"), MEAS)
        assert len(figs) == 3

    def route_measurement_figs_meas2(self):
        """meas=azimuth+range."""
        figs = db.get_measurement_figs(SE, ST, EST_XYZ, _cfg("1", "2"), MEAS)
        assert len(figs) == 3

    def route_measurement_figs_meas1(self):
        """meas=azimuth."""
        figs = db.get_measurement_figs(SE, ST, EST_XYZ, _cfg("1", "1"), MEAS)
        assert len(figs) == 3

    def route_measurement_figs_meas0(self):
        """meas=range (range-only state)."""
        figs = db.get_measurement_figs(SE, ST, EST_RNG, _cfg("0", "0"), MEAS)
        assert len(figs) == 3


# =========================================================================
# plot_trajectories_E (dashboard version) — runs real simulation
# =========================================================================

class RoutesPlotTrajectoriesE:

    def _write_valid_config(self):
        """Write a known-valid config (state=1/meas=4) so tests are independent
        of whatever state the dashboard left in input.json."""
        cfg = copy.deepcopy(db._backup_cfg)
        cfg["scenario_parameters"]["states"] = "1"
        cfg["scenario_parameters"]["measurements"] = "4"
        cfg["scenario_parameters"]["num_states"] = 4
        target = os.path.join(REPO_ROOT, "input", "input.json")
        with open(target, "w") as f:
            json.dump(cfg, f, indent=4)

    def route_plot_trajectories_e_real_run(self, src_cwd, monkeypatch, tmp_path):
        """Runs actual OMS + KF and returns 6 figures (always forces a fresh sim run)."""
        self._write_valid_config()
        # Point _CACHE_FILE to a temp path so the disk cache never pre-populates _sim_cache
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)
        figs = db.plot_trajectories_E(db.sliderMax)
        assert len(figs) == 6

    def route_plot_trajectories_e_sim_cache_hit_different_index(self, src_cwd, monkeypatch, tmp_path):
        """Second call with different index hits sim-cache but not fig-cache."""
        self._write_valid_config()
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)
        db.plot_trajectories_E(db.sliderMax)  # populate sim_cache and fig_cache
        figs = db.plot_trajectories_E(max(1, db.sliderMax - 1))  # sim hit, new fig_key
        assert len(figs) == 6

    def route_plot_trajectories_e_fig_cache_hit(self, src_cwd, monkeypatch, tmp_path):
        """Repeated call with same index returns from figure cache."""
        self._write_valid_config()
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)
        db.plot_trajectories_E(db.sliderMax)  # first: sim + fig populated
        db.plot_trajectories_E(db.sliderMax)  # second: sim hit, fig hit
        figs = db.plot_trajectories_E(db.sliderMax)  # third: both caches hit
        assert len(figs) == 6

    def route_plot_trajectories_e_measurements_not_found(self, src_cwd, monkeypatch, tmp_path):
        """FileNotFoundError raised when measurements.ftr is missing after simulation."""
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)

        def _raise(path, *args, **kwargs):
            raise FileNotFoundError(f"No such file: {path}")

        monkeypatch.setattr(pd, "read_feather", _raise)
        with pytest.raises(FileNotFoundError, match="Check output directory"):
            db.plot_trajectories_E(1)


# =========================================================================
# generate_json callback — 3 branches
# =========================================================================

class RoutesGenerateJson:

    def _valid_args(self):
        return (
            1, 100, 1, 4, "1", "4",          # n_clicks, steps, timeStep, numStates, state, meas
            0.1, 0.1, 0.1, 0.001,            # noise actual + process
            0.1, 0.1, 0.1, 0.1, 0.1,         # noise guess
            3.0, 70.0, -70.0, 0.5, 0.5,      # state guess
            200.0, 200.0, 200.0, 10.0, 10.0, # cov guess
            0.0, 0.0, 1.0, 0.0, 30.0, 0.0, 0.0,  # sensor initial
            2.5, 0.0, 1.0, 0.0, 35.0, 0.0, 0.0,  # target initial
            "FreeTraj", "FreeTraj", 0.5, 0.6,     # traj + turnrates
        )

    def route_generate_json_prevent_update_no_clicks(self, src_cwd):
        with pytest.raises(dash.exceptions.PreventUpdate):
            db.generate_json(None, *self._valid_args()[1:])

    def route_generate_json_prevent_update_zero_steps(self, src_cwd):
        args = list(self._valid_args())
        args[0] = 1   # n_clicks = 1
        args[1] = 0   # steps = 0 → prevent update
        with pytest.raises(dash.exceptions.PreventUpdate):
            db.generate_json(*args)

    def route_generate_json_writes_file(self, src_cwd):
        """Calls generate_json with valid args; verifies input.json is updated."""
        result = db.generate_json(*self._valid_args())
        assert result == "JSON file generated successfully"
        # Verify the written file is valid JSON
        with open(os.path.join("..", "input", "input.json")) as f:
            written = json.load(f)
        assert written["simulation_parameters"]["steps"] == 100

    def route_generate_json_range_state_writes_num_states_1(self, src_cwd):
        """state='0' (range) must write num_states=1 regardless of the UI num-states field."""
        args = list(self._valid_args())
        args[3] = 4   # UI num-states field left at 4 (wrong — simulates user not updating it)
        args[4] = "0" # state type = range
        args[5] = "0" # meas type = range
        result = db.generate_json(*args)
        assert result == "JSON file generated successfully"
        with open(os.path.join("..", "input", "input.json")) as f:
            written = json.load(f)
        assert written["scenario_parameters"]["num_states"] == 1

    def route_generate_json_validation_negative_steps(self, src_cwd):
        """steps < 1 (but != 0) triggers validation error (bypasses PreventUpdate)."""
        args = list(self._valid_args())
        args[1] = -1  # steps = -1: not caught by steps==0 guard, fails int < 1 check
        result = db.generate_json(*args)
        assert result.startswith("Validation error:")

    def route_generate_json_validation_negative_noise(self, src_cwd):
        """Negative noise value triggers validation error return."""
        args = list(self._valid_args())
        args[6] = -0.1  # rangeNoiseActual < 0
        result = db.generate_json(*args)
        assert result.startswith("Validation error:")

    def route_generate_json_validation_bad_timestep(self, src_cwd):
        """Non-positive timeStep triggers validation error return."""
        args = list(self._valid_args())
        args[2] = 0.0  # timeStep = 0
        result = db.generate_json(*args)
        assert result.startswith("Validation error:")


# =========================================================================
# run_simulation callback — 3 branches
# =========================================================================

class RoutesRunSimulation:

    def route_run_simulation_prevent_update(self, src_cwd):
        with pytest.raises(dash.exceptions.PreventUpdate):
            db.run_simulation(None, db.sliderMax, [], None, None, None, None)

    def route_run_simulation_no_freeze(self, src_cwd, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)
        figs = db.run_simulation(1, db.sliderMax, [], None, None, None, None)
        assert len(figs) == 6

    def route_run_simulation_freeze_axes_with_limits(self, src_cwd, monkeypatch, tmp_path):
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)
        figs = db.run_simulation(1, db.sliderMax, ["freeze"], -500, 500, -500, 500)
        assert len(figs) == 6

    def route_run_simulation_freeze_axes_no_limits(self, src_cwd, monkeypatch, tmp_path):
        """freeze=True but limit inputs are None → range stays None."""
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))
        monkeypatch.setattr(db, "_sim_cache", {})
        monkeypatch.setattr(db, "_fig_cache", {})
        monkeypatch.setattr(db, "_disk_cache_loaded", False)
        figs = db.run_simulation(1, db.sliderMax, ["freeze"], None, None, None, None)
        assert len(figs) == 6


# =========================================================================
# update_slider_max callback — 2 branches
# =========================================================================

class RoutesUpdateSliderMax:

    def route_update_slider_max_none_input(self):
        """steps_value is None → PreventUpdate."""
        with pytest.raises(dash.exceptions.PreventUpdate):
            db.update_slider_max(None)

    def route_update_slider_max_small(self):
        """max_value < 10 → step_size = 1."""
        max_val, marks, val = db.update_slider_max(9)
        assert max_val == 9
        assert 1 in marks

    def route_update_slider_max_large(self):
        """max_value >= 10 → step_size = max_value // 10."""
        max_val, marks, val = db.update_slider_max(100)
        assert max_val == 100
        assert 10 in marks


# =========================================================================
# _check_environment — passes and failure (GAP-O1)
# =========================================================================

class RoutesCheckEnvironment:

    def route_check_environment_passes(self):
        """_check_environment does not raise when all required files are present."""
        db._check_environment()  # must not raise

    def route_check_environment_fails(self, monkeypatch):
        """_check_environment raises FileNotFoundError when a required file is missing."""
        monkeypatch.setattr(db.os.path, "exists", lambda p: False)
        with pytest.raises(FileNotFoundError, match="Required configuration files missing"):
            db._check_environment()


# =========================================================================
# _validate_startup_config — success and silent failure (GAP-Q1)
# =========================================================================

class RoutesValidateStartupConfig:

    def route_validate_startup_config_valid(self):
        """Valid config passes without raising (logs nothing at error level)."""
        cfg = load_input_json()
        db._validate_startup_config(cfg)  # must not raise

    def route_validate_startup_config_invalid(self, caplog):
        """Invalid config logs an error but does not raise."""
        import logging
        with caplog.at_level(logging.ERROR, logger="dashboard"):
            db._validate_startup_config({})
        assert any("validation failed" in r.message.lower() for r in caplog.records)


# =========================================================================
# _load_disk_cache / _save_disk_cache (GAP-O3)
# =========================================================================

class RoutesDiskCache:

    def route_disk_cache_load_nonexistent(self, monkeypatch, tmp_path):
        """_load_disk_cache returns {} when the cache file does not exist."""
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "nonexistent.pkl"))
        result = db._load_disk_cache()
        assert result == {}

    def route_disk_cache_load_valid(self, monkeypatch, tmp_path):
        """_load_disk_cache returns the persisted dict when the file exists."""
        cache_file = str(tmp_path / "cache.pkl")
        test_data = {"sim_key_abc": ("payload",)}
        with open(cache_file, "wb") as f:
            pickle.dump(test_data, f)
        monkeypatch.setattr(db, "_CACHE_FILE", cache_file)
        result = db._load_disk_cache()
        assert result == test_data

    def route_disk_cache_save_and_load(self, monkeypatch, tmp_path):
        """Save then load roundtrip produces identical data."""
        cache_file = str(tmp_path / "roundtrip.pkl")
        monkeypatch.setattr(db, "_CACHE_FILE", cache_file)
        payload = {"key": [1, 2, 3]}
        db._save_disk_cache(payload)
        result = db._load_disk_cache()
        assert result == payload

    def route_disk_cache_save_fails_silently(self, monkeypatch, tmp_path):
        """_save_disk_cache swallows exceptions and does not propagate them."""
        monkeypatch.setattr(db, "_CACHE_FILE", str(tmp_path / "cache.pkl"))

        def _raise(obj, f):
            raise OSError("disk full")

        monkeypatch.setattr(pickle, "dump", _raise)
        db._save_disk_cache({"k": "v"})  # must not raise

    def route_disk_cache_load_corrupt(self, monkeypatch, tmp_path):
        """_load_disk_cache returns {} when the file contains invalid pickle bytes."""
        cache_file = str(tmp_path / "corrupt.pkl")
        with open(cache_file, "wb") as f:
            f.write(b"not valid pickle bytes!!!")
        monkeypatch.setattr(db, "_CACHE_FILE", cache_file)
        result = db._load_disk_cache()
        assert result == {}


# =========================================================================
# run_simulation graceful exception handling (GAP-O6)
# =========================================================================

class RoutesRunSimulationException:

    def route_run_simulation_exception_returns_empty_figs(self, monkeypatch):
        """Exception in plot_trajectories_E returns 6 empty figures instead of crashing."""

        def _fail(idx):
            raise RuntimeError("simulated pipeline failure")

        monkeypatch.setattr(db, "plot_trajectories_E", _fail)
        figs = db.run_simulation(1, db.sliderMax, [], None, None, None, None)
        assert len(figs) == 6
        # All figures should be empty (no traces)
        for fig in figs:
            assert len(fig.data) == 0
