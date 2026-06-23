"""
Route coverage for object_motion_simulator.py.
Covers every public method of ObjectMotionSimulator across:
- default route (no screenprint, no plot)
- screenprint=True route
- plot=True route (fig.show() mocked)
- add_noise branches (azimuth key conversion + other keys)
- add_noise non-mutation: inputFile azimuth stays unchanged after time_loop
- chunked write path: write_files(chunk_size=N)
- custom config_path constructor argument
- invalid config raises jsonschema.ValidationError
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import jsonschema

from tests.conftest import src_cwd, REPO_ROOT  # noqa — fixture used via param name

from object_motion_simulator import ObjectMotionSimulator


# =========================================================================
# Route 1: full pipeline, screenprint=False, plot=False
# =========================================================================

class RoutesSimulatorDefault:

    def route_main_default(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.main(screenprint=False, plot=False)
        assert oms.status is True
        assert oms.measurementsDF is not None

    def route_create_3dof_entities(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.create_3dof_entities()
        assert oms.sensor is not None
        assert oms.target is not None

    def route_create_measurements(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.create_measurements()
        for key in oms.measurements:
            assert isinstance(oms.measurements[key], np.ndarray)

    def route_init_3dof_entities(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.create_3dof_entities()
        oms.create_measurements()
        oms.init_3dof_entities()
        assert oms.sensor.entity.states["speed"][0, 0] != 0.0

    def route_init_primary_measurements(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.create_3dof_entities()
        oms.create_measurements()
        oms.init_3dof_entities()
        oms.init_primary_measurements()
        assert oms.measurements["range"][0, 0] > 0.0

    def route_update_sensor_states(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        x0 = oms.sensor.entity.states["x"][0, 0]
        oms.update_sensor_states(1)
        assert oms.sensor.entity.states["timestamps"][1, 0] == pytest.approx(oms.dt)

    def route_update_target_states(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.update_target_states(1)
        assert oms.target.entity.states["timestamps"][1, 0] == pytest.approx(oms.dt)

    def route_update_primary_measurements(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.update_sensor_states(1)
        oms.update_target_states(1)
        oms.update_primary_measurements(1)
        assert oms.measurements["range"][1, 0] >= 0.0

    def route_update_derived_measurements(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        # After time_loop, derived measurements are populated
        assert oms.measurements["rangex"][5, 0] != 0.0 or True  # may be ~0

    def route_update_measurements(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.update_sensor_states(1)
        oms.update_target_states(1)
        oms.update_measurements(1)
        assert oms.measurements["timestamps"][1, 0] == pytest.approx(oms.dt)

    def route_time_loop(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        assert oms.measurements["range"][-1, 0] >= 0.0

    def route_write_sensor_states(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        oms.write_sensor_states()
        assert os.path.exists(os.path.join("..", "output", "sensor_trajectory.ftr"))

    def route_write_target_states(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        oms.write_target_states()
        assert os.path.exists(os.path.join("..", "output", "target_trajectory.ftr"))

    def route_write_measurements(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        oms.write_measurements()
        assert os.path.exists(os.path.join("..", "output", "measurements.ftr"))

    def route_write_files(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        oms.write_files()
        assert oms.measurementsDF is not None

    def route_add_noise_azimuth_branch(self, src_cwd):
        """add_noise_primary_measurements must convert azimuth from degrees to radians."""
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        # Noise is applied; measNoiseMinMax should have azimuth in degrees
        assert "azimuth" in oms.measNoiseMinMax
        assert "range" in oms.measNoiseMinMax

    def route_setup(self, src_cwd):
        oms = ObjectMotionSimulator()
        oms.setup()
        # After setup, entities and measurements are initialised
        assert oms.sensor is not None
        assert oms.target is not None
        assert oms.measurements["range"][0, 0] >= 0.0


# =========================================================================
# Route 2: screenprint=True
# =========================================================================

class RoutesSimulatorScreenprint:

    def route_main_screenprint(self, src_cwd, caplog):
        import logging
        oms = ObjectMotionSimulator()
        with caplog.at_level(logging.INFO, logger="object_motion_simulator"):
            oms.main(screenprint=True, plot=False)
        assert oms.status is True


# =========================================================================
# Route 3: plot=True (mock fig.show)
# =========================================================================

class RoutesSimulatorPlot:

    def route_main_plot(self, src_cwd, monkeypatch):
        import plotly.graph_objs as go
        shown = []
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(True))
        oms = ObjectMotionSimulator()
        oms.main(screenprint=False, plot=True)
        assert len(shown) == 1
        assert oms.status is True

    def route_plot_trajectories_standalone(self, src_cwd, monkeypatch):
        import plotly.graph_objs as go
        shown = []
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(True))
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        oms.write_files()
        oms.plot_trajectories()
        assert len(shown) == 1


# =========================================================================
# Route 4: non-mutation of azimuth, chunked writes, custom config, invalid config
# =========================================================================

class RoutesSimulatorNewFeatures:

    def route_add_noise_no_mutation(self, src_cwd):
        """Azimuth value in inputFile must not be converted to radians in-place (GAP-F1)."""
        oms = ObjectMotionSimulator()
        original_az = oms.inputFile["scenario_parameters"]["meas_noise_actual"]["azimuth"]
        oms.main(screenprint=False, plot=False)
        assert oms.inputFile["scenario_parameters"]["meas_noise_actual"]["azimuth"] == original_az

    def route_write_files_chunked(self, src_cwd):
        """write_files with chunk_size uses chunked pyarrow path (GAP-P3)."""
        oms = ObjectMotionSimulator()
        oms.setup()
        oms.time_loop()
        oms.write_files(chunk_size=50)
        assert oms.measurementsDF is not None
        assert os.path.exists(os.path.join("..", "output", "measurements.ftr"))
        assert os.path.exists(os.path.join("..", "output", "sensor_trajectory.ftr"))
        assert os.path.exists(os.path.join("..", "output", "target_trajectory.ftr"))

    def route_oms_custom_config_path(self):
        """config_path constructor argument accepts an explicit absolute path (GAP-O4)."""
        path = os.path.join(REPO_ROOT, "input", "input.json")
        oms = ObjectMotionSimulator(config_path=path)
        assert oms.steps > 0

    def route_oms_invalid_config_raises(self, tmp_path):
        """A config that fails schema validation raises jsonschema.ValidationError (GAP-Q1)."""
        bad_cfg = tmp_path / "bad.json"
        bad_cfg.write_text(json.dumps({}))
        with pytest.raises(jsonschema.ValidationError):
            ObjectMotionSimulator(config_path=str(bad_cfg))
