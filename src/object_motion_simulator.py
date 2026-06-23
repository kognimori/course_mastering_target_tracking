"""Simulates sensor and target motion and generates synthetic measurements.

Reads configuration from ``input/input.json``, propagates trajectories using
:class:`~trajectory.Trajectory`, and writes output Feather files to
``output/``.
"""
import os
import logging
import json
from copy import deepcopy
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from numpy import zeros, sin, cos, sqrt, arctan2, radians, degrees, random
from numpy import set_printoptions

set_printoptions(suppress=True, precision=2)

from trajectory import Trajectory
from config_schema import validate_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(_SRC_DIR, "..", "input")
_OUTPUT_DIR = os.path.join(_SRC_DIR, "..", "output")


class MeasurementDict(TypedDict):
    """Shape contract for the per-step measurement dict (GAP-Q5)."""

    timestamps: np.ndarray
    range: np.ndarray
    azimuth: np.ndarray
    doppler: np.ndarray
    azimuthrate: np.ndarray
    rangex: np.ndarray
    rangey: np.ndarray
    vrelx: np.ndarray
    vrely: np.ndarray


class ObjectMotionSimulator:
    """Orchestrates trajectory generation, measurement simulation, and I/O.

    Workflow: ``__init__`` → :meth:`setup` → :meth:`time_loop` →
    :meth:`write_files` → optional :meth:`plot_trajectories`.  The full
    pipeline is available via :meth:`main`.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Load scenario configuration and measurement template from disk.

        Reads ``input/input.json`` (or *config_path*) and validates the
        configuration against the JSON schema.

        @param config_path: Optional explicit path to a configuration JSON file.
            When *None* (default) the canonical ``input/input.json`` is used.
        @type config_path: str | None
        @raises jsonschema.ValidationError: If the configuration is invalid.
        """
        path = config_path if config_path is not None else os.path.join(_INPUT_DIR, "input.json")
        with open(path, "r") as file:
            self.inputFile: dict[str, Any] = json.loads(file.read())
        validate_config(self.inputFile)

        with open(os.path.join(_INPUT_DIR, "measurements.json"), "r") as file:
            self.measurements: MeasurementDict = json.loads(file.read())  # type: ignore[assignment]

        self.dt: float = self.inputFile["simulation_parameters"]["time_step"]
        self.steps: int = self.inputFile["simulation_parameters"]["steps"]
        self.status: bool = False

    def create_3dof_entities(self) -> None:
        """Instantiate sensor and target :class:`~trajectory.Trajectory` objects.

        Reads ``input/object_motion_parameters.json`` for the state-name
        template, then constructs both entities from the trajectory-selection
        config.
        """
        with open(os.path.join(_INPUT_DIR, "object_motion_parameters.json"), "r") as file:
            objectMotionTemplate = json.loads(file.read())

        type_ = self.inputFile["trajectory_parameters"]["selection"]["sensor"]
        pars = self.inputFile["trajectory_parameters"][
            self.inputFile["trajectory_parameters"]["selection"]["sensor"]
        ]["sensor"]
        self.sensor = Trajectory(type_, pars, self.steps, objectMotionTemplate)

        type_ = self.inputFile["trajectory_parameters"]["selection"]["target"]
        pars = self.inputFile["trajectory_parameters"][
            self.inputFile["trajectory_parameters"]["selection"]["target"]
        ]["target"]
        self.target = Trajectory(type_, pars, self.steps, objectMotionTemplate)

    def create_measurements(self) -> None:
        """Zero-initialise measurement arrays for each measurement key."""
        for key in self.measurements.keys():
            self.measurements[key] = zeros([self.steps, 1], dtype=float)

    def init_3dof_entities(self) -> None:
        """Apply initial conditions to sensor and target from config."""
        states = self.inputFile["initial_states"]["sensor"]
        self.sensor.initialise(states)

        states = self.inputFile["initial_states"]["target"]
        self.target.initialise(states)

    def init_primary_measurements(self) -> None:
        """Compute noiseless primary measurements at t=0 from initial entity states."""
        Rx = self.target.entity.states["x"][0, 0] - self.sensor.entity.states["x"][0, 0]
        Ry = self.target.entity.states["y"][0, 0] - self.sensor.entity.states["y"][0, 0]
        R = sqrt(Rx ** 2 + Ry ** 2)
        Vx = self.target.entity.states["vx"][0, 0]
        Vy = self.target.entity.states["vy"][0, 0]
        # Doppler is the radial component of the target velocity RELATIVE to the
        # sensor.  Subtracting sensor velocity is essential for moving sensors;
        # the Kalman filter's measurement model mirrors this convention.
        Vx_sensor = self.sensor.entity.states["vx"][0, 0]
        Vy_sensor = self.sensor.entity.states["vy"][0, 0]
        self.measurements["timestamps"][0, 0] = 0.0
        self.measurements["range"][0, 0] = sqrt(Rx ** 2 + Ry ** 2)
        self.measurements["azimuth"][0, 0] = arctan2(Ry, Rx)
        self.measurements["doppler"][0, 0] = (Rx * (Vx - Vx_sensor) + Ry * (Vy - Vy_sensor)) / R
        self.measurements["azimuthrate"][0, 0] = 0.0

    def update_sensor_states(self, t: int) -> None:
        """Propagate the sensor trajectory to time step *t*.

        @param t: Current time index.
        @type t: int
        """
        self.sensor.update(t, self.dt)

    def update_target_states(self, t: int) -> None:
        """Propagate the target trajectory to time step *t*.

        @param t: Current time index.
        @type t: int
        """
        self.target.update(t, self.dt)

    def update_primary_measurements(self, t: int) -> None:
        """Compute noiseless primary measurements at time step *t*.

        @param t: Current time index.
        @type t: int
        """
        Rx = self.target.entity.states["x"][t, 0] - self.sensor.entity.states["x"][t, 0]
        Ry = self.target.entity.states["y"][t, 0] - self.sensor.entity.states["y"][t, 0]
        Vx = self.target.entity.states["vx"][t, 0]
        Vy = self.target.entity.states["vy"][t, 0]
        # Doppler = radial component of target velocity relative to sensor.
        Vx_sensor = self.sensor.entity.states["vx"][t, 0]
        Vy_sensor = self.sensor.entity.states["vy"][t, 0]
        R = sqrt(Rx ** 2 + Ry ** 2)
        self.measurements["range"][t, 0] = sqrt(Rx ** 2 + Ry ** 2)
        self.measurements["azimuth"][t, 0] = arctan2(Ry, Rx)
        self.measurements["doppler"][t, 0] = (Rx * (Vx - Vx_sensor) + Ry * (Vy - Vy_sensor)) / R
        self.measurements["azimuthrate"][t, 0] = 0.0

    def add_noise_primary_measurements(self) -> None:
        """Add Gaussian noise to all primary measurement channels.

        Azimuth noise standard deviation is treated as degrees and converted
        to radians for sampling without mutating ``self.inputFile`` (GAP-F1).
        Per-channel maximum absolute noise is stored in ``self.measNoiseMinMax``.
        """
        # Build a local copy so self.inputFile is not mutated (GAP-F1)
        raw_noises = self.inputFile["scenario_parameters"]["meas_noise_actual"]
        noises = {
            k: (float(radians(v)) if k == "azimuth" else float(v))
            for k, v in raw_noises.items()
        }
        measNoises = {key: noises[key] * random.randn(self.steps) for key in noises.keys()}

        self.measNoiseMinMax: dict[str, float] = {}
        for key in measNoises.keys():
            if key != "azimuth":
                self.measNoiseMinMax[key] = max(abs(measNoises[key]))
            else:
                self.measNoiseMinMax[key] = degrees(max(abs(measNoises[key])))
        for i in range(self.steps):
            for key in self.measurements.keys():
                if key not in ["timestamps", "rangex", "rangey", "vrelx", "vrely", "azimuthrate"]:
                    self.measurements[key][i, 0] += measNoises[key][i]

    def update_derived_measurements(self) -> None:
        """Compute ``rangex`` and ``rangey`` from range and azimuth for all steps."""
        for t in range(self.steps):
            self.measurements["rangex"][t, 0] = (
                self.measurements["range"][t, 0] * cos(self.measurements["azimuth"][t, 0])
            )
            self.measurements["rangey"][t, 0] = (
                self.measurements["range"][t, 0] * sin(self.measurements["azimuth"][t, 0])
            )

    def update_measurements(self, t: int) -> None:
        """Advance timestamp and call :meth:`update_primary_measurements`.

        @param t: Current time index (must be >= 1).
        @type t: int
        """
        self.measurements["timestamps"][t, 0] = self.measurements["timestamps"][t - 1, 0] + self.dt
        self.update_primary_measurements(t)

    def write_sensor_states(self) -> None:
        """Write sensor trajectory DataFrame to ``output/sensor_trajectory.ftr``."""
        self.sensor.write_states(os.path.join(_OUTPUT_DIR, "sensor_trajectory.ftr"))

    def write_target_states(self) -> None:
        """Write target trajectory DataFrame to ``output/target_trajectory.ftr``."""
        self.target.write_states(os.path.join(_OUTPUT_DIR, "target_trajectory.ftr"))

    def write_measurements(self) -> None:
        """Flatten measurement arrays into a DataFrame and write to Feather.

        Persists the full measurement set (all channels) to
        ``output/measurements.ftr``.
        """
        for key in self.measurements.keys():
            self.measurements[key] = list(self.measurements[key].flatten())
        self.measurementsDF = pd.DataFrame.from_dict(self.measurements)
        self.measurementsDF.to_feather(os.path.join(_OUTPUT_DIR, "measurements.ftr"))

    def _write_chunked(self, chunk_size: int) -> None:
        """Write output files in chunks via pyarrow to reduce peak write-phase memory (GAP-P3).

        Instead of serialising the entire state arrays at once, each entity's
        states are split into *chunk_size*-row batches and concatenated into a
        single Feather file, limiting per-batch arrow buffer overhead.

        @param chunk_size: Maximum number of rows per arrow batch.
        @type chunk_size: int
        """
        import pyarrow as pa
        import pyarrow.feather as pf

        def _write_entity(entity: Any, path: str) -> None:
            keys = list(entity.states.keys())
            n = entity.states[keys[0]].shape[0]
            batches = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                batch = {k: entity.states[k][start:end, 0] for k in keys}
                batches.append(pa.table(batch))
            table = pa.concat_tables(batches)
            pf.write_feather(table, path)
            entity.statesDF = table.to_pandas()

        _write_entity(self.sensor.entity, os.path.join(_OUTPUT_DIR, "sensor_trajectory.ftr"))
        _write_entity(self.target.entity, os.path.join(_OUTPUT_DIR, "target_trajectory.ftr"))

        # Write measurements in chunks
        keys = list(self.measurements.keys())
        n = self.measurements[keys[0]].shape[0]
        batches = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = {k: self.measurements[k][start:end, 0] for k in keys}
            batches.append(pa.table(batch))
        table = pa.concat_tables(batches)
        pf.write_feather(table, os.path.join(_OUTPUT_DIR, "measurements.ftr"))
        self.measurementsDF = table.to_pandas()
        # Convert to list form for interface consistency with normal write path
        for k in keys:
            self.measurements[k] = list(self.measurements[k].flatten())

    def setup(self) -> None:
        """Run the initialisation sequence: create entities, init states and measurements."""
        self.create_3dof_entities()
        self.create_measurements()
        self.init_3dof_entities()
        self.init_primary_measurements()

    def time_loop(self) -> None:
        """Propagate all entities and measurements over the full simulation horizon."""
        log_interval = max(1, self.steps // 10)
        for t in range(1, self.steps):
            if t % log_interval == 0:
                logger.info("OMS step %d/%d (%.0f%%)", t, self.steps, 100 * t / self.steps)
            self.update_sensor_states(t)
            self.update_target_states(t)
            self.update_measurements(t)
        self.add_noise_primary_measurements()
        self.update_derived_measurements()

    def write_files(self, chunk_size: int | None = None) -> None:
        """Write all output files.

        @param chunk_size: When set, uses chunked pyarrow writes to reduce
            peak write-phase memory usage (GAP-P3).  When *None* (default)
            uses the standard per-entity write path.
        @type chunk_size: int | None
        """
        if chunk_size is None:
            self.write_sensor_states()
            self.write_target_states()
            self.write_measurements()
        else:
            self._write_chunked(chunk_size)

    def plot_trajectories(self) -> None:
        """Render sensor and target Cartesian trajectories in a Plotly figure."""
        trace1 = go.Scatter(
            x=self.target.entity.statesDF['x'],
            y=self.target.entity.statesDF['y'],
            mode='lines',
            name='Target Trajectory',
        )
        trace2 = go.Scatter(
            x=self.sensor.entity.statesDF['x'],
            y=self.sensor.entity.statesDF['y'],
            mode='lines',
            name='Ego Trajectory',
        )

        fig = go.Figure()
        fig.add_trace(trace1)
        fig.add_trace(trace2)

        fig.update_layout(
            title='Trajectories',
            xaxis_title='x',
            yaxis_title='y',
        )

        fig.show()

    def main(self, screenprint: bool = False, plot: bool = False, chunk_size: int | None = None) -> None:
        """Execute the complete simulation pipeline.

        @param screenprint: If ``True``, log entity states and measurements
            at INFO level after the run.
        @type screenprint: bool
        @param plot: If ``True``, render the trajectory plot after writing
            output files.
        @type plot: bool
        @param chunk_size: When set, use chunked file writes to reduce peak
            write-phase memory (GAP-P3).
        @type chunk_size: int | None
        """
        self.setup()
        self.time_loop()
        self.write_files(chunk_size=chunk_size)
        if screenprint:
            logger.info("Sensor states:\n%s", self.sensor.entity.statesDF)
            logger.info("Target states:\n%s", self.target.entity.statesDF)
            logger.info("Measurements:\n%s", self.measurementsDF)
        if plot:
            self.plot_trajectories()
        self.status = True


if __name__ == "__main__":  # pragma: no cover

    set_printoptions(suppress=True, precision=2)

    oms = ObjectMotionSimulator()
    oms.main(screenprint=False, plot=False)
