"""Command-line entry point for the object tracking pipeline.

Ties together :class:`~object_motion_simulator.ObjectMotionSimulator` and
:class:`~kalman_filter.KalmanFilter`, then dispatches to the appropriate
trajectory-plot function based on the active state-type configuration.
"""
import argparse
import logging

import pandas as pd
from numpy import set_printoptions, sqrt

from object_motion_simulator import ObjectMotionSimulator
from kalman_filter import KalmanFilter

import plotly.graph_objs as go

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the pipeline (GAP-O4).

    @param args: Argument list to parse. When *None*, reads ``sys.argv[1:]``.
    @type args: list[str] | None
    @return: Parsed namespace with attributes ``config`` and ``no_plot``.
    @rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Object tracking pipeline: simulate, filter, and visualise.",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to input.json (default: input/input.json relative to src/).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the trajectory plot after the pipeline completes.",
    )
    return parser.parse_args(args)


def plot_trajectories_E(allStatesT: pd.DataFrame, allStatesE: pd.DataFrame, estimates: pd.DataFrame) -> None:
    """Render target, ego, and estimated Cartesian trajectories.

    @param allStatesT: Ground-truth target state history (columns: x, y, …).
    @type allStatesT: pd.DataFrame
    @param allStatesE: Ground-truth ego/sensor state history.
    @type allStatesE: pd.DataFrame
    @param estimates: Kalman filter position estimates (columns: x, y, timestamps).
    @type estimates: pd.DataFrame
    @return: None; opens an interactive Plotly browser window.
    @rtype: None
    """
    trace1 = go.Scatter(x=allStatesT['x'], y=allStatesT['y'], mode='lines', name='Target Trajectory')
    trace2 = go.Scatter(x=allStatesE['x'], y=allStatesE['y'], mode='lines', name='Ego Trajectory')
    trace3 = go.Scatter(x=estimates['x'], y=estimates['y'], mode='markers', name='Estimated Trajectory')
    trace3['marker']['size'] = 2

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)

    fig.update_layout(
        title='Trajectories',
        xaxis_title='x',
        yaxis_title='y'
    )

    fig.show()


def plot_trajectories_S(allStatesT: pd.DataFrame, allStatesE: pd.DataFrame, estimates: pd.DataFrame) -> None:
    """Render actual vs estimated relative range for range-only scenarios.

    @param allStatesT: Ground-truth target state history.
    @type allStatesT: pd.DataFrame
    @param allStatesE: Ground-truth ego/sensor state history.
    @type allStatesE: pd.DataFrame
    @param estimates: Kalman filter range estimates (columns: range, timestamps).
    @type estimates: pd.DataFrame
    @return: None; opens an interactive Plotly browser window.
    @rtype: None
    """
    actual = sqrt(
        (allStatesT['x'].to_numpy() - allStatesE['x'].to_numpy()) ** 2
        + (allStatesT['y'].to_numpy() - allStatesE['y'].to_numpy()) ** 2
    )
    trace1 = go.Scatter(x=estimates['timestamps'], y=estimates['range'], mode='markers', name='Estimate')
    trace2 = go.Scatter(x=allStatesE["timestamps"], y=actual, mode='lines', name='Actual')
    trace1['marker']['size'] = 2

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)

    fig.update_layout(
        title='Actual and Estimate',
        xaxis_title='time',
        yaxis_title='range'
    )

    fig.show()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    set_printoptions(suppress=True, precision=5)

    cli = _parse_args()

    oms = ObjectMotionSimulator(config_path=cli.config)
    oms.main(screenprint=False, plot=False)

    kf = KalmanFilter(config_path=cli.config)
    estimates = kf.main()
    logger.info("\n%s", estimates)

    if not cli.no_plot:
        states = kf.inputFile["scenario_parameters"]["stateTypes"][kf.inputFile["scenario_parameters"]["states"]]
        if states == ["vx", "vy", "x", "y"]:
            plot_trajectories_E(oms.target.entity.statesDF, oms.sensor.entity.statesDF, estimates)
        elif states == ["range"]:
            plot_trajectories_S(oms.target.entity.statesDF, oms.sensor.entity.statesDF, estimates)
