"""Dash dashboard for interactive object tracking simulation.

Provides figure-building helpers, a :class:`~dash.Dash` layout, and three
Dash callbacks: ``generate_json`` (writes updated ``input.json``),
``run_simulation`` (runs OMS + KF and renders figures), and
``update_slider_max`` (adjusts the playback slider range).

Simulation results are cached by MD5 hash of ``input.json`` content so that
slider moves do not re-run the simulation unnecessarily, and cached results
are persisted to disk so they survive server restarts (GAP-O3, GAP-F4).
"""
import copy
import hashlib
import logging
import os
import pickle
import tempfile
import time

import dash
import pandas as pd
import numpy as np
from numpy import sqrt, arctan2, degrees, maximum, cumsum, arange
from dash import html, dcc, Input, Output, State
import json

import plotly.graph_objs as go

from object_motion_simulator import ObjectMotionSimulator
from kalman_filter import KalmanFilter
from config_schema import validate_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.join(_SRC_DIR, "..", "input")
_OUTPUT_DIR = os.path.join(_SRC_DIR, "..", "output")
_CACHE_FILE = os.path.join(_SRC_DIR, "..", ".sim_cache.pkl")


# ---------------------------------------------------------------------------
# Pre-flight environment check (GAP-O1)
# ---------------------------------------------------------------------------

def _check_environment() -> None:
    """Verify required configuration files exist before the server starts.

    @raises FileNotFoundError: If any required file or directory is absent.
    """
    required = [
        os.path.join(_INPUT_DIR, "input.json"),
        os.path.join(_INPUT_DIR, "input_backup.json"),
        os.path.join(_INPUT_DIR, "measurements.json"),
        os.path.join(_INPUT_DIR, "object_motion_parameters.json"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        names = [os.path.basename(p) for p in missing]
        raise FileNotFoundError(
            f"Required configuration files missing: {names!r}. "
            "Ensure the repository is complete and input/ directory is intact."
        )
    logger.debug("Environment check passed: all required configuration files present")


# ---------------------------------------------------------------------------
# Disk-backed simulation cache (GAP-O3)
# ---------------------------------------------------------------------------

def _load_disk_cache() -> dict:
    """Load the persisted simulation cache from disk.

    Returns an empty dict if the file does not exist, is unreadable, or
    was written by a different Python version (unpickling fails silently).

    @return: Previously saved cache dict, or ``{}`` on any error.
    @rtype: dict
    """
    try:
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_disk_cache(cache: dict) -> None:
    """Persist the simulation cache to disk.

    Failures are logged as warnings and silently swallowed so a disk-full
    condition never crashes the server.

    @param cache: Cache dict to persist.
    @type cache: dict
    """
    try:
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception as exc:
        logger.warning("Failed to persist simulation cache to disk: %s", exc)


# ---------------------------------------------------------------------------
# Startup config validation (GAP-Q1)
# ---------------------------------------------------------------------------

def _validate_startup_config(cfg: dict) -> None:
    """Validate the startup configuration; log on failure but do not crash.

    @param cfg: Parsed configuration dictionary.
    @type cfg: dict
    """
    try:
        validate_config(cfg)
        logger.debug("Startup config validated OK")
    except Exception as exc:
        logger.error(
            "Startup config validation failed (simulation may be unreliable): %s", exc
        )


# ---------------------------------------------------------------------------
# Dark-theme helper
# ---------------------------------------------------------------------------

_TICK_COLOUR = "#8BA3B0"


def _apply_dark_theme(fig: go.Figure, x_title: str, y_title: str) -> None:
    """Apply dark background, white axis styling, and axis labels to *fig* in place.

    @param fig: Plotly Figure to modify.
    @type fig: go.Figure
    @param x_title: Label for the x-axis (pass empty string to leave untitled).
    @type x_title: str
    @param y_title: Label for the y-axis (pass empty string to leave untitled).
    @type y_title: str
    """
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        paper_bgcolor=plotColours,
        plot_bgcolor=plotColours,
        margin=dict(l=12, r=12, t=12, b=12),
        title_font_family="NK57 Monospace Condensed Light",
        xaxis=dict(
            tickcolor=_TICK_COLOUR,
            tickfont=dict(color=_TICK_COLOUR, family="NK57 Monospace Condensed Light"),
            title_font=dict(color=_TICK_COLOUR),
            gridcolor="#0D2236",
        ),
        yaxis=dict(
            tickcolor=_TICK_COLOUR,
            tickfont=dict(color=_TICK_COLOUR, family="NK57 Monospace Condensed Light"),
            title_font=dict(color=_TICK_COLOUR),
            gridcolor="#0D2236",
        ),
        legend=dict(font=dict(color=_TICK_COLOUR)),
        legend_font_family="NK57 Monospace Condensed Light",
        transition_easing="cubic",
    )


def get_trajectory_fig(
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Build the trajectory comparison figure.

    Dispatches on whether the active state type is range-only or Cartesian
    (vx/vy/x/y) and renders the appropriate traces with start/end markers.

    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter position estimates.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration.
    @type inputFile: dict
    @param measurements: Measurement DataFrame (kept for interface consistency).
    @type measurements: pd.DataFrame
    @return: Configured Plotly Figure.
    @rtype: go.Figure
    """
    if len(allStatesT) == 0:
        fig = go.Figure()
        _apply_dark_theme(fig, "", "")
        return fig

    if (inputFile["scenario_parameters"]["stateTypes"][inputFile["scenario_parameters"]["states"]] != ["range"]):
        trace1 = go.Scatter(x=allStatesT['x'], y=allStatesT['y'], mode='lines', name='T')
        trace2 = go.Scatter(x=allStatesE['x'], y=allStatesE['y'], mode='lines', name='O')
        trace3 = go.Scatter(x=estimates['x'], y=estimates['y'], mode='markers', name='E')
        trace4 = go.Scatter(x=[allStatesT['x'].iloc[0]], y=[allStatesT['y'].iloc[0]], mode='markers', name='StartT')
        trace5 = go.Scatter(x=[allStatesT['x'].iloc[-1]], y=[allStatesT['y'].iloc[-1]], mode='markers', name='EndT')
        trace6 = go.Scatter(x=[allStatesE['x'].iloc[0]], y=[allStatesE['y'].iloc[0]], mode='markers', name='StartO')
        trace7 = go.Scatter(x=[allStatesE['x'].iloc[-1]], y=[allStatesE['y'].iloc[-1]], mode='markers', name='EndO')

        trace3['marker']['size'] = 2
        trace4['marker']['size'] = 10
        trace5["marker"]["symbol"] = "x"
        trace5['marker']['size'] = 10
        trace6['marker']['size'] = 10
        trace7["marker"]["symbol"] = "x"
        trace7['marker']['size'] = 10
        fig = go.Figure()
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.add_trace(trace4)
        fig.add_trace(trace5)
        fig.add_trace(trace6)
        fig.add_trace(trace7)
        x_title = "x"
        y_title = "y"
    else:
        fig = go.Figure()
        if (inputFile["initial_states"]["target"]["speed"] == 0):
            trace1 = go.Scatter(x=allStatesT['x'], y=allStatesT['y'], mode='markers', name='T')
            trace2 = go.Scatter(x=allStatesE['x'], y=allStatesE['y'], mode='lines', name='O')

            trace6 = go.Scatter(x=[allStatesE['x'].iloc[0]], y=[allStatesE['y'].iloc[0]], mode='markers', name='StartO')
            trace7 = go.Scatter(x=[allStatesE['x'].iloc[-1]], y=[allStatesE['y'].iloc[-1]], mode='markers', name='EndO')
            trace1["marker"]["size"] = 10
            trace6['marker']['size'] = 10
            trace7["marker"]["symbol"] = "x"
            trace7['marker']['size'] = 10
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            fig.add_trace(trace6)
            fig.add_trace(trace7)
        else:
            trace1 = go.Scatter(x=allStatesT['x'], y=allStatesT['y'], mode='lines', name='T')
            trace2 = go.Scatter(x=allStatesE['x'], y=allStatesE['y'], mode='lines', name='O')
            trace4 = go.Scatter(x=[allStatesT['x'].iloc[0]], y=[allStatesT['y'].iloc[0]], mode='markers', name='StartT')
            trace5 = go.Scatter(x=[allStatesT['x'].iloc[-1]], y=[allStatesT['y'].iloc[-1]], mode='markers', name='EndT')
            trace6 = go.Scatter(x=[allStatesE['x'].iloc[0]], y=[allStatesE['y'].iloc[0]], mode='markers', name='StartO')
            trace7 = go.Scatter(x=[allStatesE['x'].iloc[-1]], y=[allStatesE['y'].iloc[-1]], mode='markers', name='EndO')
            trace4['marker']['size'] = 10
            trace5["marker"]["symbol"] = "x"
            trace5['marker']['size'] = 10
            trace6['marker']['size'] = 10
            trace7["marker"]["symbol"] = "x"
            trace7['marker']['size'] = 10
            fig.add_trace(trace1)
            fig.add_trace(trace2)
            fig.add_trace(trace4)
            fig.add_trace(trace5)
            fig.add_trace(trace6)
            fig.add_trace(trace7)
        x_title = "x"
        y_title = "y"

    _apply_dark_theme(fig, x_title, y_title)
    return fig


def get_error_dist_fig(
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    maxNoise: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Build the position-error distribution figure.

    Plots time-averaged position error and standard deviation (from the
    diagonal of the Kalman filter covariance) for each active state.

    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter estimates including covariance columns ``P_*``.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration.
    @type inputFile: dict
    @param maxNoise: Per-channel maximum absolute measurement noise.
    @type maxNoise: dict
    @param measurements: Measurement DataFrame used for range-only error.
    @type measurements: pd.DataFrame
    @return: Configured Plotly Figure.
    @rtype: go.Figure
    """
    fig = go.Figure()
    x_title = "timestamps"
    y_title = "errors"

    noise_ = []
    for k in maxNoise.keys():
        noise_.append(k + ":" + str(maxNoise[k]))

    noise_ = ", ".join(noise_)
    if (inputFile["scenario_parameters"]["stateTypes"][inputFile["scenario_parameters"]["states"]] == ["vx", "vy", "x", "y"]):
        errorX = abs(estimates["x"].to_numpy().flatten() - allStatesT["x"].to_numpy().flatten())
        errorY = abs(estimates["y"].to_numpy().flatten() - allStatesT["y"].to_numpy().flatten())
        n = arange(1, len(errorX) + 1)
        timeavgX = cumsum(errorX) / n
        timeavgY = cumsum(errorY) / n

        trace1 = go.Scatter(x=estimates['timestamps'], y=timeavgX, mode='lines', name='Avg Error X')
        trace2 = go.Scatter(x=estimates['timestamps'], y=sqrt(estimates["P_x"]), mode="lines", name="std dev X")
        trace3 = go.Scatter(x=estimates['timestamps'], y=timeavgY, mode='lines', name='Avg Error Y')
        trace4 = go.Scatter(x=estimates['timestamps'], y=sqrt(estimates["P_y"]), mode="lines", name="std dev Y")

        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.add_trace(trace4)
    elif (inputFile["scenario_parameters"]["stateTypes"][inputFile["scenario_parameters"]["states"]] == ["range"]):
        meas = measurements["range"].to_numpy()
        error = abs((meas.flatten()) - estimates["range"])
        timeavg = cumsum(error) / arange(1, len(error) + 1)

        trace1 = go.Scatter(x=estimates['timestamps'], y=timeavg, mode='lines', name='Average_error')
        trace2 = go.Scatter(x=estimates['timestamps'], y=sqrt(estimates["P_range"]), mode="lines", name="standard deviation")

        fig.add_trace(trace1)
        fig.add_trace(trace2)

    else:
        logger.warning(
            "get_error_dist_fig: unrecognised state type %r — returning empty figure",
            inputFile["scenario_parameters"]["stateTypes"][inputFile["scenario_parameters"]["states"]],
        )

    _apply_dark_theme(fig, x_title, y_title)
    fig.update_layout(title={"text": "MaxNoise - " + noise_, "x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top", "font": {"color": "white"}})
    return fig


def get_error_vel_fig(
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    maxNoise: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Build the velocity-error figure.

    Only populated for Cartesian (vx/vy/x/y) state configurations; for
    range-only states the figure is returned empty.

    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter estimates including covariance columns ``P_vx``, ``P_vy``.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration.
    @type inputFile: dict
    @param maxNoise: Per-channel maximum absolute measurement noise.
    @type maxNoise: dict
    @param measurements: Measurement DataFrame (unused here; kept for interface consistency).
    @type measurements: pd.DataFrame
    @return: Configured Plotly Figure.
    @rtype: go.Figure
    """
    fig = go.Figure()
    x_title = "timestamps"
    y_title = "errors"

    noise_ = []
    for k in maxNoise.keys():
        noise_.append(k + ":" + str(maxNoise[k]))

    noise_ = ", ".join(noise_)
    if (inputFile["scenario_parameters"]["stateTypes"][inputFile["scenario_parameters"]["states"]] == ["vx", "vy", "x", "y"]):
        errorVX = abs(estimates["vx"].to_numpy().flatten() - allStatesT["vx"].to_numpy().flatten())
        errorVY = abs(estimates["vy"].to_numpy().flatten() - allStatesT["vy"].to_numpy().flatten())
        nv = arange(1, len(errorVX) + 1)
        timeavgVX = cumsum(errorVX) / nv
        timeavgVY = cumsum(errorVY) / nv

        trace1 = go.Scatter(x=estimates['timestamps'], y=timeavgVX, mode='lines', name='Avg Error VX')
        trace2 = go.Scatter(x=estimates['timestamps'], y=sqrt(estimates["P_vx"]), mode="lines", name="std dev VX")
        trace3 = go.Scatter(x=estimates['timestamps'], y=timeavgVY, mode='lines', name='Avg Error VY')
        trace4 = go.Scatter(x=estimates['timestamps'], y=sqrt(estimates["P_vy"]), mode="lines", name="std dev VY")

        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.add_trace(trace4)

    else:
        logger.warning(
            "get_error_vel_fig: unrecognised state type %r — returning empty figure",
            inputFile["scenario_parameters"]["stateTypes"][inputFile["scenario_parameters"]["states"]],
        )

    _apply_dark_theme(fig, x_title, y_title)
    fig.update_layout(title={"text": "MaxNoise - " + noise_, "x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top", "font": {"color": "white"}})
    return fig


def get_range_measurement(
    fig: go.Figure,
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Add range traces (estimate, actual, measured) to *fig*.

    @param fig: Figure to populate.
    @type fig: go.Figure
    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter position estimates.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration (unused; kept for consistency).
    @type inputFile: dict
    @param measurements: Measurement DataFrame providing the ``range`` column.
    @type measurements: pd.DataFrame
    @return: The updated figure.
    @rtype: go.Figure
    """
    Rx = allStatesT['x'].to_numpy() - allStatesE['x'].to_numpy()
    Ry = allStatesT['y'].to_numpy() - allStatesE['y'].to_numpy()
    R = sqrt(Rx ** 2 + Ry ** 2)
    RxE = estimates['x'].to_numpy() - allStatesE['x'].to_numpy()
    RyE = estimates['y'].to_numpy() - allStatesE['y'].to_numpy()
    RE = sqrt(RxE ** 2 + RyE ** 2)
    actualRange = R
    estimatesRange = RE

    trace1 = go.Scatter(x=estimates['timestamps'], y=estimatesRange, mode='markers', name='Estimate')
    trace2 = go.Scatter(x=allStatesE["timestamps"], y=actualRange, mode='lines', name='Actual')
    trace3 = go.Scatter(x=allStatesE["timestamps"], y=measurements["range"], mode='markers', name='Meas')

    trace1['marker']['size'] = 3
    trace3['marker']['size'] = 3

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)

    _apply_dark_theme(fig, "time-steps", "range")
    return fig


def get_range_only_measurement(
    fig: go.Figure,
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Add range traces for the range-only state configuration.

    @param fig: Figure to populate.
    @type fig: go.Figure
    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter range estimates (column: ``range``).
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration (unused; kept for consistency).
    @type inputFile: dict
    @param measurements: Measurement DataFrame providing the ``range`` column.
    @type measurements: pd.DataFrame
    @return: The updated figure.
    @rtype: go.Figure
    """
    Rx = allStatesT['x'].to_numpy() - allStatesE['x'].to_numpy()
    Ry = allStatesT['y'].to_numpy() - allStatesE['y'].to_numpy()
    R = sqrt(Rx ** 2 + Ry ** 2)
    actualRange = R

    trace1 = go.Scatter(x=estimates['timestamps'], y=estimates["range"], mode='markers', name='Estimate')
    trace2 = go.Scatter(x=allStatesE["timestamps"], y=actualRange, mode='lines', name='Actual')
    trace3 = go.Scatter(x=allStatesE["timestamps"], y=measurements["range"], mode='markers', name='Meas')

    trace1['marker']['size'] = 3
    trace3['marker']['size'] = 3

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)

    _apply_dark_theme(fig, "time-steps", "range")
    return fig


def get_azimuth_measurement(
    fig: go.Figure,
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Add azimuth traces (estimate, actual, measured) to *fig*.

    All azimuth values are converted to degrees for display.

    @param fig: Figure to populate.
    @type fig: go.Figure
    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter position estimates.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration (unused; kept for consistency).
    @type inputFile: dict
    @param measurements: Measurement DataFrame providing the ``azimuth`` column (radians).
    @type measurements: pd.DataFrame
    @return: The updated figure.
    @rtype: go.Figure
    """
    Rx = allStatesT['x'].to_numpy() - allStatesE['x'].to_numpy()
    Ry = allStatesT['y'].to_numpy() - allStatesE['y'].to_numpy()
    RxE = estimates['x'].to_numpy() - allStatesE['x'].to_numpy()
    RyE = estimates['y'].to_numpy() - allStatesE['y'].to_numpy()
    actualAz = degrees(arctan2(Ry, Rx))
    estimateAz = degrees(arctan2(RyE, RxE))

    trace1 = go.Scatter(x=estimates['timestamps'], y=estimateAz, mode='markers', name='Estimate')
    trace2 = go.Scatter(x=allStatesE["timestamps"], y=actualAz, mode='lines', name='Actual')
    trace3 = go.Scatter(x=allStatesE["timestamps"], y=degrees(measurements["azimuth"].to_numpy()), mode='markers', name='Meas')

    trace1['marker']['size'] = 3
    trace3['marker']['size'] = 3

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)

    _apply_dark_theme(fig, "time-steps", "azimuth")
    return fig


def get_doppler_measurement(
    fig: go.Figure,
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    measurements: pd.DataFrame,
) -> go.Figure:
    """Add radial-velocity (Doppler) traces (estimate, actual, measured) to *fig*.

    @param fig: Figure to populate.
    @type fig: go.Figure
    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter velocity and position estimates.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration (unused; kept for consistency).
    @type inputFile: dict
    @param measurements: Measurement DataFrame providing the ``doppler`` column.
    @type measurements: pd.DataFrame
    @return: The updated figure.
    @rtype: go.Figure
    """
    Rx = allStatesT['x'].to_numpy() - allStatesE['x'].to_numpy()
    Ry = allStatesT['y'].to_numpy() - allStatesE['y'].to_numpy()
    R = maximum(sqrt(Rx ** 2 + Ry ** 2), 1e-9)
    RxE = estimates['x'].to_numpy() - allStatesE['x'].to_numpy()
    RyE = estimates['y'].to_numpy() - allStatesE['y'].to_numpy()
    RE = maximum(sqrt(RxE ** 2 + RyE ** 2), 1e-9)
    Vx = allStatesT["vx"].to_numpy()
    Vy = allStatesT["vy"].to_numpy()
    VxE = estimates["vx"].to_numpy()
    VyE = estimates["vy"].to_numpy()
    actualDop = (Rx * Vx + Ry * Vy) / R
    estimateDop = (RxE * VxE + RyE * VyE) / RE

    trace1 = go.Scatter(x=estimates['timestamps'], y=estimateDop, mode='markers', name='Estimate')
    trace2 = go.Scatter(x=allStatesE["timestamps"], y=actualDop, mode='lines', name='Actual')
    trace3 = go.Scatter(x=allStatesE["timestamps"], y=measurements["doppler"], mode='markers', name='Meas')

    trace1['marker']['size'] = 3
    trace3['marker']['size'] = 3

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)

    _apply_dark_theme(fig, "time-steps", "doppler")
    return fig


def get_measurement_figs(
    allStatesE: pd.DataFrame,
    allStatesT: pd.DataFrame,
    estimates: pd.DataFrame,
    inputFile: dict,
    measurements: pd.DataFrame,
) -> list[go.Figure]:
    """Build up to three measurement comparison figures based on active measurement type.

    Always returns exactly three figures; figures not populated for the active
    measurement type remain as empty dark-themed placeholders.

    @param allStatesE: Ego (sensor) state history.
    @type allStatesE: pd.DataFrame
    @param allStatesT: Target state history.
    @type allStatesT: pd.DataFrame
    @param estimates: Kalman filter estimates.
    @type estimates: pd.DataFrame
    @param inputFile: Parsed ``input.json`` configuration.
    @type inputFile: dict
    @param measurements: Full measurement DataFrame.
    @type measurements: pd.DataFrame
    @return: List of three Plotly Figures (range, azimuth, doppler slots).
    @rtype: list[go.Figure]
    """
    meas = inputFile["scenario_parameters"]["measurementsTypes"][inputFile["scenario_parameters"]["measurements"]]
    figs = []
    for i in range(3):
        fig = go.Figure()
        _apply_dark_theme(fig, "", "")
        figs.append(fig)

    if (meas == ["range"]):
        figs[0] = get_range_only_measurement(figs[0], allStatesE, allStatesT, estimates, inputFile, measurements)
    elif (meas == ["azimuth"]):
        figs[1] = get_azimuth_measurement(figs[1], allStatesE, allStatesT, estimates, inputFile, measurements)
    elif (meas == ["azimuth", "range"]):
        figs[0] = get_range_measurement(figs[0], allStatesE, allStatesT, estimates, inputFile, measurements)
        figs[1] = get_azimuth_measurement(figs[1], allStatesE, allStatesT, estimates, inputFile, measurements)
    elif (meas == ["azimuth", "range", "doppler"]):
        figs[0] = get_range_measurement(figs[0], allStatesE, allStatesT, estimates, inputFile, measurements)
        figs[1] = get_azimuth_measurement(figs[1], allStatesE, allStatesT, estimates, inputFile, measurements)
        figs[2] = get_doppler_measurement(figs[2], allStatesE, allStatesT, estimates, inputFile, measurements)
    elif (meas == ["rangex", "rangey", "doppler"]):
        figs[0] = get_range_measurement(figs[0], allStatesE, allStatesT, estimates, inputFile, measurements)
        figs[1] = get_azimuth_measurement(figs[1], allStatesE, allStatesT, estimates, inputFile, measurements)
        figs[2] = get_doppler_measurement(figs[2], allStatesE, allStatesT, estimates, inputFile, measurements)

    return figs


# ---------------------------------------------------------------------------
# Simulation cache (GAP-O3 — lazy disk load, GAP-F4 — MD5 key)
# ---------------------------------------------------------------------------

_sim_cache: dict[str, tuple] = {}
_fig_cache: dict[tuple, list] = {}
_disk_cache_loaded: bool = False


def _sim_cache_key() -> str:
    """Return the MD5 hash of ``input.json`` content as the simulation cache key.

    Using file content checksum (GAP-F4) avoids false cache hits caused by
    filesystem clock granularity on FAT32/exFAT filesystems that limit mtime
    resolution to 2 seconds.

    @return: Hex MD5 digest of the current ``input.json`` content.
    @rtype: str
    """
    with open(os.path.join(_INPUT_DIR, "input.json"), "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def plot_trajectories_E(index: int) -> list[go.Figure]:
    """Run or re-use cached simulation results and return six figures sliced to *index*.

    Two cache layers are used:

    * ``_sim_cache`` — keyed by MD5 of ``input.json``; holds raw DataFrames.
      Rebuilt only when the file changes. Persisted to disk between restarts (GAP-O3).
    * ``_fig_cache`` — keyed by ``(sim_key, index)``; holds the six built
      figures. Avoids re-constructing Plotly objects on every slider move.

    @param index: Upper row index for slicing all DataFrames (non-inclusive).
    @type index: int
    @return: List of six Plotly Figures: trajectory, error-dist, error-vel,
        range, azimuth, doppler.
    @rtype: list[go.Figure]
    @raises FileNotFoundError: If the measurements Feather file is absent
        (raised from :class:`~object_motion_simulator.ObjectMotionSimulator`).
    """
    global _disk_cache_loaded
    if not _disk_cache_loaded:
        loaded = _load_disk_cache()
        _sim_cache.update(loaded)
        _disk_cache_loaded = True

    key = _sim_cache_key()
    fig_key = (key, index)

    if fig_key in _fig_cache:
        logger.debug("Figure cache hit (key=%s index=%s)", key[:8], index)
        return _fig_cache[fig_key]

    if key not in _sim_cache:
        t0 = time.monotonic()
        oms = ObjectMotionSimulator()
        kf = KalmanFilter()
        oms.main(screenprint=False, plot=False)
        meas_path = os.path.join(_OUTPUT_DIR, "measurements.ftr")
        try:
            measurements_full = pd.read_feather(meas_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Measurements file not found after simulation: {meas_path!r}. "
                "Check output directory permissions."
            ) from exc
        estimates_full = kf.main()
        elapsed = time.monotonic() - t0
        logger.info("Simulation complete in %.2fs (key=%s)", elapsed, key[:8])
        maxNoise = {k: round(v, 4) for k, v in oms.measNoiseMinMax.items()}
        _sim_cache.clear()
        _fig_cache.clear()
        _sim_cache[key] = (
            oms.target.entity.statesDF,
            oms.sensor.entity.statesDF,
            estimates_full,
            maxNoise,
            kf.inputFile,
            measurements_full,
        )
        _save_disk_cache(_sim_cache)
    else:
        logger.debug("Simulation cache hit (key=%s)", key[:8])

    allStatesT_full, allStatesE_full, estimates_full, maxNoise, inputFile, measurements_full = _sim_cache[key]
    allStatesT = allStatesT_full[0:index]
    allStatesE = allStatesE_full[0:index]
    estimates = estimates_full[0:index]
    measurements = measurements_full[0:index]

    trajectory_fig = get_trajectory_fig(allStatesE, allStatesT, estimates, inputFile, measurements)
    error_dist_fig = get_error_dist_fig(allStatesE, allStatesT, estimates, inputFile, maxNoise, measurements)
    error_vel_fig = get_error_vel_fig(allStatesE, allStatesT, estimates, inputFile, maxNoise, measurements)
    meas_figs = get_measurement_figs(allStatesE, allStatesT, estimates, inputFile, measurements)
    figs = [trajectory_fig, error_dist_fig, error_vel_fig, *meas_figs]
    _fig_cache[fig_key] = figs
    return figs


# ---------------------------------------------------------------------------
# Module startup: load config and run pre-flight checks
# ---------------------------------------------------------------------------

_check_environment()

with open(os.path.join(_INPUT_DIR, "input.json"), "r") as _f:
    _startup_cfg: dict = json.load(_f)

_validate_startup_config(_startup_cfg)

with open(os.path.join(_INPUT_DIR, "input_backup.json"), "r") as _f:
    _backup_cfg: dict = json.load(_f)

stateTypes: list[dict] = [
    {"label": "_".join(v), "value": k}
    for k, v in _startup_cfg["scenario_parameters"]["stateTypes"].items()
]
measurementTypes: list[dict] = [
    {"label": "_".join(v), "value": k}
    for k, v in _startup_cfg["scenario_parameters"]["measurementsTypes"].items()
]
sliderMax: int = int(_startup_cfg["simulation_parameters"]["steps"])
trajectories = [
    {"label": "S-Traj", "value": "STraj"},
    {"label": "Free-Traj", "value": "FreeTraj"},
    {"label": "FreeTurn-Traj", "value": "FreeTurnTraj"},
]

# ---------------------------------------------------------------------------
# Colour constants — Dark Ops Terminal theme
# ---------------------------------------------------------------------------

backgroundColour = "#0A1628"       # deep navy-slate page bg
parameterSectionColour = "#0F2A3F" # panel surface, one step above page bg
plotColours = "#081420"            # darkest surface — draws eye to data
inputFieldColour = "#0D2235"       # dark-well inputs
buttonColour = "#163248"           # raised button bg

# Semantic design tokens (used in layout below)
_AMBER = "#E8A830"           # amber accent — title + active tab only (max 2 uses)
_TEXT_PRIMARY = "#E8F0F5"    # primary readable text
_TEXT_SECTION = "#8BA3B0"    # panel section headers
_TEXT_MUTED = "#6B8CA4"      # input group labels
_BORDER_COL = "#1A3A52"      # panel and component borders
_INPUT_BORDER = "#1E4060"    # input field border

# ---------------------------------------------------------------------------
# UI style constants — eliminates repeated inline dicts (GAP-Q4)
# ---------------------------------------------------------------------------

_S_INPUT_BASE = {
    "color": _TEXT_PRIMARY,
    "backgroundColor": inputFieldColour,
    "border": f"1px solid {_INPUT_BORDER}",
    "borderRadius": "4px",
    "minHeight": "28px",
    "padding": "2px 6px",
    "fontSize": "12px",
    "fontFamily": "system-ui, -apple-system, sans-serif",
}

_S_IN_WIDE = {"marginLeft": "4px", "marginRight": "4px", "width": "20%", **_S_INPUT_BASE}
_S_IN_8 = {"marginLeft": "4px", "marginRight": "4px", "width": "8%", **_S_INPUT_BASE}
_S_IN_MED = {"marginLeft": "2px", "marginRight": "2px", "width": "25%", **_S_INPUT_BASE}
_S_IN_SM = {"marginLeft": "2px", "marginRight": "2px", "width": "15%", **_S_INPUT_BASE}
_S_IN_20 = {"marginLeft": "2px", "marginRight": "2px", "width": "20%", **_S_INPUT_BASE}
_S_IN_TR = {"marginLeft": "4px", "marginRight": "4px", "width": "15%", **_S_INPUT_BASE}
_S_DD = {"marginLeft": "4px", "marginRight": "4px", "width": "100%"}
_S_DD_TRAJ = {"marginLeft": "4px", "marginRight": "4px", "width": "70%"}
_S_AX = {"marginLeft": "1px", "marginRight": "1px", "width": "20%", **_S_INPUT_BASE}
_S_ROW_MB = {"display": "flex", "marginBottom": "4px", "alignItems": "center"}
_S_ROW = {"display": "flex", "alignItems": "center"}

figureTemplate = {
    "layout": {
        "xaxis": {
            "title": "X",
            "showgrid": True,
            "title_font": {"color": _TICK_COLOUR},
            "gridcolor": "#0D2236",
        },
        "yaxis": {
            "title": "Y",
            "showgrid": True,
            "title_font": {"color": _TICK_COLOUR},
            "gridcolor": "#0D2236",
        },
        "legend": {
            "font": {"color": _TICK_COLOUR},
        },
        "plot_bgcolor": plotColours,
        "paper_bgcolor": plotColours,
    }
}

# ---------------------------------------------------------------------------
# Dash app and layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__)

_LABEL_STYLE = {
    "font-family": "system-ui, -apple-system, sans-serif",
    "fontSize": "11px",
    "fontWeight": "600",
    "letterSpacing": "0.08em",
    "textTransform": "uppercase",
    "color": _TEXT_MUTED,
    "margin": "5px 0 2px 2px",
    "display": "block",
}

_STYLE_BTN = {
    "backgroundColor": buttonColour,
    "color": _TEXT_PRIMARY,
    "border": f"1px solid {_INPUT_BORDER}",
    "borderRadius": "4px",
    "fontSize": "12px",
    "fontWeight": "600",
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "padding": "4px 10px",
    "cursor": "pointer",
    "marginRight": "4px",
    "minHeight": "28px",
}

_STYLE_TAB = {
    "width": "16.6%",
    "backgroundColor": parameterSectionColour,
    "color": _TEXT_SECTION,
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "fontSize": "11px",
    "fontWeight": "600",
    "letterSpacing": "0.04em",
    "borderBottom": f"2px solid {_BORDER_COL}",
    "padding": "4px 8px",
}

_STYLE_TAB_SELECTED = {
    **_STYLE_TAB,
    "color": _AMBER,
    "borderBottom": f"2px solid {_AMBER}",
    "backgroundColor": parameterSectionColour,
}

app.layout = html.Div([
    html.H1(
        [html.I(className="fas fa-crosshairs"), " TARGET TRACKER"],
        style={
            "alignItems": "center",
            "justifyContent": "center",
            "height": "28px",
            "font-family": "NK57 Monospace Condensed Light",
            "fontSize": 22,
            "fontWeight": 600,
            "letterSpacing": "0.06em",
            "textAlign": "center",
            "color": _AMBER,
            "background-color": backgroundColour,
            "margin": "0",
            "padding": "2px 0",
        },
    ),
    html.Div([
        html.Div([
            html.H3(
                "PARAMETERS",
                style={
                    "fontFamily": "system-ui, -apple-system, sans-serif",
                    "fontSize": "12px",
                    "fontWeight": 700,
                    "letterSpacing": "0.1em",
                    "textTransform": "uppercase",
                    "color": _TEXT_SECTION,
                    "margin": "0 0 6px 0",
                    "textAlign": "center",
                    "borderBottom": f"1px solid {_BORDER_COL}",
                    "paddingBottom": "6px",
                },
            ),
            html.Label("Basic", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="steps-input", value=1000, type="number", className="input-focus-effect", placeholder="steps", style=_S_IN_WIDE),
                dcc.Input(id="time-step-input", value=1, type="number", className="input-focus-effect", placeholder="time-step", style=_S_IN_WIDE),
                dcc.Input(id="num-states", value=4, type="number", className="input-focus-effect", placeholder="num-states", style=_S_IN_8),
            ], style=_S_ROW_MB),

            html.Label("States & Measurements", style=_LABEL_STYLE),
            html.Div([
                dcc.Dropdown(id="state-types-dropdown", value="1", options=stateTypes, className="dropdown-hover-effect", placeholder="states", style=_S_DD),
                dcc.Dropdown(id="measurement-types-dropdown", value="4", options=measurementTypes, className="dropdown-hover-effect", placeholder="measurements", style=_S_DD),
            ], style=_S_ROW_MB),

            html.Label("Meas Noise Actual", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="range-noise-actual", type="number", value=0.1, placeholder="range", style=_S_IN_MED),
                dcc.Input(id="azimuth-noise-actual", type="number", value=0.1, placeholder="azimuth", style=_S_IN_MED),
            ], style=_S_ROW),
            html.Div([
                dcc.Input(id="doppler-noise-actual", type="number", value=0.1, placeholder="doppler", style=_S_IN_MED),
                dcc.Input(id="process-noise-input", type="number", value=0.001, placeholder="process", style=_S_IN_MED),
            ], style=_S_ROW_MB),

            html.Label("Meas Noise Guess", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="range-noise-guess", type="number", value=0.1, placeholder="range", style=_S_IN_MED),
                dcc.Input(id="azimuth-noise-guess", type="number", value=0.1, placeholder="azimuth", style=_S_IN_MED),
                dcc.Input(id="doppler-noise-guess", type="number", value=0.1, placeholder="doppler", style=_S_IN_MED),
            ], style=_S_ROW),
            html.Div([
                dcc.Input(id="rangex-noise-guess", type="number", value=0.1, placeholder="rangex", style=_S_IN_MED),
                dcc.Input(id="rangey-noise-guess", type="number", value=0.1, placeholder="rangey", style=_S_IN_MED),
            ], style=_S_ROW_MB),

            html.Label("Initial State Guess", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="range-state", type="number", value=100, placeholder="range", style=_S_IN_MED),
                dcc.Input(id="x-state", type="number", value=70, placeholder="x", style=_S_IN_MED),
                dcc.Input(id="y-state", type="number", value=-70, placeholder="y", style=_S_IN_MED),
            ], style=_S_ROW),
            html.Div([
                dcc.Input(id="vx-state", type="number", value=0.5, placeholder="vx", style=_S_IN_SM),
                dcc.Input(id="vy-state", type="number", value=0.5, placeholder="vy", style=_S_IN_SM),
            ], style=_S_ROW_MB),

            html.Label("Initial Covariance Guess", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="range-covariance", type="number", value=20000, placeholder="range", style=_S_IN_MED),
                dcc.Input(id="x-covariance", type="number", value=20000, placeholder="x", style=_S_IN_MED),
                dcc.Input(id="y-covariance", type="number", value=20000, placeholder="y", style=_S_IN_MED),
            ], style=_S_ROW),
            html.Div([
                dcc.Input(id="vx-covariance", type="number", value=10, placeholder="vx", style=_S_IN_SM),
                dcc.Input(id="vy-covariance", type="number", value=10, placeholder="vy", style=_S_IN_SM),
            ], style=_S_ROW_MB),

            html.Label("Sensor", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="range-sensor", type="number", value=-100, placeholder="range", style=_S_IN_MED),
                dcc.Input(id="azimuth-sensor", type="number", value=0.0, placeholder="azimuth", style=_S_IN_20),
                dcc.Input(id="speed-sensor", type="number", value=1.0, placeholder="speed", style=_S_IN_SM),
            ], style=_S_ROW),
            html.Div([
                dcc.Input(id="acceleration-sensor", type="number", value=0.0, placeholder="acceleration", style=_S_IN_SM),
                dcc.Input(id="yaw-sensor", type="number", value=30, placeholder="yaw", style=_S_IN_20),
                dcc.Input(id="yawrate-sensor", type="number", value=0.0, placeholder="yawrate", style=_S_IN_SM),
                dcc.Input(id="yawraterate-sensor", type="number", value=0.0, placeholder="yawraterate", style=_S_IN_SM),
            ], style=_S_ROW_MB),

            html.Label("Target", style=_LABEL_STYLE),
            html.Div([
                dcc.Input(id="range-target", type="number", value=200, placeholder="range", style=_S_IN_MED),
                dcc.Input(id="azimuth-target", type="number", value=0.0, placeholder="azimuth", style=_S_IN_20),
                dcc.Input(id="speed-target", type="number", value=1.0, placeholder="speed", style=_S_IN_SM),
            ], style=_S_ROW),
            html.Div([
                dcc.Input(id="acceleration-target", type="number", value=0.0, placeholder="acceleration", style=_S_IN_SM),
                dcc.Input(id="yaw-target", type="number", value=45, placeholder="yaw", style=_S_IN_20),
                dcc.Input(id="yawrate-target", type="number", value=0.0, placeholder="yawrate", style=_S_IN_SM),
                dcc.Input(id="yawraterate-target", type="number", value=0.0, placeholder="yawraterate", style=_S_IN_SM),
            ], style=_S_ROW_MB),

            html.Label("Trajectory", style=_LABEL_STYLE),
            html.Div([
                dcc.Dropdown(id="sensor-trajectory-dropdown", value="FreeTraj", options=trajectories, placeholder="sensor", style=_S_DD_TRAJ),
                dcc.Dropdown(id="target-trajectory-dropdown", value="FreeTraj", options=trajectories, placeholder="target", style=_S_DD_TRAJ),
            ], style=_S_ROW_MB),
            html.Div([
                dcc.Checklist(
                    id="freeze-axes-checkbox",
                    options=[{"label": "Freeze Axes", "value": "freeze"}],
                    value=[],
                    style={
                        "color": _TEXT_MUTED,
                        "fontSize": "11px",
                        "fontFamily": "system-ui, -apple-system, sans-serif",
                    },
                ),
                dcc.Input(id="x-axis-limit-min", type="number", placeholder="Xmin", style=_S_AX),
                dcc.Input(id="x-axis-limit-max", type="number", placeholder="Xmax", style={**_S_AX, "marginLeft": "0px"}),
                dcc.Input(id="y-axis-limit-min", type="number", placeholder="Ymin", style=_S_AX),
                dcc.Input(id="y-axis-limit-max", type="number", placeholder="Ymax", style=_S_AX),
            ], style={"padding": "4px 2px"}),
            html.Div([
                dcc.Input(id="sensor-turnrate", type="number", value=0.5, placeholder="sensor-TR", style=_S_IN_TR),
                dcc.Input(id="target-turnrate", type="number", value=0.6, placeholder="target-TR", style=_S_IN_TR),
                html.Button(
                    [html.I(className="fas fa-sliders-h"), " Input"],
                    id="generate-json-btn",
                    style=_STYLE_BTN,
                    className="button-hover-effect",
                ),
                html.Button(
                    [html.I(className="fas fa-play"), " Run"],
                    id="run-btn",
                    style=_STYLE_BTN,
                    className="button-hover-effect",
                ),
                html.Div(id="json-output"),
                html.Div(id="run-output"),
            ], style=_S_ROW_MB),
        ], style={
            "height": "87vh",
            "width": "19.99%",
            "display": "inline-block",
            "background-color": parameterSectionColour,
            "color": _TEXT_PRIMARY,
            "marginLeft": "5px",
            "padding": "10px 8px",
            "fontSize": "12px",
            "fontFamily": "system-ui, -apple-system, sans-serif",
            "border-radius": "8px",
            "box-shadow": "0 2px 16px 0 rgba(0,0,0,0.5)",
            "border": f"1px solid {_BORDER_COL}",
            "overflowY": "auto",
        }),
        html.Div([
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label="Trajectory", children=[dcc.Graph(id="scene-plot-1", style={"width": "100%", "height": "83vh"}, figure=figureTemplate)], style=_STYLE_TAB, selected_style=_STYLE_TAB_SELECTED),
                dcc.Tab(label="Error-Dist", children=[dcc.Graph(id="scene-plot-2", style={"width": "100%", "height": "83vh"}, figure=figureTemplate)], style=_STYLE_TAB, selected_style=_STYLE_TAB_SELECTED),
                dcc.Tab(label="Error-Vel", children=[dcc.Graph(id="scene-plot-3", style={"width": "100%", "height": "83vh"}, figure=figureTemplate)], style=_STYLE_TAB, selected_style=_STYLE_TAB_SELECTED),
                dcc.Tab(label="Meas-Range", children=[dcc.Graph(id="scene-plot-4", style={"width": "100%", "height": "83vh"}, figure=figureTemplate)], style=_STYLE_TAB, selected_style=_STYLE_TAB_SELECTED),
                dcc.Tab(label="Meas-Az", children=[dcc.Graph(id="scene-plot-5", style={"width": "100%", "height": "83vh"}, figure=figureTemplate)], style=_STYLE_TAB, selected_style=_STYLE_TAB_SELECTED),
                dcc.Tab(label="Meas-Dop", children=[dcc.Graph(id="scene-plot-6", style={"width": "100%", "height": "83vh"}, figure=figureTemplate)], style=_STYLE_TAB, selected_style=_STYLE_TAB_SELECTED),
            ], style={"width": "100%", "height": "5vh", "alignItems": "center", "justifyContent": "center"}),
        ], style={
            "overflow": "hidden",
            "height": "100vh",
            "width": "80%",
            "display": "inline-block",
            "background-color": plotColours,
            "flex-grow": 1,
            "margin": "5px",
            "padding": "5px",
            "border-radius": "8px",
            "box-shadow": "0 2px 24px 0 rgba(0,0,0,0.7)",
            "border": f"1px solid {_BORDER_COL}",
        }),
    ], style={"height": "88vh", "display": "flex", "flexDirection": "row"}),
    html.Div([
        dcc.Slider(
            id="data-index-slider",
            min=0,
            max=sliderMax,
            value=sliderMax,
            step=1,
            marks={i: str(i) for i in range(0, sliderMax + 1, int(sliderMax))},
        ),
    ], style={"padding": "8px 20px"}),
], style={"height": "100vh", "background-color": backgroundColour, "color": _TEXT_PRIMARY, "flexWrap": "wrap"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("json-output", "children"),
    Input("generate-json-btn", "n_clicks"),
    State("steps-input", "value"),
    State("time-step-input", "value"),
    State("num-states", "value"),
    State("state-types-dropdown", "value"),
    State("measurement-types-dropdown", "value"),
    State("range-noise-actual", "value"),
    State("azimuth-noise-actual", "value"),
    State("doppler-noise-actual", "value"),
    State("process-noise-input", "value"),
    State("range-noise-guess", "value"),
    State("azimuth-noise-guess", "value"),
    State("doppler-noise-guess", "value"),
    State("rangex-noise-guess", "value"),
    State("rangey-noise-guess", "value"),
    State("range-state", "value"),
    State("x-state", "value"),
    State("y-state", "value"),
    State("vx-state", "value"),
    State("vy-state", "value"),
    State("range-covariance", "value"),
    State("x-covariance", "value"),
    State("y-covariance", "value"),
    State("vx-covariance", "value"),
    State("vy-covariance", "value"),
    State("range-sensor", "value"),
    State("azimuth-sensor", "value"),
    State("speed-sensor", "value"),
    State("acceleration-sensor", "value"),
    State("yaw-sensor", "value"),
    State("yawrate-sensor", "value"),
    State("yawraterate-sensor", "value"),
    State("range-target", "value"),
    State("azimuth-target", "value"),
    State("speed-target", "value"),
    State("acceleration-target", "value"),
    State("yaw-target", "value"),
    State("yawrate-target", "value"),
    State("yawraterate-target", "value"),
    State("sensor-trajectory-dropdown", "value"),
    State("target-trajectory-dropdown", "value"),
    State("sensor-turnrate", "value"),
    State("target-turnrate", "value"),
    prevent_initial_call=True,
)
def generate_json(
    n_clicks: int | None,
    steps: int | None,
    timeStep: float | None,
    numStates: int | None,
    stateType: str | None,
    measType: str | None,
    rangeNoiseActual: float | None,
    azimuthNoiseActual: float | None,
    dopplerNoiseActual: float | None,
    processNoise: float | None,
    rangeNoiseGuess: float | None,
    azimuthNoiseGuess: float | None,
    dopplerNoiseGuess: float | None,
    rangexNoiseGuess: float | None,
    rangeyNoiseGuess: float | None,
    rangeState: float | None,
    xState: float | None,
    yState: float | None,
    vxState: float | None,
    vyState: float | None,
    rangeCov: float | None,
    xCov: float | None,
    yCov: float | None,
    vxCov: float | None,
    vyCov: float | None,
    rangeSensor: float | None,
    azimuthSensor: float | None,
    speedSensor: float | None,
    accelSensor: float | None,
    yawSensor: float | None,
    yawrateSensor: float | None,
    yawraterateSensor: float | None,
    rangeTarget: float | None,
    azimuthTarget: float | None,
    speedTarget: float | None,
    accelTarget: float | None,
    yawTarget: float | None,
    yawrateTarget: float | None,
    yawraterateTarget: float | None,
    sensorTraj: str | None,
    targetTraj: str | None,
    sensorTR: float | None,
    targetTR: float | None,
) -> str:
    """Write updated scenario parameters to ``input/input.json`` atomically.

    Uses the module-level ``_backup_cfg`` dict (loaded once at import) as a
    template via ``copy.deepcopy``, merges all UI control values into it, and
    replaces ``input.json`` via a temp-file + ``os.replace`` to avoid partial
    writes.  Logs a config-change audit record before writing (GAP-Q6).

    @return: Status message rendered into the json-output div.
    @rtype: str
    @raises dash.exceptions.PreventUpdate: If the button has not been clicked
        or if ``steps`` is zero.
    """
    logger.debug("generate_json: n_clicks=%s steps=%s stateType=%s measType=%s", n_clicks, steps, stateType, measType)

    if n_clicks is None or steps == 0:
        raise dash.exceptions.PreventUpdate

    errors = []
    if steps is None or int(steps) < 1:
        errors.append("steps must be a positive integer")
    if timeStep is None or timeStep <= 0:
        errors.append("time_step must be positive")
    for name, val in [
        ("range noise actual", rangeNoiseActual),
        ("azimuth noise actual", azimuthNoiseActual),
        ("doppler noise actual", dopplerNoiseActual),
        ("process noise", processNoise),
        ("range noise guess", rangeNoiseGuess),
        ("azimuth noise guess", azimuthNoiseGuess),
        ("doppler noise guess", dopplerNoiseGuess),
        ("rangex noise guess", rangexNoiseGuess),
        ("rangey noise guess", rangeyNoiseGuess),
    ]:
        if val is None or val < 0:
            errors.append(f"{name} must be non-negative")
    if errors:
        logger.warning("generate_json validation failed: %s", "; ".join(errors))
        return "Validation error: " + "; ".join(errors)

    # Config audit trail (GAP-Q6)
    logger.info(
        "Config write: steps=%s timeStep=%s state=%s meas=%s sensorTraj=%s targetTraj=%s",
        steps, timeStep, stateType, measType, sensorTraj, targetTraj,
    )

    json_object = copy.deepcopy(_backup_cfg)

    json_object["simulation_parameters"]["steps"] = steps
    json_object["simulation_parameters"]["time_step"] = timeStep

    json_object["scenario_parameters"]["num_states"] = len(
        _backup_cfg["scenario_parameters"]["stateTypes"][str(stateType)]
    )
    json_object["scenario_parameters"]["states"] = stateType
    json_object["scenario_parameters"]["measurements"] = measType

    json_object["scenario_parameters"]["meas_noise_actual"] = {"range": rangeNoiseActual, "azimuth": azimuthNoiseActual, "doppler": dopplerNoiseActual}
    json_object["scenario_parameters"]["meas_noise_guess"] = {"range": rangeNoiseGuess, "azimuth": azimuthNoiseGuess, "doppler": dopplerNoiseGuess, "rangex": rangexNoiseGuess, "rangey": rangeyNoiseGuess}
    json_object["scenario_parameters"]["process_noise"] = processNoise
    json_object["scenario_parameters"]["initial_state_guess"] = {"range": rangeState, "x": xState, "y": yState, "vx": vxState, "vy": vyState}
    json_object["scenario_parameters"]["initial_covariance_guess"] = {"range": rangeCov, "x": xCov, "y": yCov, "vx": vxCov, "vy": vyCov}

    json_object["initial_states"]["sensor"] = {"range": rangeSensor, "azimuth": azimuthSensor, "speed": speedSensor, "acceleration": accelSensor, "yaw": yawSensor, "yawrate": yawrateSensor, "yawraterate": yawraterateSensor}
    json_object["initial_states"]["target"] = {"range": rangeTarget, "azimuth": azimuthTarget, "speed": speedTarget, "acceleration": accelTarget, "yaw": yawTarget, "yawrate": yawrateTarget, "yawraterate": yawraterateTarget}
    logger.debug("Trajectory selection: sensor=%s target=%s", sensorTraj, targetTraj)
    json_object["trajectory_parameters"]["selection"] = {"sensor": sensorTraj, "target": targetTraj}
    json_object["trajectory_parameters"]["STraj"] = {"sensor": {"turnRate": sensorTR}, "target": {"turnRate": targetTR}}
    json_object["trajectory_parameters"]["FreeTurnTraj"] = {"sensor": {"turnRate": sensorTR}, "target": {"turnRate": targetTR}}

    target = os.path.join(_INPUT_DIR, "input.json")
    tmp_fd, tmp_path = tempfile.mkstemp(dir=_INPUT_DIR, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(json.dumps(json_object, indent=4))
        os.replace(tmp_path, target)
    except Exception:  # pragma: no cover
        os.unlink(tmp_path)
        raise

    return "JSON file generated successfully"


@app.callback(
    [Output("scene-plot-1", "figure"),
     Output("scene-plot-2", "figure"),
     Output("scene-plot-3", "figure"),
     Output("scene-plot-4", "figure"),
     Output("scene-plot-5", "figure"),
     Output("scene-plot-6", "figure")],
    [Input("run-btn", "n_clicks"),
     Input("data-index-slider", "value"),
     Input("freeze-axes-checkbox", "value"),
     Input("x-axis-limit-min", "value"),
     Input("x-axis-limit-max", "value"),
     Input("y-axis-limit-min", "value"),
     Input("y-axis-limit-max", "value")],
    prevent_initial_call=True,
)
def run_simulation(
    n_clicks: int | None,
    index: int,
    freeze_axes: list,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
) -> list[go.Figure]:
    """Run the simulation pipeline and return all six figures.

    Wraps :func:`plot_trajectories_E` in a try/except so that simulation
    errors return six empty figures rather than an unhandled exception page
    (GAP-O6).

    @param n_clicks: Number of times the Run button has been clicked.
    @type n_clicks: int | None
    @param index: Slider position controlling how many time steps are shown.
    @type index: int
    @param freeze_axes: List containing ``"freeze"`` when the freeze checkbox
        is checked, empty otherwise.
    @type freeze_axes: list
    @param xmin: Minimum x-axis limit (used when axes are frozen).
    @type xmin: float | None
    @param xmax: Maximum x-axis limit.
    @type xmax: float | None
    @param ymin: Minimum y-axis limit.
    @type ymin: float | None
    @param ymax: Maximum y-axis limit.
    @type ymax: float | None
    @return: List of six Plotly Figures in display order.
    @rtype: list[go.Figure]
    @raises dash.exceptions.PreventUpdate: If the Run button has not yet been clicked.
    """
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    try:
        figs = plot_trajectories_E(index)
    except Exception as exc:
        logger.exception("Simulation pipeline failed: %s", exc)
        empty = go.Figure()
        _apply_dark_theme(empty, "", "")
        return [empty] * 6

    if freeze_axes:
        for fig in figs:
            fig.update_layout(
                xaxis_range=[xmin, xmax] if xmin is not None and xmax is not None else None,
                yaxis_range=[ymin, ymax] if ymin is not None and ymax is not None else None,
            )
    return figs


@app.callback(
    [Output("data-index-slider", "max"),
     Output("data-index-slider", "marks"),
     Output("data-index-slider", "value")],
    [Input("steps-input", "value")],
    prevent_initial_call=True,
)
def update_slider_max(steps_value: int | None) -> tuple[int, dict, int]:
    """Update slider range and tick marks when the steps input changes.

    @param steps_value: New value from the steps input field; ``None`` if
        the field is empty.
    @type steps_value: int | None
    @return: Tuple ``(max_value, marks, value)`` for the slider component.
    @rtype: tuple[int, dict, int]
    @raises dash.exceptions.PreventUpdate: If ``steps_value`` is ``None``.
    """
    if steps_value is None:
        raise dash.exceptions.PreventUpdate
    max_value = steps_value if steps_value is not None else sliderMax
    if max_value is not None:
        if max_value < 10:
            step_size = 1
        else:
            step_size = int(max_value / 10)
    else:  # pragma: no cover
        step_size = 10
    marks = {i: str(i) for i in range(0, max_value + 1, step_size)}

    return max_value, marks, max_value


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app.run(debug=True)
