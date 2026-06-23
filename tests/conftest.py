"""
Shared fixtures for route coverage tests.
All modules are imported from src/; file-I/O-dependent tests use
`src_cwd` to change working directory so ../input/ and ../output/
resolve correctly.
"""
import os
import sys
import json
import copy

import numpy as np
import pandas as pd
import pytest

# Put src/ on the path at import time so modules can be imported.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Working-directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def src_cwd(monkeypatch):
    """Change cwd to src/ so ../input/ and ../output/ resolve correctly."""
    monkeypatch.chdir(SRC_DIR)


# ---------------------------------------------------------------------------
# Canonical input config (mirrors input/input.json)
# ---------------------------------------------------------------------------

INPUT_JSON_PATH = os.path.join(REPO_ROOT, "input", "input.json")

def load_input_json():
    with open(INPUT_JSON_PATH) as f:
        return json.load(f)


def make_input(states="1", measurements="4", steps=20, time_step=1.0):
    """Return a minimal deep-copy of the input config with overrides."""
    cfg = load_input_json()
    cfg["scenario_parameters"]["states"] = states
    cfg["scenario_parameters"]["measurements"] = measurements
    cfg["simulation_parameters"]["steps"] = steps
    cfg["simulation_parameters"]["time_step"] = time_step
    if states == "0":
        cfg["scenario_parameters"]["num_states"] = 1
    elif states == "2":
        cfg["scenario_parameters"]["num_states"] = 6
    else:
        cfg["scenario_parameters"]["num_states"] = 4
    return cfg


# ---------------------------------------------------------------------------
# Object-motion template (mirrors input/object_motion_parameters.json)
# ---------------------------------------------------------------------------

OBJECT_MOTION_TEMPLATE = {
    "x": [], "y": [], "yaw": [],
    "vx": [], "vy": [], "yawrate": [],
    "ax": [], "ay": [], "yawraterate": [],
    "speed": [], "acceleration": [], "timestamps": [],
}

# ---------------------------------------------------------------------------
# Tiny synthetic DataFrames for dashboard figure tests
# ---------------------------------------------------------------------------

N = 5  # number of rows


def make_sensor_df():
    t = np.arange(N, dtype=float)
    return pd.DataFrame({
        "timestamps": t, "x": t * 0.1, "y": t * 0.05,
        "vx": np.ones(N) * 0.1, "vy": np.zeros(N),
        "ax": np.zeros(N), "ay": np.zeros(N),
        "yaw": np.zeros(N), "yawrate": np.zeros(N),
        "yawraterate": np.zeros(N), "speed": np.ones(N) * 0.1,
        "acceleration": np.zeros(N),
    })


def make_target_df():
    t = np.arange(N, dtype=float)
    return pd.DataFrame({
        "timestamps": t, "x": t * 0.2 + 2.0, "y": t * 0.1 + 2.0,
        "vx": np.ones(N) * 0.2, "vy": np.ones(N) * 0.1,
        "ax": np.zeros(N), "ay": np.zeros(N),
        "yaw": np.zeros(N), "yawrate": np.zeros(N),
        "yawraterate": np.zeros(N), "speed": np.ones(N) * 0.2,
        "acceleration": np.zeros(N),
    })


def make_estimates_xyz(n=N):
    t = np.arange(n, dtype=float)
    return pd.DataFrame({
        "timestamps": t, "x": t * 0.18 + 2.1, "y": t * 0.09 + 2.1,
        "vx": np.ones(n) * 0.19, "vy": np.ones(n) * 0.09,
        "P_x": np.ones(n) * 0.01, "P_y": np.ones(n) * 0.01,
        "P_vx": np.ones(n) * 0.001, "P_vy": np.ones(n) * 0.001,
    })


def make_estimates_range(n=N):
    t = np.arange(n, dtype=float)
    return pd.DataFrame({
        "timestamps": t, "range": np.ones(n) * 2.8,
        "P_range": np.ones(n) * 0.05,
    })


def make_measurements_df(n=N):
    t = np.arange(n, dtype=float)
    return pd.DataFrame({
        "timestamps": t,
        "range":    np.ones(n) * 2.5,
        "azimuth":  np.ones(n) * 0.1,
        "doppler":  np.ones(n) * 0.05,
        "rangex":   np.ones(n) * 2.4,
        "rangey":   np.ones(n) * 0.5,
        "vrelx":    np.zeros(n),
        "vrely":    np.zeros(n),
        "azimuthrate": np.zeros(n),
    })
