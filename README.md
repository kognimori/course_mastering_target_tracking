# Kalman Filter with Simulator

<img width="1600" height="900" alt="Screenshot from 2026-06-23 19-28-55" src="https://github.com/user-attachments/assets/f373eced-9ccb-4bd9-8c30-0511b3b60b1e" />


An interactive python Dash dashboard for simulating 2-D kinematic scenarios and evaluating Extended Kalman Filter (EKF) performance against ground-truth trajectories meant as a supplementary material for the course https://www.udemy.com/course/mastering-target-tracking-for-adas-autonomous-driving/?referralCode=C63D0

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Running the dashboard](#5-running-the-dashboard)
6. [Running the command-line pipeline](#6-running-the-command-line-pipeline)
7. [Running the tests](#7-running-the-tests)
8. [Configuration reference — `input/input.json`](#8-configuration-reference--inputinputjson)
9. [Trajectory models](#9-trajectory-models)
10. [Measurement types](#10-measurement-types)
11. [State-estimation types](#11-state-estimation-types)
12. [Dashboard walkthrough](#12-dashboard-walkthrough)
13. [Output files](#13-output-files)
14. [Common issues and fixes](#14-common-issues-and-fixes)

---

## 1. Project overview

The project simulates a sensor (ego) platform and a moving target, generates noisy sensor measurements, and fuses them using a discrete-time EKF. Results are displayed in an interactive Plotly/Dash browser dashboard.

Core pipeline:

```
input/input.json
      │
      ▼
ObjectMotionSimulator   ←── trajectories (FreeTraj | STraj | FreeTurnTraj)
      │                          writes output/*.ftr (feather/arrow)
      ▼
KalmanFilter            ←── EKF predict/update loop
      │                          returns estimate DataFrame
      ▼
Dashboard figures       ←── trajectory, error, measurement plots
```

---

## 2. Repository structure

```
object_tracking/
├── input/
│   ├── input.json                  # Active configuration (edited by dashboard or manually)
│   ├── input_backup.json           # Template restored when "Input" button is pressed
│   ├── measurements.json           # Measurement template (noise models, column names)
│   └── object_motion_parameters.json  # State-array template (all kinematic fields)
│
├── output/
│   ├── sensor_trajectory.ftr       # Ego platform ground-truth states
│   ├── target_trajectory.ftr       # Target ground-truth states
│   ├── measurements.ftr            # Noisy measurements written by OMS
│   └── estimates.ftr               # KF position/velocity estimates
│
├── src/
│   ├── dashboard.py                # Dash app — layout, callbacks, figure builders
│   ├── object_motion_simulator.py  # OMS — runs trajectories and writes measurements
│   ├── kalman_filter.py            # EKF — predict, update, state dispatch
│   ├── trajectory.py               # Trajectory models (FreeTraj, STraj, FreeTurnTraj)
│   └── run.py                      # CLI entry point (no dashboard)
│
├── tests/
│   ├── conftest.py                 # Shared fixtures (SRC_DIR, OBJECT_MOTION_TEMPLATE, …)
│   └── routes/
│       ├── routes_dashboard.py     # Dashboard callback and figure-builder routes
│       ├── routes_kalman.py        # Kalman filter method routes
│       ├── routes_run.py           # run.py plot-function routes
│       ├── routes_simulator.py     # ObjectMotionSimulator routes
│       └── routes_trajectory.py   # FreeTraj / STraj / FreeTurnTraj routes
│
├── docs/
│
├── pytest.ini                      # Test discovery and coverage settings
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

---

## 3. Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| Python | 3.10 |
| pip | 22 |

All other dependencies are listed in `requirements.txt`.

---

## 4. Installation

### 4.1 Clone the repository

```bash
git clone <repo-url>
cd object_tracking
```

### 4.2 Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | Purpose |
|---------|---------|
| `numpy` | Array maths throughout |
| `pandas` | DataFrame I/O for `.ftr` files |
| `pyarrow` | Feather/Arrow backend for `pandas` |
| `scipy` | `linalg.inv` used in EKF update |
| `plotly` | Interactive figures |
| `dash` | Web dashboard framework |
| `pytest` | Test runner |
| `pytest-cov` | Coverage enforcement |

---

## 5. Running the dashboard

The dashboard is the primary interface. It lets you configure scenarios, run simulations, and explore results interactively.

### 5.1 Start the server

From the **repository root**:

```powershell
# Windows
.venv\Scripts\python.exe src\dashboard.py

# macOS / Linux
.venv/bin/python src/dashboard.py
```

The terminal prints:

```
Dash is running on http://127.0.0.1:8050/
```

### 5.2 Open the dashboard

Navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) in any browser.

### 5.3 Typical workflow

1. **Configure** — fill in all parameter panels on the left (simulation, scenario, trajectory, sensor/target initial states).
2. **Input** — click the **Input** button. This validates the form, writes `input/input.json`, and reloads the slider range to match `steps`.
3. **Run** — click the **Run** button. The OMS and EKF execute, results are cached, and all six figures are rendered.
4. **Explore** — drag the **Slider** to step through the simulation time history. The cached results are re-sliced; no re-simulation occurs.
5. **Iterate** — change parameters and click **Input** then **Run** again to compare scenarios.

---

## 6. Running the command-line pipeline

`src/run.py` executes the full pipeline (OMS + KF) without the dashboard, then opens a Plotly figure in the browser.

```powershell
# Windows — from repository root
.venv\Scripts\python.exe src\run.py

# macOS / Linux
.venv/bin/python src/run.py
```

`run.py` reads `input/input.json` directly, runs the simulation, and calls `plot_trajectories_E` or `plot_trajectories_S` depending on the configured state type. Output `.ftr` files are written to `output/`.

---

## 7. Running the tests

The test suite enforces **100 % branch and line coverage** on all five source modules.

### 7.1 Run all tests with coverage

From the **repository root** (virtual environment active):

```powershell
.venv\Scripts\python.exe -m pytest
```

`pytest.ini` sets `testpaths = tests`, `addopts = --cov=src --cov-report=term-missing --cov-fail-under=100`, so coverage is always checked.

Expected output tail:

```
Name                             Stmts   Miss  Cover
----------------------------------------------------
src\dashboard.py                   346      0   100%
src\kalman_filter.py               162      0   100%
src\object_motion_simulator.py     134      0   100%
src\run.py                          29      0   100%
src\trajectory.py                  147      0   100%
----------------------------------------------------
TOTAL                              818      0   100%
Required test coverage of 100% reached. Total coverage: 100.00%
106 passed in X.XXs
```

### 7.2 Run a single test file

```powershell
.venv\Scripts\python.exe -m pytest tests/routes/routes_trajectory.py -v
```

### 7.3 Run tests without coverage (faster iteration)

```powershell
.venv\Scripts\python.exe -m pytest tests/ --no-cov
```

### 7.4 Test naming convention

| pytest element | Convention |
|----------------|-----------|
| Files | `routes_<module>.py` |
| Classes | `Routes<ClassName>` |
| Methods | `route_<description>` |

Test discovery is configured in `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = routes_*.py
python_classes = Routes*
python_functions = route_*
addopts = --cov=src --cov-report=term-missing --cov-fail-under=100
```

---

## 8. Configuration reference — `input/input.json`

All simulation parameters live in `input/input.json`. The dashboard writes this file when **Input** is clicked. You can also edit it manually between CLI runs.

`input/input_backup.json` is the factory-reset template: clicking **Input** in the dashboard loads this file and applies the current UI values on top of it, so any fields not exposed in the UI revert to the backup defaults.

### 8.1 Top-level structure

```json
{
  "paths": { ... },
  "simulation_parameters": { ... },
  "scenario_parameters": { ... },
  "initial_states": { ... },
  "trajectory_parameters": { ... }
}
```

### 8.2 `simulation_parameters`

| Key | Type | Description |
|-----|------|-------------|
| `steps` | integer | Total number of time steps to simulate. The playback slider maximum is set to this value. |
| `time_step` | float | Duration of each step in seconds. |

**Example:**

```json
"simulation_parameters": {
  "steps": 100,
  "time_step": 1
}
```

### 8.3 `scenario_parameters`

| Key | Type | Description |
|-----|------|-------------|
| `num_states` | integer | Dimension of the state vector given to the KF. Must match the chosen `states` type. |
| `states` | string key | Selects the active state type from `stateTypes`. `"0"` = range-only; `"1"` = Cartesian (vx, vy, x, y). |
| `stateTypes` | object | Maps string keys to lists of state variable names. |
| `measurements` | string key | Selects the active measurement type from `measurementsTypes`. |
| `measurementsTypes` | object | Maps string keys to lists of measured quantities (see [§10](#10-measurement-types)). |
| `meas_noise_actual` | object | True (ground-truth) measurement noise standard deviations used by the OMS. |
| `meas_noise_guess` | object | Assumed measurement noise standard deviations used by the KF (`R` matrix diagonal). |
| `process_noise` | float | Scalar multiplier for the KF process noise matrix `Q`. |
| `initial_state_guess` | object | KF initial state vector values (per-variable). |
| `initial_covariance_guess` | object | KF initial covariance `P₀` diagonal values (per-variable). |

**State types:**

| Key | State vector |
|-----|-------------|
| `"0"` | `[range]` — scalar range between sensor and target |
| `"1"` | `[vx, vy, x, y]` — Cartesian velocity and position |

### 8.4 `initial_states`

Sets the ground-truth starting conditions for `sensor` and `target` independently.

| Key | Unit | Description |
|-----|------|-------------|
| `range` | metres | Radial distance from origin |
| `azimuth` | degrees | Bearing angle from North/x-axis |
| `speed` | m/s | Initial scalar speed |
| `acceleration` | m/s² | Initial scalar acceleration |
| `yaw` | degrees | Heading angle |
| `yawrate` | degrees/s | Rate of heading change (used by `FreeTraj`) |
| `yawraterate` | degrees/s² | Rate of yaw-rate change |

**Example:**

```json
"initial_states": {
  "sensor": { "range": 0.0, "azimuth": 0.0, "speed": 1.0, "acceleration": 0.0, "yaw": 30.0, "yawrate": 0.0, "yawraterate": 0.0 },
  "target": { "range": 2.5, "azimuth": 0.0, "speed": 1.0, "acceleration": 0.0, "yaw": 35.0, "yawrate": 0.0, "yawraterate": 0.0 }
}
```

### 8.5 `trajectory_parameters`

| Key | Description |
|-----|-------------|
| `selection` | Object with `"sensor"` and `"target"` keys each naming one of the registered trajectory types. |
| `FreeTraj` | Parameter block for `FreeTraj` (currently empty — uses `initial_states` directly). |
| `STraj` | Parameter block for `STraj`. Requires `turnRate` (degrees) per entity. |
| `FreeTurnTraj` | Parameter block for `FreeTurnTraj`. Requires `turnRate` (degrees/step) per entity. |

**Example — constant-turn-rate target with free-trajectory sensor:**

```json
"trajectory_parameters": {
  "selection": { "sensor": "FreeTraj", "target": "FreeTurnTraj" },
  "FreeTraj": { "sensor": {}, "target": {} },
  "STraj": { "sensor": { "turnRate": 0.5 }, "target": { "turnRate": 0.6 } },
  "FreeTurnTraj": { "sensor": { "turnRate": 0.0 }, "target": { "turnRate": 1.5 } }
}
```

---

## 9. Trajectory models

Three kinematic trajectory models are available. Select one per entity (sensor / target) via `trajectory_parameters.selection`.

### 9.1 `FreeTraj` — free yaw

Integrates speed, yaw rate, and yaw rate rate from the `initial_states` values.  The heading evolves naturally if `yawrate` or `yawraterate` are non-zero.  Suitable for straight-line or gently curving manoeuvres.

**Required parameters:** none (uses `initial_states` only).

### 9.2 `STraj` — sinusoidal S-curve

Generates a parametric sinusoidal path in the body frame. The trajectory switches sign and amplitude every 180 ° of cumulative phase (the *n-switch*), producing a serpentine track.

**Required parameters in `trajectory_parameters.STraj.<entity>`:**

| Key | Unit | Description |
|-----|------|-------------|
| `turnRate` | degrees | Controls the spatial period of the S-curve. Larger values → tighter, faster turns. |

### 9.3 `FreeTurnTraj` — constant turn rate

Like `FreeTraj` but the yaw rate is locked to a user-specified constant for every time step, ignoring the `yawrate` value in `initial_states`. Suitable for modelling coordinated constant-rate turning manoeuvres (e.g., aircraft holding pattern, circular orbit).

**Required parameters in `trajectory_parameters.FreeTurnTraj.<entity>`:**

| Key | Unit | Description |
|-----|------|-------------|
| `turnRate` | degrees/step | Fixed yaw rate applied at each step. Positive = counter-clockwise. |

---

## 10. Measurement types

Select a measurement type via `scenario_parameters.measurements` (string key into `measurementsTypes`).

| Key | Measured quantities | Description |
|-----|---------------------|-------------|
| `"0"` | `range` | Radial distance only (1-D). |
| `"1"` | `azimuth` | Bearing angle only. |
| `"2"` | `azimuth`, `range` | Polar 2-D (range + bearing). |
| `"3"` | `azimuth`, `range`, `doppler` | Polar + radial velocity (active radar). |
| `"4"` | `rangex`, `rangey`, `doppler` | Cartesian range components + radial velocity. |

Noise standard deviations for each quantity are configured in `meas_noise_actual` (OMS simulation) and `meas_noise_guess` (KF model).

---

## 11. State-estimation types

Select a state-estimation type via `scenario_parameters.states` (string key into `stateTypes`).

| Key | State vector | Notes |
|-----|-------------|-------|
| `"0"` | `[range]` | Range-only EKF. `num_states` must be 1. |
| `"1"` | `[vx, vy, x, y]` | Cartesian velocity + position EKF. `num_states` must be 4. |

The chosen state type determines:

- Which transition matrix `A` and process noise `Q` the KF uses.
- Which measurement linearisation `H` is applied during the update step.
- Which dashboard figures are rendered (trajectory XY vs. range time-series).

---

## 12. Dashboard walkthrough

### 12.1 Panel layout

The dashboard is divided into a left **controls panel** and a right **figures panel**.

**Controls panel (left):**

| Section | Controls |
|---------|---------|
| Simulation | Steps, time step |
| Scenario | Number of states, state type, measurement type |
| Noise (actual) | Range, azimuth, Doppler measurement noise σ |
| Noise (guess) | KF-assumed range, azimuth, Doppler, rangex, rangey noise σ |
| Process noise | KF process noise scalar |
| Initial state guess | KF state vector at t=0 |
| Initial covariance guess | KF covariance P₀ diagonal |
| Sensor initial states | Sensor platform ground-truth start |
| Target initial states | Target ground-truth start |
| Trajectory | Sensor trajectory type + turn rate; target trajectory type + turn rate |

**Action buttons:**

| Button | Effect |
|--------|--------|
| **Input** | Validates and writes all control values to `input/input.json`. Resets the slider maximum to the new `steps` value. |
| **Run** | Runs OMS + KF on the current `input.json`. Caches results by MD5 of the config. Renders all six figures. |

**Slider:** steps through cached simulation frames 0 … `steps`. No re-simulation occurs on drag.

### 12.2 Figure descriptions

| Figure | Description |
|--------|-------------|
| **Trajectory** | Ground-truth target and sensor paths (XY for Cartesian; time-series range for range-only) plus KF estimates. Start/end markers shown. |
| **Distance error** | Per-step position error between KF estimate and ground truth, with running time-average. |
| **Velocity error** | Per-step velocity error, with running time-average. |
| **Range measurement** | Raw range measurements (with noise) vs. ground-truth range. |
| **Azimuth measurement** | Raw azimuth measurements vs. ground truth. |
| **Doppler measurement** | Actual and estimated radial velocity (Doppler). |

---

## 13. Output files

All outputs are written to `output/` in Apache Feather format (`.ftr`), readable with `pandas.read_feather`.

| File | Written by | Contents |
|------|-----------|---------|
| `sensor_trajectory.ftr` | `ObjectMotionSimulator` | Sensor platform state history (x, y, vx, vy, yaw, speed, …) |
| `target_trajectory.ftr` | `ObjectMotionSimulator` | Target state history |
| `measurements.ftr` | `ObjectMotionSimulator` | Noisy sensor measurements at each step |
| `estimates.ftr` | `KalmanFilter` | KF state estimates (x, y, vx, vy, range, timestamps, P_x, P_y, …) |

**Reading output files in Python:**

```python
import pandas as pd

target = pd.read_feather("output/target_trajectory.ftr")
estimates = pd.read_feather("output/estimates.ftr")
print(target.columns.tolist())
print(estimates.head())
```

These files are tracked in git as versioned reference outputs — they record the baseline simulation results corresponding to the committed `input/input.json`.

---

## 14. Common issues and fixes

### `ModuleNotFoundError: No module named 'trajectory'` (or similar)

The `src/` directory must be on `sys.path`. Run the scripts from the **repository root** using the full paths shown above (`src/dashboard.py`, `src/run.py`). Do not `cd src/` before running.

The `conftest.py` in `tests/` inserts `src/` into `sys.path` at test collection time, so tests do not require the same adjustment.

---

### `FileNotFoundError: input/input.json`

The scripts resolve `input/` relative to the location of the source file (`__file__`), not the working directory. As long as the repo structure is intact and you have not moved `src/` or `input/`, this should not occur. If it does, verify that `input/input.json` exists and has not been corrupted.

---

### `ValueError: Unknown trajectory type: 'X'`

The trajectory type name in `trajectory_parameters.selection` is not registered. Registered types are: `FreeTraj`, `STraj`, `FreeTurnTraj`. Check for typos in `input.json`.

---

### Dashboard slider goes past the end of the data

Click **Input** after changing the `steps` value. The `update_slider_max` callback fires only on **Input** clicks, not on page load from a stale config. Alternatively, set `steps` and click **Input** before **Run**.

---

### Tests fail with `ImportError` or `AttributeError`

Ensure the virtual environment is activated (`(.venv)` prefix in your shell prompt) and that you ran `pip install -r requirements.txt`. Run tests with:

```powershell
.venv\Scripts\python.exe -m pytest
```

not with the system `python` or `pytest` binary.

---

### Coverage drops below 100 %

The gate is enforced by `pytest.ini --cov-fail-under=100`. If you add new code, add corresponding routes in `tests/routes/routes_<module>.py` following the naming convention in [§7.4](#74-test-naming-convention). Each new public method needs at least one `route_` function exercising it.
