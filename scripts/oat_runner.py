#!/usr/bin/env python
"""
OAT (One-At-a-Time) Sensitivity Analysis
=========================================
Sweeps each of 40 parameters across its realistic range, measures
five Kalman-filter error metrics, then generates a self-contained
HTML report with two interactive Plotly tabs.

Usage
-----
  python oat_runner.py               # full run  (~1 126 simulations)
  python oat_runner.py --plot-only   # skip sims, rebuild HTML from saved CSV
  python oat_runner.py --quick 5     # 5 sweep points per param (smoke test)
"""

import argparse, copy, json, math, os, shutil, sys, tempfile, time, traceback
import numpy as np
import pandas as pd

# ── Paths (absolute) ──────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(REPO, "src")
INP  = os.path.join(REPO, "input")
OUT  = os.path.join(REPO, "output")
CSV  = os.path.join(REPO, "oat_results.csv")
HTML = os.path.join(REPO, "oat_report.html")
TCFG = os.path.join(INP,  "_oat_temp.json")

sys.path.insert(0, SRC)
os.chdir(SRC)   # OMS/KF resolve ../input/ and ../output/ from here

from object_motion_simulator import ObjectMotionSimulator
from kalman_filter import KalmanFilter

# ── Baseline config ───────────────────────────────────────────────────────────
with open(os.path.join(INP, "input.json")) as _f:
    _RAW = json.load(_f)

def _make_baseline() -> dict:
    b = copy.deepcopy(_RAW)
    b["simulation_parameters"]["steps"]     = 1000
    b["simulation_parameters"]["time_step"] = 1
    sp = b["scenario_parameters"]
    sp["states"]      = "1"
    sp["measurements"]= "4"
    sp["num_states"]  = 4
    sp["process_noise"]                         = 0.001
    sp["meas_noise_actual"]["range"]            = 0.1
    sp["meas_noise_actual"]["azimuth"]          = 0.1
    sp["meas_noise_actual"]["doppler"]          = 0.1
    sp["meas_noise_guess"]["range"]             = 0.1
    sp["meas_noise_guess"]["azimuth"]           = 0.1
    sp["meas_noise_guess"]["doppler"]           = 0.1
    sp["meas_noise_guess"]["rangex"]            = 0.1
    sp["meas_noise_guess"]["rangey"]            = 0.1
    sp["initial_state_guess"]["range"]          = 100
    sp["initial_state_guess"]["x"]              = 70
    sp["initial_state_guess"]["y"]              = -70
    sp["initial_state_guess"]["vx"]             = 0.5
    sp["initial_state_guess"]["vy"]             = 0.5
    sp["initial_covariance_guess"]["range"]     = 20000
    sp["initial_covariance_guess"]["x"]         = 20000
    sp["initial_covariance_guess"]["y"]         = 20000
    sp["initial_covariance_guess"]["vx"]        = 10
    sp["initial_covariance_guess"]["vy"]        = 10
    b["initial_states"]["sensor"] = dict(
        range=-100, azimuth=0, speed=1, acceleration=0,
        yaw=30, yawrate=0, yawraterate=0)
    b["initial_states"]["target"] = dict(
        range=200, azimuth=0, speed=1, acceleration=0,
        yaw=45, yawrate=0, yawraterate=0)
    b["trajectory_parameters"]["selection"]["sensor"] = "FreeTraj"
    b["trajectory_parameters"]["selection"]["target"] = "FreeTraj"
    b["trajectory_parameters"]["STraj"]["sensor"]["turnRate"]        = 0.5
    b["trajectory_parameters"]["STraj"]["target"]["turnRate"]        = 0.6
    b["trajectory_parameters"]["FreeTurnTraj"]["sensor"]["turnRate"] = 0.5
    b["trajectory_parameters"]["FreeTurnTraj"]["target"]["turnRate"] = 0.6
    return b

BASELINE = _make_baseline()


def _make_ca_baseline() -> dict:
    """Return a CA (constant-acceleration) baseline config.

    Identical to the CV baseline except:
    - states = "2"  (CA: ax, ay, vx, vy, x, y)
    - num_states = 6
    - stateTypes["2"] defined
    - initial_state_guess / initial_covariance_guess include ax, ay
    Target acceleration is non-zero so the CA model's advantage over CV is
    immediately visible in the P32 sweep.
    """
    b = _make_baseline()
    sp = b["scenario_parameters"]
    sp["states"] = "2"
    sp["num_states"] = 6
    sp["stateTypes"]["2"] = ["ax", "ay", "vx", "vy", "x", "y"]
    sp["initial_state_guess"]["ax"] = 0.0
    sp["initial_state_guess"]["ay"] = 0.0
    sp["initial_covariance_guess"]["ax"] = 10.0
    sp["initial_covariance_guess"]["ay"] = 10.0
    return b


CA_BASELINE = _make_ca_baseline()

# ── Sweep table ───────────────────────────────────────────────────────────────
_L = lambda lo, hi, n: list(np.logspace(np.log10(lo), np.log10(hi), n))
_I = lambda lo, hi, n: list(np.linspace(lo, hi, n))

COMBO_STATES = [
    ("range+range",              "0", "0", 1),
    ("CV+azimuth",               "1", "1", 4),
    ("CV+azi+range",             "1", "2", 4),
    ("CV+azi+range+doppler",     "1", "3", 4),
    ("CV+rangex+rangey+doppler", "1", "4", 4),
]

SWEEPS = [
    # Simulation
    dict(id="P01", label="Steps",                     group="Simulation",
         values=[100,200,300,500,750,1000,1500,2000,3000,5000]),
    dict(id="P02", label="Time step (s)",              group="Simulation",
         values=_I(0.1, 2.0, 20)),
    # Meas noise actual
    dict(id="P03", label="Range noise actual (m)",     group="Meas Noise Actual",
         values=_L(0.1, 100, 40)),
    dict(id="P04", label="Azimuth noise actual (°)",   group="Meas Noise Actual",
         values=_L(0.01, 10, 40)),
    dict(id="P05", label="Doppler noise actual (m/s)", group="Meas Noise Actual",
         values=_L(0.01, 10, 40)),
    # Process noise
    dict(id="P06", label="Process noise q",            group="Process Noise",
         values=_L(1e-5, 100, 40)),
    # Meas noise guess
    dict(id="P07", label="Range noise guess (m)",      group="Meas Noise Guess",
         values=_L(0.01, 10, 40)),
    dict(id="P08", label="Azimuth noise guess (°)",    group="Meas Noise Guess",
         values=_L(0.001, 1, 40)),
    dict(id="P09", label="Doppler noise guess (m/s)",  group="Meas Noise Guess",
         values=_L(0.001, 1, 40)),
    dict(id="P10", label="RangeX noise guess (m)",     group="Meas Noise Guess",
         values=_L(0.01, 10, 40)),
    dict(id="P11", label="RangeY noise guess (m)",     group="Meas Noise Guess",
         values=_L(0.01, 10, 40)),
    # Initial state guess
    dict(id="P12", label="Init range guess (m)",       group="Initial State",
         values=_I(10, 500, 25)),
    dict(id="P13", label="Init x guess (m)",           group="Initial State",
         values=_I(-300, 300, 25)),
    dict(id="P14", label="Init y guess (m)",           group="Initial State",
         values=_I(-300, 300, 25)),
    dict(id="P15", label="Init vx guess (m/s)",        group="Initial State",
         values=_I(-10, 10, 25)),
    dict(id="P16", label="Init vy guess (m/s)",        group="Initial State",
         values=_I(-10, 10, 25)),
    # Initial covariance guess
    dict(id="P17", label="P0 range (m²)",              group="Init Covariance",
         values=_L(1, 1e7, 40)),
    dict(id="P18", label="P0 x (m²)",                  group="Init Covariance",
         values=_L(1, 1e7, 40)),
    dict(id="P19", label="P0 y (m²)",                  group="Init Covariance",
         values=_L(1, 1e7, 40)),
    dict(id="P20", label="P0 vx ((m/s)²)",             group="Init Covariance",
         values=_L(0.01, 1e4, 40)),
    dict(id="P21", label="P0 vy ((m/s)²)",             group="Init Covariance",
         values=_L(0.01, 1e4, 40)),
    # Sensor init
    dict(id="P22", label="Sensor range0 (m)",          group="Sensor Init",
         values=_I(-1000, -10, 25)),
    dict(id="P23", label="Sensor azimuth0 (°)",        group="Sensor Init",
         values=_I(-180, 180, 25)),
    dict(id="P24", label="Sensor speed (m/s)",         group="Sensor Init",
         values=_I(0, 50, 25)),
    dict(id="P25", label="Sensor accel (m/s²)",        group="Sensor Init",
         values=_I(-10, 10, 25)),
    dict(id="P26", label="Sensor yaw (°)",             group="Sensor Init",
         values=_I(0, 360, 25)),
    dict(id="P27", label="Sensor yaw rate (°/s)",      group="Sensor Init",
         values=_I(-20, 20, 25)),
    dict(id="P28", label="Sensor yaw rate rate (deg/s^2)", group="Sensor Init",
         values=_I(-5, 5, 25)),
    # Target init
    dict(id="P29", label="Target range0 (m)",          group="Target Init",
         values=_I(50, 1000, 25)),
    dict(id="P30", label="Target azimuth0 (°)",        group="Target Init",
         values=_I(-180, 180, 25)),
    dict(id="P31", label="Target speed (m/s)",         group="Target Init",
         values=_I(0.1, 50, 25)),
    dict(id="P32", label="Target accel (m/s²)",        group="Target Init",
         values=_I(-15, 15, 25)),
    dict(id="P33", label="Target yaw (°)",             group="Target Init",
         values=_I(0, 360, 25)),
    dict(id="P34", label="Target yaw rate (°/s)",      group="Target Init",
         values=_I(-30, 30, 25)),
    dict(id="P35", label="Target yaw rate rate (deg/s^2)", group="Target Init",
         values=_I(-5, 5, 25)),
    # Turn rates (force STraj selection)
    dict(id="P36", label="Sensor turn rate (°/s)",     group="Turn Rates",
         values=_I(0, 10, 25)),
    dict(id="P37", label="Target turn rate (°/s)",     group="Turn Rates",
         values=_I(0, 20, 25)),
    # Categoricals
    dict(id="C12", label="State+Meas config",          group="Categorical",
         values=[c[0] for c in COMBO_STATES]),
    dict(id="C3",  label="Sensor trajectory",          group="Categorical",
         values=["STraj", "FreeTraj", "FreeTurnTraj"]),
    dict(id="C4",  label="Target trajectory",          group="Categorical",
         values=["STraj", "FreeTraj", "FreeTurnTraj"]),
]

# ── Column metadata for HTML ──────────────────────────────────────────────────
# Maps each DataFrame column → (display label, sweep_id or None, is_output)
COL_META: dict[str, dict] = {
    "in_steps":               dict(label="Steps",                   sweep="P01", out=False, cat=False),
    "in_time_step":           dict(label="Time step (s)",           sweep="P02", out=False, cat=False),
    "in_range_noise_act":     dict(label="Range noise actual (m)",  sweep="P03", out=False, cat=False),
    "in_az_noise_act":        dict(label="Azimuth noise actual (°)",sweep="P04", out=False, cat=False),
    "in_dop_noise_act":       dict(label="Doppler noise actual (m/s)",sweep="P05",out=False,cat=False),
    "in_process_noise_q":     dict(label="Process noise q",         sweep="P06", out=False, cat=False),
    "in_range_noise_gss":     dict(label="Range noise guess (m)",   sweep="P07", out=False, cat=False),
    "in_az_noise_gss":        dict(label="Azimuth noise guess (°)", sweep="P08", out=False, cat=False),
    "in_dop_noise_gss":       dict(label="Doppler noise guess (m/s)",sweep="P09",out=False,cat=False),
    "in_rangex_noise_gss":    dict(label="RangeX noise guess (m)",  sweep="P10", out=False, cat=False),
    "in_rangey_noise_gss":    dict(label="RangeY noise guess (m)",  sweep="P11", out=False, cat=False),
    "in_init_range":          dict(label="Init range guess (m)",    sweep="P12", out=False, cat=False),
    "in_init_x":              dict(label="Init x guess (m)",        sweep="P13", out=False, cat=False),
    "in_init_y":              dict(label="Init y guess (m)",        sweep="P14", out=False, cat=False),
    "in_init_vx":             dict(label="Init vx guess (m/s)",     sweep="P15", out=False, cat=False),
    "in_init_vy":             dict(label="Init vy guess (m/s)",     sweep="P16", out=False, cat=False),
    "in_P0_range":            dict(label="P0 range (m²)",           sweep="P17", out=False, cat=False),
    "in_P0_x":                dict(label="P0 x (m²)",               sweep="P18", out=False, cat=False),
    "in_P0_y":                dict(label="P0 y (m²)",               sweep="P19", out=False, cat=False),
    "in_P0_vx":               dict(label="P0 vx ((m/s)²)",          sweep="P20", out=False, cat=False),
    "in_P0_vy":               dict(label="P0 vy ((m/s)²)",          sweep="P21", out=False, cat=False),
    "in_sensor_range0":       dict(label="Sensor range0 (m)",       sweep="P22", out=False, cat=False),
    "in_sensor_az0":          dict(label="Sensor azimuth0 (°)",     sweep="P23", out=False, cat=False),
    "in_sensor_speed":        dict(label="Sensor speed (m/s)",      sweep="P24", out=False, cat=False),
    "in_sensor_accel":        dict(label="Sensor accel (m/s²)",     sweep="P25", out=False, cat=False),
    "in_sensor_yaw":          dict(label="Sensor yaw (°)",          sweep="P26", out=False, cat=False),
    "in_sensor_yawrate":      dict(label="Sensor yaw rate (°/s)",   sweep="P27", out=False, cat=False),
    "in_sensor_yawraterate":  dict(label="Sensor yaw rate rate (deg/s^2)",sweep="P28",out=False,cat=False),
    "in_target_range0":       dict(label="Target range0 (m)",       sweep="P29", out=False, cat=False),
    "in_target_az0":          dict(label="Target azimuth0 (°)",     sweep="P30", out=False, cat=False),
    "in_target_speed":        dict(label="Target speed (m/s)",      sweep="P31", out=False, cat=False),
    "in_target_accel":        dict(label="Target accel (m/s²)",     sweep="P32", out=False, cat=False),
    "in_target_yaw":          dict(label="Target yaw (°)",          sweep="P33", out=False, cat=False),
    "in_target_yawrate":      dict(label="Target yaw rate (°/s)",   sweep="P34", out=False, cat=False),
    "in_target_yawraterate":  dict(label="Target yaw rate rate (deg/s^2)",sweep="P35",out=False,cat=False),
    "in_sensor_turnrate":     dict(label="Sensor turn rate (°/s)",  sweep="P36", out=False, cat=False),
    "in_target_turnrate":     dict(label="Target turn rate (°/s)",  sweep="P37", out=False, cat=False),
    # Categoricals (stored as strings)
    "in_state_meas":          dict(label="State+Meas config",       sweep="C12", out=False, cat=True),
    "in_sensor_traj":         dict(label="Sensor trajectory",       sweep="C3",  out=False, cat=True),
    "in_target_traj":         dict(label="Target trajectory",       sweep="C4",  out=False, cat=True),
    # Outputs
    "rmse_rangex":            dict(label="RMSE RangeX (m)",         sweep=None,  out=True,  cat=False),
    "rmse_rangey":            dict(label="RMSE RangeY (m)",         sweep=None,  out=True,  cat=False),
    "rmse_vx":                dict(label="RMSE Vx (m/s)",           sweep=None,  out=True,  cat=False),
    "rmse_vy":                dict(label="RMSE Vy (m/s)",           sweep=None,  out=True,  cat=False),
    "rmse_azimuth":           dict(label="RMSE Azimuth (°)",        sweep=None,  out=True,  cat=False),
}

INPUT_COLS  = [c for c, m in COL_META.items() if not m["out"]]
OUTPUT_COLS = [c for c, m in COL_META.items() if m["out"]]
NUMERIC_IN  = [c for c, m in COL_META.items() if not m["out"] and not m["cat"]]

# ── Apply sweep to config ─────────────────────────────────────────────────────
def apply_sweep(cfg: dict, sid: str, value) -> None:
    sp = cfg["scenario_parameters"]
    ip = cfg["initial_states"]
    tp = cfg["trajectory_parameters"]
    if   sid == "P01": cfg["simulation_parameters"]["steps"]            = int(value)
    elif sid == "P02": cfg["simulation_parameters"]["time_step"]        = float(value)
    elif sid == "P03": sp["meas_noise_actual"]["range"]                 = float(value)
    elif sid == "P04": sp["meas_noise_actual"]["azimuth"]               = float(value)
    elif sid == "P05": sp["meas_noise_actual"]["doppler"]               = float(value)
    elif sid == "P06": sp["process_noise"]                              = float(value)
    elif sid == "P07": sp["meas_noise_guess"]["range"]                  = float(value)
    elif sid == "P08": sp["meas_noise_guess"]["azimuth"]                = float(value)
    elif sid == "P09": sp["meas_noise_guess"]["doppler"]                = float(value)
    elif sid == "P10": sp["meas_noise_guess"]["rangex"]                 = float(value)
    elif sid == "P11": sp["meas_noise_guess"]["rangey"]                 = float(value)
    elif sid == "P12": sp["initial_state_guess"]["range"]               = float(value)
    elif sid == "P13": sp["initial_state_guess"]["x"]                   = float(value)
    elif sid == "P14": sp["initial_state_guess"]["y"]                   = float(value)
    elif sid == "P15": sp["initial_state_guess"]["vx"]                  = float(value)
    elif sid == "P16": sp["initial_state_guess"]["vy"]                  = float(value)
    elif sid == "P17": sp["initial_covariance_guess"]["range"]          = float(value)
    elif sid == "P18": sp["initial_covariance_guess"]["x"]              = float(value)
    elif sid == "P19": sp["initial_covariance_guess"]["y"]              = float(value)
    elif sid == "P20": sp["initial_covariance_guess"]["vx"]             = float(value)
    elif sid == "P21": sp["initial_covariance_guess"]["vy"]             = float(value)
    elif sid == "P22": ip["sensor"]["range"]                            = float(value)
    elif sid == "P23": ip["sensor"]["azimuth"]                          = float(value)
    elif sid == "P24": ip["sensor"]["speed"]                            = float(value)
    elif sid == "P25": ip["sensor"]["acceleration"]                     = float(value)
    elif sid == "P26": ip["sensor"]["yaw"]                              = float(value)
    elif sid == "P27": ip["sensor"]["yawrate"]                          = float(value)
    elif sid == "P28": ip["sensor"]["yawraterate"]                      = float(value)
    elif sid == "P29": ip["target"]["range"]                            = float(value)
    elif sid == "P30": ip["target"]["azimuth"]                          = float(value)
    elif sid == "P31": ip["target"]["speed"]                            = float(value)
    elif sid == "P32": ip["target"]["acceleration"]                     = float(value)
    elif sid == "P33": ip["target"]["yaw"]                              = float(value)
    elif sid == "P34": ip["target"]["yawrate"]                          = float(value)
    elif sid == "P35": ip["target"]["yawraterate"]                      = float(value)
    elif sid == "P36":
        tp["selection"]["sensor"] = "STraj"
        tp["STraj"]["sensor"]["turnRate"]        = float(value)
        tp["FreeTurnTraj"]["sensor"]["turnRate"] = float(value)
    elif sid == "P37":
        tp["selection"]["target"] = "STraj"
        tp["STraj"]["target"]["turnRate"]        = float(value)
        tp["FreeTurnTraj"]["target"]["turnRate"] = float(value)
    elif sid == "C12":
        for name, states, meas, ns in COMBO_STATES:
            if name == value:
                sp["states"]      = states
                sp["measurements"] = meas
                sp["num_states"]  = ns
                break
    elif sid == "C3": tp["selection"]["sensor"] = value
    elif sid == "C4": tp["selection"]["target"] = value

# ── Extract param snapshot for CSV ───────────────────────────────────────────
def extract_params(cfg: dict) -> dict:
    sp = cfg["scenario_parameters"]
    ip = cfg["initial_states"]
    tp = cfg["trajectory_parameters"]
    return {
        "in_steps":              cfg["simulation_parameters"]["steps"],
        "in_time_step":          cfg["simulation_parameters"]["time_step"],
        "in_range_noise_act":    sp["meas_noise_actual"]["range"],
        "in_az_noise_act":       sp["meas_noise_actual"]["azimuth"],
        "in_dop_noise_act":      sp["meas_noise_actual"]["doppler"],
        "in_process_noise_q":    sp["process_noise"],
        "in_range_noise_gss":    sp["meas_noise_guess"]["range"],
        "in_az_noise_gss":       sp["meas_noise_guess"]["azimuth"],
        "in_dop_noise_gss":      sp["meas_noise_guess"]["doppler"],
        "in_rangex_noise_gss":   sp["meas_noise_guess"]["rangex"],
        "in_rangey_noise_gss":   sp["meas_noise_guess"]["rangey"],
        "in_init_range":         sp["initial_state_guess"]["range"],
        "in_init_x":             sp["initial_state_guess"]["x"],
        "in_init_y":             sp["initial_state_guess"]["y"],
        "in_init_vx":            sp["initial_state_guess"]["vx"],
        "in_init_vy":            sp["initial_state_guess"]["vy"],
        "in_P0_range":           sp["initial_covariance_guess"]["range"],
        "in_P0_x":               sp["initial_covariance_guess"]["x"],
        "in_P0_y":               sp["initial_covariance_guess"]["y"],
        "in_P0_vx":              sp["initial_covariance_guess"]["vx"],
        "in_P0_vy":              sp["initial_covariance_guess"]["vy"],
        "in_sensor_range0":      ip["sensor"]["range"],
        "in_sensor_az0":         ip["sensor"]["azimuth"],
        "in_sensor_speed":       ip["sensor"]["speed"],
        "in_sensor_accel":       ip["sensor"]["acceleration"],
        "in_sensor_yaw":         ip["sensor"]["yaw"],
        "in_sensor_yawrate":     ip["sensor"]["yawrate"],
        "in_sensor_yawraterate": ip["sensor"]["yawraterate"],
        "in_target_range0":      ip["target"]["range"],
        "in_target_az0":         ip["target"]["azimuth"],
        "in_target_speed":       ip["target"]["speed"],
        "in_target_accel":       ip["target"]["acceleration"],
        "in_target_yaw":         ip["target"]["yaw"],
        "in_target_yawrate":     ip["target"]["yawrate"],
        "in_target_yawraterate": ip["target"]["yawraterate"],
        "in_sensor_turnrate":    tp["STraj"]["sensor"]["turnRate"],
        "in_target_turnrate":    tp["STraj"]["target"]["turnRate"],
        "in_state_meas":         f"{sp['states']}+{sp['measurements']}",
        "in_sensor_traj":        tp["selection"]["sensor"],
        "in_target_traj":        tp["selection"]["target"],
    }

# ── Run one simulation ────────────────────────────────────────────────────────
_NAN = dict(rmse_rangex=float("nan"), rmse_rangey=float("nan"),
            rmse_vx=float("nan"), rmse_vy=float("nan"),
            rmse_azimuth=float("nan"))

def run_one(cfg: dict) -> dict:
    """Run one OMS + KF simulation and return error metrics.

    Metrics are computed for both CV (states="1") and CA (states="2") models.
    All other state types return NaN metrics with run_failed=False.

    Each call uses a private temp directory for all intermediate files
    (config JSON, measurements.ftr, sensor_trajectory.ftr,
    target_trajectory.ftr).  This prevents cross-contamination when two
    separate OAT processes run simultaneously: they write to different
    filesystem locations and never overwrite each other's files.
    """
    import object_motion_simulator as _oms_mod
    import kalman_filter as _kf_mod

    tmp_out = tempfile.mkdtemp(prefix="oat_run_")
    tmp_cfg = os.path.join(tmp_out, "config.json")
    _orig_oms_out = _oms_mod._OUTPUT_DIR
    _orig_kf_out  = _kf_mod._OUTPUT_DIR

    try:
        with open(tmp_cfg, "w") as f:
            json.dump(cfg, f)

        # Redirect both modules' output/input paths to the temp dir so that
        # concurrent processes write to independent locations on disk.
        _oms_mod._OUTPUT_DIR = tmp_out
        _kf_mod._OUTPUT_DIR  = tmp_out

        oms = ObjectMotionSimulator(tmp_cfg)
        oms.main(screenprint=False, plot=False)
        if not oms.status:
            return dict(_NAN, run_failed=True)

        target = pd.read_feather(os.path.join(tmp_out, "target_trajectory.ftr"))
        sensor = pd.read_feather(os.path.join(tmp_out, "sensor_trajectory.ftr"))

        kf = KalmanFilter(tmp_cfg)
        est = kf.main()
        if not kf.status or est is None or len(est) == 0:
            return dict(_NAN, run_failed=True)

    except Exception:
        return dict(_NAN, run_failed=True)
    finally:
        # Always restore module globals and clean up the temp dir, even on error.
        _oms_mod._OUTPUT_DIR = _orig_oms_out
        _kf_mod._OUTPUT_DIR  = _orig_kf_out
        shutil.rmtree(tmp_out, ignore_errors=True)

    state_key = cfg["scenario_parameters"]["states"]
    # Only CV ("1") and CA ("2") produce comparable vx/vy/x/y outputs.
    if state_key not in ("1", "2"):
        return dict(_NAN, run_failed=False)

    n = min(len(est), len(target), len(sensor))
    e  = est.iloc[:n]
    tg = target.iloc[:n]
    sn = sensor.iloc[:n]

    try:
        rx_true = tg["x"].values - sn["x"].values
        ry_true = tg["y"].values - sn["y"].values
        # Filter x/y estimate is world-frame position; subtract sensor to get rangex/y.
        rx_est  = e["x"].values  - sn["x"].values
        ry_est  = e["y"].values  - sn["y"].values

        rmse_rx = math.sqrt(float(np.mean((rx_est - rx_true) ** 2)))
        rmse_ry = math.sqrt(float(np.mean((ry_est - ry_true) ** 2)))
        # vx/vy estimates are world-frame after the doppler measurement model fix.
        rmse_vx = math.sqrt(float(np.mean((e["vx"].values - tg["vx"].values) ** 2)))
        rmse_vy = math.sqrt(float(np.mean((e["vy"].values - tg["vy"].values) ** 2)))

        az_true = np.degrees(np.arctan2(ry_true, rx_true))
        az_est  = np.degrees(np.arctan2(ry_est,  rx_est))
        az_err  = (az_est - az_true + 180.0) % 360.0 - 180.0
        rmse_az = math.sqrt(float(np.mean(az_err ** 2)))

        return dict(rmse_rangex=rmse_rx, rmse_rangey=rmse_ry,
                    rmse_vx=rmse_vx, rmse_vy=rmse_vy,
                    rmse_azimuth=rmse_az, run_failed=False)
    except Exception:
        return dict(_NAN, run_failed=True)

# ── Main OAT loop ─────────────────────────────────────────────────────────────
def main_run(
    quick: int | None = None,
    baseline: dict | None = None,
    sweep_filter: str | None = None,
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Run the OAT sweep and save results to CSV.

    @param quick: If set, subsample each sweep to at most N points.
    @param baseline: Config dict to use as the baseline (defaults to BASELINE).
    @param sweep_filter: If set, run only the sweep whose id matches this string
        (e.g. ``"P32"``).  Useful for quick targeted validation runs.
    @param csv_path: Output CSV path (defaults to the global CSV).
    """
    if baseline is None:
        baseline = BASELINE
    if csv_path is None:
        csv_path = CSV

    active_sweeps = [sw for sw in SWEEPS
                     if sweep_filter is None or sw["id"] == sweep_filter]

    rows: list[dict] = []
    total_runs = 0
    for sw in active_sweeps:
        vals = sw["values"]
        if quick:
            step = max(1, len(vals) // quick)
            vals = vals[::step][:quick]
        total_runs += len(vals)
    total_runs += 1  # baseline

    print(f"OAT run: {total_runs} simulations"
          + (f" (sweep={sweep_filter})" if sweep_filter else ""))
    t0 = time.time()
    done = 0

    def _tick(label: str):
        nonlocal done
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (total_runs - done) if done > 0 else 0
        print(f"\r  [{done:4d}/{total_runs}]  {label[:55]:<55}"
              f"  elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m", end="", flush=True)

    # ── Baseline run ──────────────────────────────────────────────────────────
    bl_cfg = copy.deepcopy(baseline)
    bl_metrics = run_one(bl_cfg)
    bl_row = dict(
        swept_param_id="BASELINE", swept_param_label="Baseline",
        swept_param_group="Baseline", param_value=float("nan"),
        param_value_str="baseline",
        **extract_params(bl_cfg), **bl_metrics,
    )
    rows.append(bl_row)
    _tick("Baseline")

    # ── Parameter sweeps ──────────────────────────────────────────────────────
    for sw in active_sweeps:
        vals = sw["values"]
        if quick:
            step = max(1, len(vals) // quick)
            vals = vals[::step][:quick]
        for v in vals:
            cfg = copy.deepcopy(baseline)
            apply_sweep(cfg, sw["id"], v)
            metrics = run_one(cfg)
            pval = float(v) if not isinstance(v, str) else float("nan")
            rows.append(dict(
                swept_param_id=sw["id"],
                swept_param_label=sw["label"],
                swept_param_group=sw["group"],
                param_value=pval,
                param_value_str=str(v),
                **extract_params(cfg), **metrics,
            ))
            _tick(f"{sw['id']} {sw['label']} = {v!r}")

    print()  # newline after progress
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    elapsed = time.time() - t0
    n_fail  = int(df["run_failed"].sum())
    print(f"Done. {len(df)} rows, {n_fail} failures. "
          f"Saved to {csv_path}. ({elapsed/60:.1f} min)")
    return df

# ── HTML generation ───────────────────────────────────────────────────────────
def build_html(df: pd.DataFrame) -> None:
    for c in OUTPUT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    records = json.loads(df.to_json(orient="records"))
    group_colours = {
        "Baseline":          "#888888",
        "Simulation":        "#4FC3F7",
        "Meas Noise Actual": "#F06292",
        "Process Noise":     "#FFB74D",
        "Meas Noise Guess":  "#AED581",
        "Initial State":     "#CE93D8",
        "Init Covariance":   "#80DEEA",
        "Sensor Init":       "#FFCC02",
        "Target Init":       "#FF8A65",
        "Turn Rates":        "#A5D6A7",
        "Categorical":       "#EF9A9A",
    }

    # ── Static HTML head (no Python injections) ───────────────────────────────
    html = (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="UTF-8">\n'
        '<title>OAT Sensitivity Analysis — Kalman Filter</title>\n'
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        '<style>\n'
        '*{box-sizing:border-box;margin:0;padding:0}\n'
        'body{background:#080F1A;color:#C8D8E8;font-family:\'Segoe UI\',system-ui,sans-serif;font-size:13px}\n'
        'h1{padding:14px 20px;font-size:17px;color:#E8A830;letter-spacing:1px;'
            'border-bottom:1px solid #1E4060;background:#0A1628}\n'
        '.tabs{display:flex;gap:0;background:#0A1628;border-bottom:2px solid #1E4060;padding:0 16px}\n'
        '.tab-btn{padding:10px 24px;background:transparent;border:none;color:#8BA3B0;font-size:13px;'
            'cursor:pointer;border-bottom:3px solid transparent;transition:all .2s;letter-spacing:.5px}\n'
        '.tab-btn.active{color:#E8A830;border-bottom-color:#E8A830}\n'
        '.tab-btn:hover{color:#C8D8E8}\n'
        '.tab-pane{display:none;padding:12px 16px}\n'
        '.tab-pane.active{display:block}\n'
        '.ctrl-bar{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;'
            'background:#0D2235;padding:10px 14px;border-radius:6px;border:1px solid #1E4060;margin-bottom:10px}\n'
        '.ctrl-grp{display:flex;flex-direction:column;gap:4px;min-width:160px}\n'
        '.ctrl-grp label{font-size:11px;color:#6B8CA4;text-transform:uppercase;letter-spacing:.5px}\n'
        'select{background:#0A1628;color:#C8D8E8;border:1px solid #1E4060;border-radius:4px;'
            'padding:5px 8px;font-size:12px;cursor:pointer;outline:none}\n'
        'select:focus{border-color:#E8A830}\n'
        '.radio-grp{display:flex;gap:12px;align-items:center;padding:4px 0}\n'
        '.radio-grp label{display:flex;align-items:center;gap:5px;cursor:pointer;color:#C8D8E8;font-size:12px}\n'
        'input[type=radio]{accent-color:#E8A830}\n'
        '.plot-wrap{border:1px solid #1E4060;border-radius:6px;overflow:hidden}\n'
        '.info-bar{font-size:11px;color:#6B8CA4;padding:6px 0;margin-top:6px}\n'
        '#ctx-menu{position:fixed;display:none;background:#0D2235;border:1px solid #1E4060;'
            'border-radius:5px;padding:4px 0;z-index:9999;box-shadow:0 4px 14px #000c;min-width:190px}\n'
        '.ctx-item{padding:9px 16px;cursor:pointer;color:#C8D8E8;font-size:12px;user-select:none}'
            '.ctx-item:hover{background:#1E4060;color:#E8A830}\n'
        '.ctx-divider{border-top:1px solid #1E4060;margin:4px 0}\n'
        '</style>\n'
        '</head>\n'
        '<body>\n'
    )
    html += f'<h1>OAT Sensitivity Analysis &nbsp;&middot;&nbsp; Kalman Filter Tracker &nbsp;&middot;&nbsp; {len(df)} runs</h1>\n'
    html += (
        '<div class="tabs">\n'
        '  <button class="tab-btn active" onclick="showTab(\'tab2d\',this)">2D Explorer</button>\n'
        '  <button class="tab-btn"        onclick="showTab(\'tab3d\',this)">3D Surface</button>\n'
        '</div>\n'
        '<div id="tab2d" class="tab-pane active">\n'
        '  <div class="ctrl-bar">\n'
        '    <div class="ctrl-grp"><label>X axis</label>'
            '<select id="sel2x" onchange="draw2d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Y axis</label>'
            '<select id="sel2y" onchange="draw2d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Colour</label>'
            '<select id="sel2c" onchange="draw2d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Marker symbol</label>'
            '<select id="sel2m" onchange="draw2d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Plot type</label>\n'
        '      <div class="radio-grp">\n'
        '        <label><input type="radio" name="mode2d" value="lines+markers"'
            ' checked onchange="draw2d()"> Lines</label>\n'
        '        <label><input type="radio" name="mode2d" value="markers"'
            ' onchange="draw2d()"> Scatter</label>\n'
        '      </div>\n'
        '    </div>\n'
        '    <div class="ctrl-grp"><label>&nbsp;</label>\n'
        '      <button id="btn-sweep-only" onclick="toggleXSweep()"'
            ' style="background:#0A1628;color:#8BA3B0;border:1px solid #1E4060;'
            'border-radius:4px;padding:6px 12px;font-size:12px;cursor:pointer;'
            'transition:all .2s;white-space:nowrap">'
            'Hide other sweeps</button>\n'
        '    </div>\n'
        '  </div>\n'
        '  <div class="plot-wrap"><div id="plot2d" style="height:620px"></div></div>\n'
        '  <div class="info-bar" id="info2d"></div>\n'
        '</div>\n'
        '<div id="tab3d" class="tab-pane">\n'
        '  <div class="ctrl-bar">\n'
        '    <div class="ctrl-grp"><label>X axis</label>'
            '<select id="sel3x" onchange="draw3d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Y axis</label>'
            '<select id="sel3y" onchange="draw3d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Z axis</label>'
            '<select id="sel3z" onchange="draw3d()"></select></div>\n'
        '    <div class="ctrl-grp"><label>Surface colour</label>'
            '<select id="sel3c" onchange="draw3d()"></select></div>\n'
        '  </div>\n'
        '  <div class="plot-wrap"><div id="plot3d" style="height:680px"></div></div>\n'
        '  <div class="info-bar" id="info3d">'
            'Surface (go.Surface + grid) when X and Y are numeric params from different sweeps; '
            'otherwise scatter-line grid per sweep.</div>\n'
        '</div>\n'
        '<script>\n'
    )

    # ── Inject Python data as JS constants ────────────────────────────────────
    html += 'const DATA     = ' + json.dumps(records, allow_nan=False, default=lambda x: None) + ';\n'
    html += 'const COL_META = ' + json.dumps(COL_META) + ';\n'
    html += 'const IN_COLS  = ' + json.dumps(INPUT_COLS) + ';\n'
    html += 'const OUT_COLS = ' + json.dumps(OUTPUT_COLS) + ';\n'
    html += 'const NUM_IN   = ' + json.dumps(NUMERIC_IN) + ';\n'
    html += 'const GRP_COL  = ' + json.dumps(group_colours) + ';\n'
    html += (
        'DATA.forEach((d,i)=>{\n'
        '  d._idx=i;\n'
        '  if(!d.run_failed){\n'
        '    const blown=OUT_COLS.some(c=>{ const v=d[c]; return v!=null && isFinite(v) && Math.abs(v)>1e5; });\n'
        '    if(blown) d.run_failed=true;\n'
        '  }\n'
        '});\n'
    )

    # ── Pure JavaScript body (regular string — no f-string, no {{ escaping) ──
    html += r"""
let hiddenPoints = new Set(), traceRows = [], hoveredPoint = null;
let showXSweepOnly = false;

function toggleXSweep() {
  showXSweepOnly = !showXSweepOnly;
  const btn = document.getElementById('btn-sweep-only');
  if (showXSweepOnly) {
    btn.style.background    = '#1E4060';
    btn.style.color         = '#E8A830';
    btn.style.borderColor   = '#E8A830';
    btn.textContent = 'Show all sweeps';
  } else {
    btn.style.background    = '#0A1628';
    btn.style.color         = '#8BA3B0';
    btn.style.borderColor   = '#1E4060';
    btn.textContent = 'Hide other sweeps';
  }
  draw2d();
}

const ALL_COLS = [...IN_COLS, ...OUT_COLS];
const SYMBOLS = ['circle','square','diamond','cross','x','triangle-up',
                 'triangle-down','pentagon','hexagram','star'];
const PLASMA = ['#0d0887','#46039f','#7201a8','#9c179e','#bd3786',
                '#d8576b','#ed7953','#fb9f3a','#fdcb26','#f0f921'];

function showTab(id, btn) {
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b  => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  if (id === 'tab3d') draw3d();
}

function makeOpts(selId, cols, def) {
  const sel = document.getElementById(selId);
  sel.innerHTML = '';
  cols.forEach(c => {
    const o = document.createElement('option');
    o.value = c;
    o.textContent = (COL_META[c] && COL_META[c].label) ? COL_META[c].label : c;
    if (c === def) o.selected = true;
    sel.appendChild(o);
  });
}

function initSelects() {
  makeOpts('sel2x', ALL_COLS, 'in_process_noise_q');
  makeOpts('sel2y', OUT_COLS, 'rmse_rangex');
  makeOpts('sel2c', ['(group)', ...ALL_COLS], '(group)');
  makeOpts('sel2m', ['(none)','swept_param_id','in_sensor_traj','in_target_traj','in_state_meas'], '(none)');
  makeOpts('sel3x', NUM_IN, 'in_process_noise_q');
  makeOpts('sel3y', NUM_IN, 'in_range_noise_act');
  makeOpts('sel3z', OUT_COLS, 'rmse_rangex');
  makeOpts('sel3c', OUT_COLS, 'rmse_rangey');
}

function lbl(col) { return (COL_META[col] && COL_META[col].label) ? COL_META[col].label : col; }

function fmt(v) {
  if (v == null || (typeof v === 'number' && isNaN(v))) return 'N/A';
  if (typeof v === 'string') return v;
  return (Math.abs(v) < 0.001 || Math.abs(v) > 9999) ? v.toExponential(3) : v.toPrecision(4);
}

function isLogParam(col) {
  return ['in_range_noise_act','in_az_noise_act','in_dop_noise_act','in_process_noise_q',
          'in_range_noise_gss','in_az_noise_gss','in_dop_noise_gss',
          'in_rangex_noise_gss','in_rangey_noise_gss',
          'in_P0_range','in_P0_x','in_P0_y','in_P0_vx','in_P0_vy'].includes(col);
}

// ── 2D chart ─────────────────────────────────────────────────────────────────
function draw2d() {
  const xCol = document.getElementById('sel2x').value;
  const yCol = document.getElementById('sel2y').value;
  const cCol = document.getElementById('sel2c').value;
  const mCol = document.getElementById('sel2m').value;
  const mode = document.querySelector('input[name="mode2d"]:checked').value;

  const xSweepId = COL_META[xCol] && COL_META[xCol].sweep;

  const rows = DATA.filter(d => {
    if (d.run_failed) return false;
    if (hiddenPoints.has(d._idx)) return false;
    if (d[xCol] == null || isNaN(d[xCol])) return false;
    if (d[yCol] == null || isNaN(d[yCol])) return false;
    if (showXSweepOnly && xSweepId) {
      // keep only the sweep that owns this x-axis parameter, plus the baseline run
      if (d.swept_param_id !== xSweepId && d.swept_param_id !== 'BASELINE') return false;
    }
    return true;
  });

  if (!rows.length) {
    Plotly.purge('plot2d');
    document.getElementById('info2d').textContent = 'No data for selected columns.';
    return;
  }

  const useGroup = (cCol === '(group)');
  const useSymbol = (mCol !== '(none)');
  const cMeta = COL_META[cCol];
  const cIsNum = !useGroup && cMeta && !cMeta.cat;

  // Build categorical colour map (for non-numeric colour columns)
  const catColorMap = {};
  if (!useGroup && !cIsNum) {
    const cats = [...new Set(rows.map(d => String(d[cCol] != null ? d[cCol] : 'null')))].sort();
    cats.forEach((c, i) => {
      catColorMap[c] = PLASMA[Math.floor(i / Math.max(cats.length - 1, 1) * (PLASMA.length - 1))];
    });
  }

  const symMap = {}; let symIdx = 0;
  if (useSymbol) rows.forEach(d => {
    const sk = String(d[mCol] != null ? d[mCol] : 'null');
    if (!(sk in symMap)) symMap[sk] = SYMBOLS[symIdx++ % SYMBOLS.length];
  });

  // Group by sweep to insert null breaks (prevents cross-sweep lines)
  const sweepGroups = {};
  rows.forEach(d => {
    const k = d.swept_param_id || 'BASELINE';
    if (!sweepGroups[k]) sweepGroups[k] = [];
    sweepGroups[k].push(d);
  });
  const sweepKeys = Object.keys(sweepGroups).sort();

  // Build single flat arrays; insert null breaks between sweeps for line mode
  const xArr = [], yArr = [], colorArr = [], symArr = [], textArr = [], idxArr = [];
  const isLineMode = mode.includes('lines');

  sweepKeys.forEach((k, ki) => {
    const g = sweepGroups[k];
    g.forEach(d => {
      xArr.push(d[xCol]);
      yArr.push(d[yCol]);
      const c = useGroup   ? (GRP_COL[d.swept_param_group] || '#888888') :
                cIsNum     ? d[cCol] :
                             catColorMap[String(d[cCol] != null ? d[cCol] : 'null')];
      colorArr.push(c);
      symArr.push(useSymbol ? (symMap[String(d[mCol] != null ? d[mCol] : 'null')] || 'circle') : 'circle');
      textArr.push(
        'Param: ' + d.swept_param_id + ' (' + d.swept_param_group + ')<br>' +
        'Swept val: ' + d.param_value_str + '<br>' +
        lbl(xCol) + ': ' + fmt(d[xCol]) + '<br>' +
        lbl(yCol) + ': ' + fmt(d[yCol]) +
        (cIsNum ? '<br>' + lbl(cCol) + ': ' + fmt(d[cCol]) : '')
      );
      idxArr.push(d._idx);
    });
    if (isLineMode && ki < sweepKeys.length - 1) {
      xArr.push(null); yArr.push(null); colorArr.push(null);
      symArr.push('circle'); textArr.push(''); idxArr.push(null);
    }
  });

  traceRows = [idxArr]; // single trace at curveNumber 0

  const trace = {
    x: xArr, y: yArr, mode,
    type: 'scatter',
    marker: {
      color: colorArr,
      colorscale: cIsNum ? 'Plasma' : undefined,
      showscale: cIsNum,
      colorbar: cIsNum ? {title: {text: lbl(cCol), side: 'right'}, thickness: 12} : undefined,
      size: mode.includes('markers') ? 7 : 5,
      symbol: useSymbol ? symArr : 'circle',
    },
    line: {width: 1.5},
    text: textArr,
    hoverinfo: 'text',
    showlegend: false,
  };

  const layout = {
    paper_bgcolor: '#080F1A', plot_bgcolor: '#0A1628',
    font: {color: '#C8D8E8', size: 12},
    showlegend: false,
    xaxis: {
      title: {text: lbl(xCol)}, gridcolor: '#1E4060', zerolinecolor: '#1E4060',
      type: isLogParam(xCol) ? 'log' : '-', autorange: true,
    },
    yaxis: {
      title: {text: lbl(yCol)}, gridcolor: '#1E4060', zerolinecolor: '#1E4060',
      type: isLogParam(yCol) ? 'log' : '-', autorange: true,
    },
    margin: {l: 60, r: 20, t: 20, b: 60},
    hovermode: 'closest',
  };

  Plotly.react('plot2d', [trace], layout, {responsive: true});
  const hiddenCount = hiddenPoints.size;
  document.getElementById('info2d').textContent =
    rows.length + ' data points shown' +
    (hiddenCount ? ' · ' + hiddenCount + ' hidden (right-click a point to hide; menu has Reset)' : '') +
    '. Failed runs excluded.';
}

// ── 3D chart ─────────────────────────────────────────────────────────────────
function draw3d() {
  const xCol = document.getElementById('sel3x').value;
  const yCol = document.getElementById('sel3y').value;
  const zCol = document.getElementById('sel3z').value;
  const cCol = document.getElementById('sel3c').value;

  const xMeta = COL_META[xCol]; const yMeta = COL_META[yCol];
  const xSweep = xMeta && xMeta.sweep; const ySweep = yMeta && yMeta.sweep;

  const canSurface = xSweep && ySweep && xSweep !== ySweep &&
                     !(xMeta && xMeta.cat) && !(yMeta && yMeta.cat);

  const sceneLayout = {
    paper_bgcolor: '#080F1A', font: {color: '#C8D8E8', size: 12},
    showlegend: false,
    scene: {
      xaxis: {title: lbl(xCol), gridcolor: '#1E4060', backgroundcolor: '#0A1628',
               type: isLogParam(xCol) ? 'log' : '-'},
      yaxis: {title: lbl(yCol), gridcolor: '#1E4060', backgroundcolor: '#0A1628',
               type: isLogParam(yCol) ? 'log' : '-'},
      zaxis: {title: lbl(zCol), gridcolor: '#1E4060', backgroundcolor: '#0A1628'},
      bgcolor: '#0A1628',
    },
    margin: {l: 0, r: 0, t: 20, b: 0},
  };

  if (canSurface) {
    const xRows = DATA.filter(d => !d.run_failed && !hiddenPoints.has(d._idx) &&
      d.swept_param_id === xSweep &&
      d[xCol] != null && !isNaN(d[xCol]) && d[zCol] != null && !isNaN(d[zCol])
    ).sort((a, b) => a[xCol] - b[xCol]);

    const yRows = DATA.filter(d => !d.run_failed && !hiddenPoints.has(d._idx) &&
      d.swept_param_id === ySweep &&
      d[yCol] != null && !isNaN(d[yCol]) && d[zCol] != null && !isNaN(d[zCol])
    ).sort((a, b) => a[yCol] - b[yCol]);

    const bl = DATA.find(d => d.swept_param_id === 'BASELINE') || {};
    const zBase = (bl[zCol] != null && !isNaN(bl[zCol])) ? bl[zCol] : 0;
    const cBase = (bl[cCol] != null && !isNaN(bl[cCol])) ? bl[cCol] : 0;

    if (xRows.length < 2 || yRows.length < 2) { fallbackScatter(xCol,yCol,zCol,cCol,sceneLayout); return; }

    const xs = xRows.map(d => d[xCol]);
    const ys = yRows.map(d => d[yCol]);
    const zGrid = [], cGrid = [], htGrid = [];

    for (let j = 0; j < ys.length; j++) {
      const zRow = [], cRow = [], htRow = [];
      const zy = (yRows[j][zCol] != null && !isNaN(yRows[j][zCol])) ? yRows[j][zCol] : zBase;
      const cy = (yRows[j][cCol] != null && !isNaN(yRows[j][cCol])) ? yRows[j][cCol] : cBase;
      for (let i = 0; i < xs.length; i++) {
        const zx = (xRows[i][zCol] != null && !isNaN(xRows[i][zCol])) ? xRows[i][zCol] : zBase;
        const cx = (xRows[i][cCol] != null && !isNaN(xRows[i][cCol])) ? xRows[i][cCol] : cBase;
        const zv = zx + zy - zBase, cv = cx + cy - cBase;
        zRow.push(zv); cRow.push(cv);
        htRow.push(lbl(xCol)+': '+fmt(xs[i])+'<br>'+lbl(yCol)+': '+fmt(ys[j])+'<br>'+
                   lbl(zCol)+': '+fmt(zv)+'<br>'+lbl(cCol)+': '+fmt(cv)+' (colour)');
      }
      zGrid.push(zRow); cGrid.push(cRow); htGrid.push(htRow);
    }

    const surf = {
      type: 'surface', x: xs, y: ys, z: zGrid,
      surfacecolor: cCol !== zCol ? cGrid : undefined,
      colorscale: 'Plasma', opacity: 0.88,
      contours: {
        x: {show: true, highlight: true, color: '#1E4060', width: 1},
        y: {show: true, highlight: true, color: '#1E4060', width: 1},
        z: {show: false},
      },
      colorbar: {
        title: {text: lbl(cCol), side: 'right'}, thickness: 14,
        bgcolor: '#0D2235', bordercolor: '#1E4060', tickfont: {color: '#C8D8E8'},
      },
      text: htGrid, hovertemplate: '%{text}<extra></extra>', showscale: true,
    };

    Plotly.react('plot3d', [surf], sceneLayout, {responsive: true});
    document.getElementById('info3d').textContent =
      'Surface: additive OAT model. X=' + xRows.length + ' pts x Y=' + yRows.length +
      ' pts. Grid lines show sweep directions. Z=' + lbl(zCol) + ', Colour=' + lbl(cCol) + '.';
  } else {
    fallbackScatter(xCol, yCol, zCol, cCol, sceneLayout);
  }
}

function fallbackScatter(xCol, yCol, zCol, cCol, layout) {
  const rows = DATA.filter(d => !d.run_failed && !hiddenPoints.has(d._idx) &&
    d[xCol] != null && !isNaN(d[xCol]) &&
    d[yCol] != null && !isNaN(d[yCol]) &&
    d[zCol] != null && !isNaN(d[zCol])
  );
  if (!rows.length) { Plotly.purge('plot3d'); return; }

  const cVals = rows.map(d => d[cCol]).filter(v => v != null && !isNaN(v));
  const cLo = Math.min(...cVals), cHi = Math.max(...cVals);

  const groups = {};
  rows.forEach(d => {
    const k = d.swept_param_id;
    if (!groups[k]) groups[k] = [];
    groups[k].push(d);
  });
  const keys = Object.keys(groups).sort();

  const traces = keys.map((k, ki) => {
    const g = groups[k].slice().sort((a, b) => (a.param_value || 0) - (b.param_value || 0));
    const colour = GRP_COL[g[0] && g[0].swept_param_group] ||
                   PLASMA[Math.floor(ki / keys.length * (PLASMA.length - 1))];
    return {
      type: 'scatter3d', mode: 'lines+markers',
      x: g.map(d => d[xCol]), y: g.map(d => d[yCol]), z: g.map(d => d[zCol]),
      line: {color: colour, width: 3},
      marker: {
        color: cVals.length ? g.map(d => d[cCol]) : colour,
        colorscale: 'Plasma', cmin: cLo, cmax: cHi, size: 4, showscale: false,
      },
      text: g.map(d =>
        k + ': ' + d.param_value_str + '<br>' +
        lbl(xCol) + ': ' + fmt(d[xCol]) + '<br>' +
        lbl(yCol) + ': ' + fmt(d[yCol]) + '<br>' +
        lbl(zCol) + ': ' + fmt(d[zCol]) + '<br>' +
        lbl(cCol) + ': ' + fmt(d[cCol])
      ),
      hovertemplate: '%{text}<extra>' + k + '</extra>',
      name: k,
    };
  });

  Plotly.react('plot3d', traces, layout, {responsive: true});
  document.getElementById('info3d').textContent =
    'Scatter lines — one trace per sweep group. ' + rows.length + ' data points.';
}

initSelects();
draw2d();

// ── Right-click "Hide point" context menu ────────────────────────────────────
(function() {
  const menu = document.createElement('div');
  menu.id = 'ctx-menu';
  menu.innerHTML =
    '<div class="ctx-item" id="ctx-hide">Hide this point</div>' +
    '<div class="ctx-divider"></div>' +
    '<div class="ctx-item" id="ctx-reset">Reset all hidden</div>';
  document.body.appendChild(menu);

  function closeMenu() { menu.style.display = 'none'; }
  function is3dActive() { return document.getElementById('tab3d').classList.contains('active'); }
  function redrawAll() { draw2d(); if (is3dActive()) draw3d(); }

  document.addEventListener('click', closeMenu);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });

  const plot2dEl = document.getElementById('plot2d');

  // Track hover with plotly events
  plot2dEl.on('plotly_hover', data => {
    const pt = data.points[0];
    const row = (traceRows[pt.curveNumber] || [])[pt.pointNumber];
    hoveredPoint = (row !== null && row !== undefined) ? row : null;
  });
  plot2dEl.on('plotly_unhover', () => { hoveredPoint = null; });

  // KEY FIX: capture right-mousedown BEFORE Plotly can fire plotly_unhover.
  // The capture phase runs before bubble phase, so we snapshot hoveredPoint
  // here while it is still guaranteed to be set from plotly_hover.
  let savedForMenu = null, menuX = 0, menuY = 0;

  plot2dEl.addEventListener('mousedown', e => {
    if (e.button !== 2) return;
    savedForMenu = hoveredPoint;          // snapshot before any unhover
    menuX = Math.min(e.clientX, window.innerWidth  - 200);
    menuY = Math.min(e.clientY, window.innerHeight - 80);
  }, { capture: true });

  plot2dEl.addEventListener('contextmenu', e => {
    e.preventDefault();
    e.stopPropagation();
    if (savedForMenu === null) return;
    menu.style.left = menuX + 'px';
    menu.style.top  = menuY + 'px';
    menu.style.display = 'block';
  }, { capture: true });

  document.getElementById('ctx-hide').addEventListener('click', () => {
    if (savedForMenu !== null) {
      hiddenPoints.add(savedForMenu);
      savedForMenu = null;
      hoveredPoint = null;
      closeMenu();
      redrawAll();
    }
  });

  document.getElementById('ctx-reset').addEventListener('click', () => {
    hiddenPoints.clear();
    savedForMenu = null;
    hoveredPoint = null;
    closeMenu();
    redrawAll();
  });
})();
</script>
</body>
</html>"""

    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report written to {HTML}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="OAT sensitivity analysis for the Kalman filter tracker.")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip simulations; rebuild HTML from existing CSV")
    ap.add_argument("--quick", type=int, default=None, metavar="N",
                    help="Use only N sweep points per param (smoke test)")
    ap.add_argument("--model", choices=["cv", "ca"], default="cv",
                    help="State model for the baseline: cv (default) or ca")
    ap.add_argument("--sweep", default=None, metavar="ID",
                    help="Run only the named sweep ID (e.g. P32) plus baseline")
    args = ap.parse_args()

    baseline = CA_BASELINE if args.model == "ca" else BASELINE
    # Use model-specific output paths so CV and CA results don't overwrite each other.
    suffix = f"_{args.model}" if args.model != "cv" else ""
    csv_path  = os.path.join(REPO, f"oat_results{suffix}.csv")
    html_path = os.path.join(REPO, f"oat_report{suffix}.html")

    # Patch module-level HTML path so build_html writes to the right file.
    global HTML
    HTML = html_path

    if args.plot_only:
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found. Run without --plot-only first.")
            sys.exit(1)
        df = pd.read_csv(csv_path)
    else:
        df = main_run(quick=args.quick, baseline=baseline,
                      sweep_filter=args.sweep, csv_path=csv_path)

    build_html(df)

if __name__ == "__main__":
    main()
