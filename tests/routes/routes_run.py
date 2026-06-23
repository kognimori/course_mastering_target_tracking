"""
Route coverage for run.py.
Tests plot_trajectories_E, plot_trajectories_S, and _parse_args.
fig.show() is mocked to avoid opening a browser.
"""
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_sensor_df, make_target_df, make_estimates_xyz, make_estimates_range

from run import plot_trajectories_E, plot_trajectories_S, _parse_args


N = 5


class RoutesRun:

    def route_plot_trajectories_e(self, monkeypatch):
        """Tests the vx/vy/x/y trajectory plot (XY scatter)."""
        import plotly.graph_objs as go
        shown = []
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(True))

        allStatesT = make_target_df()
        allStatesE = make_sensor_df()
        estimates = make_estimates_xyz(N)

        plot_trajectories_E(allStatesT, allStatesE, estimates)
        assert len(shown) == 1

    def route_plot_trajectories_s(self, monkeypatch):
        """Tests the range-only trajectory plot."""
        import plotly.graph_objs as go
        shown = []
        monkeypatch.setattr(go.Figure, "show", lambda self: shown.append(True))

        allStatesT = make_target_df()
        allStatesE = make_sensor_df()
        estimates = make_estimates_range(N)

        plot_trajectories_S(allStatesT, allStatesE, estimates)
        assert len(shown) == 1

    def route_parse_args_default(self):
        """Default args: config=None, no_plot=False (GAP-O4)."""
        args = _parse_args([])
        assert args.config is None
        assert args.no_plot is False

    def route_parse_args_custom_config(self):
        """--config PATH sets args.config (GAP-O4)."""
        args = _parse_args(["--config", "myconfig.json"])
        assert args.config == "myconfig.json"

    def route_parse_args_no_plot(self):
        """--no-plot flag sets args.no_plot=True (GAP-O4)."""
        args = _parse_args(["--no-plot"])
        assert args.no_plot is True
