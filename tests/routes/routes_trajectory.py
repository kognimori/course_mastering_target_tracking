"""
Route coverage for trajectory.py.
Covers FreeTraj, STraj, FreeTurnTraj, and Trajectory factory for all motion routes:
- FreeTraj: initialise + multi-step update + write
- STraj: initialise + update without n-switch + update WITH n-switch + write
- FreeTurnTraj: initialise + constant yawrate + circular motion + write
- Trajectory factory: FreeTraj dispatch, STraj dispatch, FreeTurnTraj dispatch
"""
import os
import math
import tempfile

import numpy as np
import pytest

from tests.conftest import OBJECT_MOTION_TEMPLATE, SRC_DIR, REPO_ROOT

# Modules imported after conftest adds SRC_DIR to sys.path
from trajectory import FreeTraj, STraj, FreeTurnTraj, Trajectory


STEPS = 20
DT = 1.0

FREE_PARS = {}
STRAJ_PARS = {"turnRate": 1.0}
FREE_TURN_PARS = {"turnRate": 30.0}

INITIAL_STATES = {
    "range": 5.0, "azimuth": 45.0, "speed": 1.0,
    "acceleration": 0.0, "yaw": 30.0, "yawrate": 0.0, "yawraterate": 0.0,
}


# =========================================================================
# FreeTraj routes
# =========================================================================

class RoutesFreeTraj:

    def route_freetraj_init(self):
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        for key in OBJECT_MOTION_TEMPLATE:
            assert key in ft.states
            assert ft.states[key].shape == (STEPS, 1)

    def route_freetraj_initialise(self):
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        assert math.isclose(ft.states["speed"][0, 0], 1.0)
        assert ft.states["x"][0, 0] != 0.0  # range * cos(yaw) != 0
        assert ft.states["y"][0, 0] != 0.0

    def route_freetraj_update_single_step(self):
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        ft.update(1, DT)
        # Position advances by velocity * dt
        assert ft.states["timestamps"][1, 0] == pytest.approx(DT)
        assert ft.states["x"][1, 0] != 0.0

    def route_freetraj_update_multi_step(self):
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            ft.update(t, DT)
        # Speed stays constant (zero acceleration)
        assert ft.states["speed"][-1, 0] == pytest.approx(ft.states["speed"][0, 0])

    def route_freetraj_update_with_acceleration(self):
        init = dict(INITIAL_STATES)
        init["acceleration"] = 0.5
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(init)
        for t in range(1, STEPS):
            ft.update(t, DT)
        # Speed increases each step
        assert ft.states["speed"][-1, 0] > ft.states["speed"][0, 0]

    def route_freetraj_update_with_yawrate(self):
        init = dict(INITIAL_STATES)
        init["yawrate"] = 5.0  # degrees/s
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(init)
        for t in range(1, STEPS):
            ft.update(t, DT)
        # Yaw changes over time
        assert ft.states["yaw"][-1, 0] != ft.states["yaw"][0, 0]

    def route_freetraj_write(self, tmp_path):
        ft = FreeTraj(FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            ft.update(t, DT)
        fpath = str(tmp_path / "freetraj_test.ftr")
        ft.write(fpath)
        import pandas as pd
        df = pd.read_feather(fpath)
        assert "x" in df.columns
        assert len(df) == STEPS


# =========================================================================
# STraj routes
# =========================================================================

class RoutesSTraJ:

    def route_straj_init(self):
        st = STraj(STRAJ_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        assert st.n == 180
        assert st.f == 1
        for key in OBJECT_MOTION_TEMPLATE:
            assert key in st.states

    def route_straj_initialise(self):
        st = STraj(STRAJ_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        st.initialise(INITIAL_STATES)
        assert st.RAmp > 0
        assert st.states["speed"][0, 0] == pytest.approx(1.0)

    def route_straj_update_no_switch(self):
        """Steps that do not cross the n-threshold."""
        st = STraj(STRAJ_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        st.initialise(INITIAL_STATES)
        for t in range(1, 5):
            st.update(t, DT)
        # vx and vy should now differ after the bug fix
        assert st.states["vx"][4, 0] != st.states["vy"][4, 0] or (
            st.states["vx"][4, 0] == 0.0 and st.states["vy"][4, 0] == 0.0
        )

    def route_straj_vy_not_equal_vx(self):
        """Confirm CLM-1.1 fix: vy != vx in general."""
        st = STraj(STRAJ_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        init = dict(INITIAL_STATES)
        init["yaw"] = 10.0  # non-trivial angle so x and y differ
        st.initialise(init)
        for t in range(1, STEPS):
            st.update(t, DT)
        # After many steps, vx and vy should diverge
        vx_series = [st.states["vx"][t, 0] for t in range(1, STEPS)]
        vy_series = [st.states["vy"][t, 0] for t in range(1, STEPS)]
        # They won't be identical for all steps
        assert not all(math.isclose(vx, vy) for vx, vy in zip(vx_series, vy_series))

    def route_straj_update_with_switch(self):
        """Run enough steps that degrees(thetao) > self.n triggers the switch."""
        pars = {"turnRate": 180.0}  # Large turnRate → fast theta growth → quick switch
        st = STraj(pars, 400, OBJECT_MOTION_TEMPLATE)
        st.initialise(INITIAL_STATES)
        initial_n = st.n
        initial_f = st.f
        switched = False
        for t in range(1, 400):
            st.update(t, DT)
            if st.n != initial_n or st.f != initial_f:
                switched = True
                break
        assert switched, "n-switch branch was never taken"

    def route_straj_write(self, tmp_path):
        st = STraj(STRAJ_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        st.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            st.update(t, DT)
        fpath = str(tmp_path / "straj_test.ftr")
        st.write(fpath)
        import pandas as pd
        df = pd.read_feather(fpath)
        assert "vy" in df.columns
        assert len(df) == STEPS


# =========================================================================
# FreeTurnTraj routes
# =========================================================================

class RoutesFreeTurnTraj:

    def route_freeturnrtaj_init(self):
        ft = FreeTurnTraj(FREE_TURN_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        import math
        for key in OBJECT_MOTION_TEMPLATE:
            assert key in ft.states
            assert ft.states[key].shape == (STEPS, 1)
        expected_yawrate = math.radians(FREE_TURN_PARS["turnRate"])
        assert ft._yawrate == pytest.approx(expected_yawrate)

    def route_freeturnrtaj_initialise(self):
        ft = FreeTurnTraj(FREE_TURN_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        import math
        expected_yawrate = math.radians(FREE_TURN_PARS["turnRate"])
        # yawrate at t=0 must be the fixed turn rate, not initial["yawrate"]=0
        assert ft.states["yawrate"][0, 0] == pytest.approx(expected_yawrate)
        assert ft.states["yawraterate"][0, 0] == pytest.approx(0.0)
        assert ft.states["speed"][0, 0] == pytest.approx(INITIAL_STATES["speed"])

    def route_freeturnrtaj_constant_yawrate(self):
        """Yaw rate must stay fixed at turnRate for every step."""
        ft = FreeTurnTraj(FREE_TURN_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            ft.update(t, DT)
        import math
        expected = math.radians(FREE_TURN_PARS["turnRate"])
        for t in range(STEPS):
            assert ft.states["yawrate"][t, 0] == pytest.approx(expected)
        # yawraterate must be zero at all steps >= 1
        for t in range(1, STEPS):
            assert ft.states["yawraterate"][t, 0] == pytest.approx(0.0)

    def route_freeturnrtaj_circular_motion(self):
        """Non-zero turn rate causes heading to change monotonically."""
        ft = FreeTurnTraj(FREE_TURN_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            ft.update(t, DT)
        # Heading at final step must differ from heading at step 0
        assert ft.states["yaw"][-1, 0] != pytest.approx(ft.states["yaw"][0, 0])
        # Timestamps advance monotonically
        for t in range(1, STEPS):
            assert ft.states["timestamps"][t, 0] > ft.states["timestamps"][t - 1, 0]

    def route_freeturnrtaj_write(self, tmp_path):
        ft = FreeTurnTraj(FREE_TURN_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        ft.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            ft.update(t, DT)
        fpath = str(tmp_path / "freeturnrtaj_test.ftr")
        ft.write(fpath)
        import pandas as pd
        df = pd.read_feather(fpath)
        assert "x" in df.columns
        assert "yawrate" in df.columns
        assert len(df) == STEPS


# =========================================================================
# Trajectory factory routes
# =========================================================================

class RoutesTrajectoryFactory:

    def route_trajectory_freetraj(self):
        traj = Trajectory("FreeTraj", FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        assert isinstance(traj.entity, FreeTraj)

    def route_trajectory_freeturnrtaj(self):
        traj = Trajectory("FreeTurnTraj", FREE_TURN_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        assert isinstance(traj.entity, FreeTurnTraj)

    def route_trajectory_straj(self):
        traj = Trajectory("STraj", STRAJ_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        assert isinstance(traj.entity, STraj)

    def route_trajectory_initialise_delegates(self):
        traj = Trajectory("FreeTraj", FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        traj.initialise(INITIAL_STATES)
        assert traj.entity.states["speed"][0, 0] == pytest.approx(1.0)

    def route_trajectory_update_delegates(self):
        traj = Trajectory("FreeTraj", FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        traj.initialise(INITIAL_STATES)
        traj.update(1, DT)
        assert traj.entity.states["timestamps"][1, 0] == pytest.approx(DT)

    def route_trajectory_write_states(self, tmp_path):
        traj = Trajectory("FreeTraj", FREE_PARS, STEPS, OBJECT_MOTION_TEMPLATE)
        traj.initialise(INITIAL_STATES)
        for t in range(1, STEPS):
            traj.update(t, DT)
        fpath = str(tmp_path / "traj_write_test.ftr")
        traj.write_states(fpath)
        import pandas as pd
        df = pd.read_feather(fpath)
        assert len(df) == STEPS

    def route_trajectory_unknown_type_raises(self):
        """Dispatch guard raises ValueError for unrecognised type strings."""
        with pytest.raises(ValueError, match="Unknown trajectory type"):
            Trajectory("BadType", {}, STEPS, OBJECT_MOTION_TEMPLATE)
