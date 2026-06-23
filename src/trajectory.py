"""Kinematic trajectory generators for ego and target entities.

Provides :class:`FreeTraj` (free-yaw motion), :class:`STraj`
(sinusoidal turning motion), and :class:`FreeTurnTraj`
(constant-turn-rate motion), wrapped by the :class:`Trajectory` factory.
"""
import logging
from copy import deepcopy
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy import sin, cos, radians, degrees
from numpy import zeros, set_printoptions

set_printoptions(suppress=True, precision=4)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class EntityStates(TypedDict):
    """Shape contract for the per-entity state dict (GAP-Q5).

    All values are ``np.ndarray`` of shape ``(steps, 1)`` during the
    simulation and flat Python lists after :meth:`FreeTraj.write` is called.
    """

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    yawrate: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    yawraterate: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    timestamps: np.ndarray


class FreeTraj:
    """Free-yaw kinematic trajectory model.

    Propagates position and velocity using constant yaw-rate and acceleration
    kinematics.  All angular quantities are stored internally in radians.
    """

    def __init__(self, pars: dict[str, Any], steps: int, objectMotionTemplate: dict[str, Any]) -> None:
        """Allocate state arrays for the requested number of time steps.

        @param pars: Trajectory parameter dict (unused for FreeTraj; reserved for
            extensibility).
        @type pars: dict[str, Any]
        @param steps: Total number of time steps in the simulation.
        @type steps: int
        @param objectMotionTemplate: Template dict mapping state-name to a placeholder
            value; each key becomes a (steps, 1) zero-filled array.
        @type objectMotionTemplate: dict[str, Any]
        """
        self.states = deepcopy(objectMotionTemplate)
        for key in self.states.keys():
            self.states[key] = zeros([steps, 1], dtype=float)

        self.pars = pars

    def initialise(self, initial: dict[str, float]) -> None:
        """Populate time-0 states from an initial-condition dict.

        @param initial: Mapping of state name to scalar value.  Required keys:
            ``range``, ``azimuth``, ``yaw``, ``speed``, ``yawrate``,
            ``acceleration``, ``yawraterate``.  Angles supplied in degrees.
        @type initial: dict[str, float]
        """
        self.states["x"][0, 0] = initial["range"] * cos(radians(initial["azimuth"]))
        self.states["y"][0, 0] = initial["range"] * sin(radians(initial["azimuth"]))
        self.states["yaw"][0, 0] = radians(initial["yaw"])
        self.states["speed"][0, 0] = initial["speed"]
        self.states["vx"][0, 0] = initial["speed"] * cos(radians(initial["yaw"]))
        self.states["vy"][0, 0] = initial["speed"] * sin(radians(initial["yaw"]))
        self.states["yawrate"][0, 0] = radians(initial["yawrate"])
        self.states["acceleration"][0, 0] = initial["acceleration"]
        self.states["ax"][0, 0] = initial["acceleration"] * cos(radians(initial["yaw"]))
        self.states["ay"][0, 0] = initial["acceleration"] * sin(radians(initial["yaw"]))
        self.states["yawraterate"][0, 0] = radians(initial["yawraterate"])

    def update(self, t: int, dt: float) -> None:
        """Propagate kinematics from step t-1 to step t.

        @param t: Current time index (must be >= 1).
        @type t: int
        @param dt: Time-step duration in seconds.
        @type dt: float
        """
        self.states["timestamps"][t, 0] = self.states["timestamps"][t - 1, 0] + dt
        self.states["acceleration"][t, 0] = self.states["acceleration"][t - 1, 0]
        self.states["yawraterate"][t, 0] = self.states["yawraterate"][t - 1, 0]

        self.states["speed"][t, 0] = self.states["speed"][t - 1, 0] + self.states["acceleration"][t, 0] * dt
        self.states["yawrate"][t, 0] = self.states["yawrate"][t - 1, 0] + self.states["yawraterate"][t, 0] * dt

        self.states["yaw"][t, 0] = self.states["yaw"][t - 1, 0] + self.states["yawrate"][t, 0] * dt

        self.states["ax"][t, 0] = self.states["acceleration"][t, 0] * cos(self.states["yaw"][t, 0])
        self.states["ay"][t, 0] = self.states["acceleration"][t, 0] * sin(self.states["yaw"][t, 0])

        self.states["vx"][t, 0] = self.states["speed"][t, 0] * cos(self.states["yaw"][t, 0])
        self.states["vy"][t, 0] = self.states["speed"][t, 0] * sin(self.states["yaw"][t, 0])

        self.states["x"][t, 0] = self.states["x"][t - 1, 0] + self.states["vx"][t, 0] * dt
        self.states["y"][t, 0] = self.states["y"][t - 1, 0] + self.states["vy"][t, 0] * dt

    def write(self, filePath: str) -> None:
        """Flatten state arrays into a DataFrame and persist as Feather.

        Mutates ``self.states`` values from numpy arrays to flat Python lists
        and sets ``self.statesDF``.

        @param filePath: Destination path for the ``.ftr`` file.
        @type filePath: str
        """
        for key in self.states.keys():
            self.states[key] = list(self.states[key].flatten())

        self.statesDF = pd.DataFrame.from_dict(self.states)
        self.statesDF.to_feather(filePath)


class STraj:
    """Sinusoidal (S-curve) trajectory model.

    Implements a parametric sinusoidal path in the body frame, with a
    sign-flip and amplitude step when the cumulative phase angle exceeds
    ``self.n`` degrees (the *n-switch* manoeuvre).
    """

    def __init__(self, pars: dict[str, Any], steps: int, objectMotionTemplate: dict[str, Any]) -> None:
        """Allocate state arrays and initialise S-curve control variables.

        @param pars: Trajectory parameter dict.  Required key: ``turnRate``
            (degrees).
        @type pars: dict[str, Any]
        @param steps: Total number of time steps.
        @type steps: int
        @param objectMotionTemplate: Template dict mapping state-name to a placeholder.
        @type objectMotionTemplate: dict[str, Any]
        """
        self.states = deepcopy(objectMotionTemplate)
        for key in self.states.keys():
            self.states[key] = zeros([steps, 1], dtype=float)

        self.pars = pars
        self.n = 180
        self.a = 1
        self.f = 1

    def initialise(self, initial: dict[str, float]) -> None:
        """Populate time-0 states and derive S-curve amplitude.

        @param initial: Initial-condition dict; same keys as
            :meth:`FreeTraj.initialise`.  Angles in degrees.
        @type initial: dict[str, float]
        """
        self.states["yaw"][0, 0] = radians(initial["yaw"])
        self.states["speed"][0, 0] = initial["speed"]
        self.states["vx"][0, 0] = initial["speed"] * cos(radians(initial["yaw"]))
        self.states["vy"][0, 0] = initial["speed"] * sin(radians(initial["yaw"]))
        self.states["yawrate"][0, 0] = radians(initial["yawrate"])
        self.states["acceleration"][0, 0] = initial["acceleration"]
        self.states["ax"][0, 0] = initial["acceleration"] * cos(radians(initial["yaw"]))
        self.states["ay"][0, 0] = initial["acceleration"] * sin(radians(initial["yaw"]))
        self.states["yawraterate"][0, 0] = radians(initial["yawraterate"])

        omega = 1 / radians(self.pars["turnRate"])
        self.RAmp = self.states["speed"][0, 0] * omega

        phi = -self.states["yaw"][0, 0]
        x = self.a - self.f * cos((self.states["timestamps"][0, 0] + 0.01) / omega)
        y = sin((self.states["timestamps"][0, 0] + 0.01) / omega)

        self.states["x"][0, 0] = self.RAmp * (x * cos(phi) + y * sin(phi))
        self.states["y"][0, 0] = self.RAmp * (-x * sin(phi) + y * cos(phi))

        omega = 1 / radians(self.pars["turnRate"])

    def update(self, t: int, dt: float) -> None:
        """Propagate S-curve kinematics from step t-1 to step t.

        Applies the *n-switch* (increments ``self.n`` by 180, steps
        ``self.a`` by 2, flips ``self.f``) when the cumulative phase angle
        exceeds the current ``self.n`` threshold.

        @param t: Current time index (must be >= 1).
        @type t: int
        @param dt: Time-step duration in seconds.
        @type dt: float
        """
        self.states["timestamps"][t, 0] = self.states["timestamps"][t - 1, 0] + dt
        self.states["acceleration"][t, 0] = self.states["acceleration"][t - 1, 0]
        self.states["yawraterate"][t, 0] = self.states["yawraterate"][t - 1, 0]
        self.states["speed"][t, 0] = self.states["speed"][t - 1, 0] + self.states["acceleration"][t, 0] * dt
        self.states["yawrate"][t, 0] = self.states["yawrate"][t - 1, 0] + self.states["yawraterate"][t, 0] * dt
        self.states["yaw"][t, 0] = self.states["yaw"][t - 1, 0] + self.states["yawrate"][t, 0] * dt
        self.states["ax"][t, 0] = self.states["speed"][t, 0] * cos(self.states["yaw"][t, 0])
        self.states["ay"][t, 0] = self.states["speed"][t, 0] * sin(self.states["yaw"][t, 0])

        omega = 1 / radians(self.pars["turnRate"])

        phi = -self.states["yaw"][t, 0]
        x = self.a - self.f * cos(self.states["timestamps"][t, 0] / omega)
        y = sin(self.states["timestamps"][t, 0] / omega)

        self.states["x"][t, 0] = self.RAmp * (x * cos(phi) + y * sin(phi))
        self.states["y"][t, 0] = self.RAmp * (-x * sin(phi) + y * cos(phi))

        thetao = self.states["timestamps"][t, 0] / omega
        if degrees(thetao) > self.n:
            self.n += 180
            self.a += 2 * 1
            self.f = -self.f

        self.states["vx"][t, 0] = (self.states["x"][t, 0] - self.states["x"][t - 1, 0]) / dt
        self.states["vy"][t, 0] = (self.states["y"][t, 0] - self.states["y"][t - 1, 0]) / dt

    def write(self, filePath: str) -> None:
        """Flatten state arrays into a DataFrame and persist as Feather.

        Mutates ``self.states`` values from numpy arrays to flat Python lists
        and sets ``self.statesDF``.

        @param filePath: Destination path for the ``.ftr`` file.
        @type filePath: str
        """
        for key in self.states.keys():
            self.states[key] = list(self.states[key].flatten())

        self.statesDF = pd.DataFrame.from_dict(self.states)
        self.statesDF.to_feather(filePath)


class FreeTurnTraj:
    """Constant-turn-rate kinematic trajectory model.

    Like :class:`FreeTraj` but the yaw rate is fixed to
    ``radians(pars["turnRate"])`` at every time step, independent of the
    initial-state ``yawrate`` field.  Suitable for modelling coordinated
    constant-rate turning manoeuvres.
    """

    def __init__(self, pars: dict[str, Any], steps: int, objectMotionTemplate: dict[str, Any]) -> None:
        """Allocate state arrays and store the fixed turn rate.

        @param pars: Trajectory parameter dict. Required key: ``turnRate``
            (degrees per time-step; defaults to 0 if absent).
        @type pars: dict[str, Any]
        @param steps: Total number of time steps.
        @type steps: int
        @param objectMotionTemplate: Template dict mapping state-name to a placeholder.
        @type objectMotionTemplate: dict[str, Any]
        """
        self.states = deepcopy(objectMotionTemplate)
        for key in self.states.keys():
            self.states[key] = zeros([steps, 1], dtype=float)
        self.pars = pars
        self._yawrate: float = radians(pars.get("turnRate", 0.0))

    def initialise(self, initial: dict[str, float]) -> None:
        """Populate time-0 states from an initial-condition dict.

        The ``yawrate`` field in *initial* is overridden by
        ``pars["turnRate"]``; all other fields follow the same convention as
        :meth:`FreeTraj.initialise`.

        @param initial: Initial-condition dict; same keys as
            :meth:`FreeTraj.initialise`. Angles in degrees.
        @type initial: dict[str, float]
        """
        self.states["x"][0, 0] = initial["range"] * cos(radians(initial["azimuth"]))
        self.states["y"][0, 0] = initial["range"] * sin(radians(initial["azimuth"]))
        self.states["yaw"][0, 0] = radians(initial["yaw"])
        self.states["speed"][0, 0] = initial["speed"]
        self.states["vx"][0, 0] = initial["speed"] * cos(radians(initial["yaw"]))
        self.states["vy"][0, 0] = initial["speed"] * sin(radians(initial["yaw"]))
        self.states["yawrate"][0, 0] = self._yawrate
        self.states["acceleration"][0, 0] = initial["acceleration"]
        self.states["ax"][0, 0] = initial["acceleration"] * cos(radians(initial["yaw"]))
        self.states["ay"][0, 0] = initial["acceleration"] * sin(radians(initial["yaw"]))
        self.states["yawraterate"][0, 0] = 0.0

    def update(self, t: int, dt: float) -> None:
        """Propagate constant-turn-rate kinematics from step t-1 to step t.

        Yaw rate is held fixed at ``self._yawrate`` regardless of any
        ``yawraterate`` value; ``yawraterate`` is always zero.

        @param t: Current time index (must be >= 1).
        @type t: int
        @param dt: Time-step duration in seconds.
        @type dt: float
        """
        self.states["timestamps"][t, 0] = self.states["timestamps"][t - 1, 0] + dt
        self.states["acceleration"][t, 0] = self.states["acceleration"][t - 1, 0]
        self.states["yawraterate"][t, 0] = 0.0
        self.states["speed"][t, 0] = (
            self.states["speed"][t - 1, 0] + self.states["acceleration"][t, 0] * dt
        )
        self.states["yawrate"][t, 0] = self._yawrate
        self.states["yaw"][t, 0] = self.states["yaw"][t - 1, 0] + self._yawrate * dt
        self.states["ax"][t, 0] = self.states["acceleration"][t, 0] * cos(self.states["yaw"][t, 0])
        self.states["ay"][t, 0] = self.states["acceleration"][t, 0] * sin(self.states["yaw"][t, 0])
        self.states["vx"][t, 0] = self.states["speed"][t, 0] * cos(self.states["yaw"][t, 0])
        self.states["vy"][t, 0] = self.states["speed"][t, 0] * sin(self.states["yaw"][t, 0])
        self.states["x"][t, 0] = self.states["x"][t - 1, 0] + self.states["vx"][t, 0] * dt
        self.states["y"][t, 0] = self.states["y"][t - 1, 0] + self.states["vy"][t, 0] * dt

    def write(self, filePath: str) -> None:
        """Flatten state arrays into a DataFrame and persist as Feather.

        Mutates ``self.states`` values from numpy arrays to flat Python lists
        and sets ``self.statesDF``.

        @param filePath: Destination path for the ``.ftr`` file.
        @type filePath: str
        """
        for key in self.states.keys():
            self.states[key] = list(self.states[key].flatten())
        self.statesDF = pd.DataFrame.from_dict(self.states)
        self.statesDF.to_feather(filePath)


_TRAJECTORY_TYPES: dict[str, type] = {
    "FreeTraj": FreeTraj,
    "STraj": STraj,
    "FreeTurnTraj": FreeTurnTraj,
}
"""Registry mapping type-string names to trajectory classes."""


class Trajectory:
    """Factory wrapper that delegates to a concrete trajectory model.

    Selects :class:`FreeTraj`, :class:`STraj`, or :class:`FreeTurnTraj` by
    name and exposes a uniform interface (``initialise``, ``update``,
    ``write_states``).
    """

    def __init__(self, type_: str, pars: dict[str, Any], steps: int, objectMotionTemplate: dict[str, Any]) -> None:
        """Construct the concrete trajectory entity identified by *type_*.

        @param type_: Trajectory class name; must be a key in
            :data:`_TRAJECTORY_TYPES`.
        @type type_: str
        @param pars: Parameter dict forwarded to the concrete class.
        @type pars: dict[str, Any]
        @param steps: Total number of time steps.
        @type steps: int
        @param objectMotionTemplate: State-name template dict.
        @type objectMotionTemplate: dict[str, Any]
        @raises ValueError: If *type_* is not registered in
            :data:`_TRAJECTORY_TYPES`.
        """
        if type_ not in _TRAJECTORY_TYPES:
            raise ValueError(f"Unknown trajectory type: {type_!r}")
        self.entity = _TRAJECTORY_TYPES[type_](pars, steps, objectMotionTemplate)

    def initialise(self, initial: dict[str, float]) -> None:
        """Delegate initial-condition setup to the wrapped entity.

        @param initial: Initial-condition dict as required by the concrete class.
        @type initial: dict[str, float]
        """
        self.entity.initialise(initial)

    def update(self, t: int, dt: float) -> None:
        """Delegate a single propagation step to the wrapped entity.

        @param t: Current time index (must be >= 1).
        @type t: int
        @param dt: Time-step duration in seconds.
        @type dt: float
        """
        self.entity.update(t, dt)

    def write_states(self, filePath: str) -> None:
        """Persist the entity's state history to a Feather file.

        @param filePath: Destination path for the ``.ftr`` file.
        @type filePath: str
        """
        self.entity.write(filePath)
