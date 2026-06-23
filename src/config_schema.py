"""JSON schema and validation helper for ``input/input.json``.

Provides :func:`validate_config` which raises
:class:`jsonschema.ValidationError` when a configuration dict does not
satisfy the required structure.
"""
import logging

import jsonschema

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_INPUT_SCHEMA: dict = {
    "type": "object",
    "required": [
        "simulation_parameters",
        "scenario_parameters",
        "initial_states",
        "trajectory_parameters",
    ],
    "properties": {
        "simulation_parameters": {
            "type": "object",
            "required": ["steps", "time_step"],
            "properties": {
                "steps": {"type": "integer", "minimum": 1},
                "time_step": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "scenario_parameters": {
            "type": "object",
            "required": [
                "num_states",
                "states",
                "measurements",
                "stateTypes",
                "measurementsTypes",
                "meas_noise_actual",
                "meas_noise_guess",
                "process_noise",
                "initial_state_guess",
                "initial_covariance_guess",
            ],
            "properties": {
                "num_states": {"type": "integer", "minimum": 1},
                "states": {"type": "string"},
                "measurements": {"type": "string"},
                "stateTypes": {"type": "object"},
                "measurementsTypes": {"type": "object"},
                "meas_noise_actual": {
                    "type": "object",
                    "required": ["range", "azimuth", "doppler"],
                    "properties": {
                        "range": {"type": "number", "minimum": 0},
                        "azimuth": {"type": "number", "minimum": 0},
                        "doppler": {"type": "number", "minimum": 0},
                    },
                },
                "meas_noise_guess": {
                    "type": "object",
                    "required": ["range", "azimuth", "doppler", "rangex", "rangey"],
                },
                "process_noise": {"type": "number", "minimum": 0},
                "initial_state_guess": {"type": "object"},
                "initial_covariance_guess": {"type": "object"},
            },
        },
        "initial_states": {
            "type": "object",
            "required": ["sensor", "target"],
            "properties": {
                "sensor": {
                    "type": "object",
                    "required": [
                        "range", "azimuth", "speed",
                        "acceleration", "yaw", "yawrate", "yawraterate",
                    ],
                },
                "target": {
                    "type": "object",
                    "required": [
                        "range", "azimuth", "speed",
                        "acceleration", "yaw", "yawrate", "yawraterate",
                    ],
                },
            },
        },
        "trajectory_parameters": {
            "type": "object",
            "required": ["selection"],
            "properties": {
                "selection": {
                    "type": "object",
                    "required": ["sensor", "target"],
                },
            },
        },
    },
}


def validate_config(cfg: dict) -> None:
    """Validate *cfg* against the input.json schema.

    @param cfg: Parsed configuration dictionary to validate.
    @type cfg: dict
    @raises jsonschema.ValidationError: If the configuration is invalid.
    @raises jsonschema.SchemaError: If the internal schema itself is malformed.
    """
    jsonschema.validate(instance=cfg, schema=_INPUT_SCHEMA)
    logger.debug("Config validation passed")
