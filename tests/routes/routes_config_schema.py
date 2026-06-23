"""
Route coverage for config_schema.py.
Tests all three public outcomes of validate_config:
  - valid config passes without exception
  - empty dict raises ValidationError (missing required top-level keys)
  - invalid field value (steps=-1) raises ValidationError (minimum: 1)
"""
import pytest
import jsonschema

from config_schema import validate_config
from tests.conftest import load_input_json


class RoutesConfigSchema:

    def route_validate_config_valid(self):
        """Valid input.json passes schema validation without raising."""
        cfg = load_input_json()
        validate_config(cfg)  # must not raise

    def route_validate_config_missing_required_field(self):
        """Empty dict missing all required top-level keys raises ValidationError."""
        with pytest.raises(jsonschema.ValidationError):
            validate_config({})

    def route_validate_config_invalid_steps_type(self):
        """steps value below minimum (1) raises ValidationError."""
        cfg = load_input_json()
        cfg["simulation_parameters"]["steps"] = -1
        with pytest.raises(jsonschema.ValidationError):
            validate_config(cfg)
