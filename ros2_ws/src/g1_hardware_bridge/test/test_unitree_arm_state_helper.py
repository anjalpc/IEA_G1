import json

import pytest

from g1_hardware_bridge.arm.state_helper import (
    OUTPUT_PREFIX,
    normalize_state,
)


def test_normalizes_observed_public_action():
    assert normalize_state(
        '{"holding": true, "id": 27, "name": "shake_hand"}'
    ) == {"id": 27, "name": "shake_hand", "holding": True}


def test_normalizes_observed_controller_action():
    assert normalize_state(
        '{"holding": false, "id": 552, "name": "vq_shake_hand"}'
    ) == {"id": 552, "name": "vq_shake_hand", "holding": False}


def test_optional_fields_have_conservative_defaults():
    assert normalize_state('{"id": 0}') == {
        "id": 0,
        "name": "",
        "holding": False,
    }


@pytest.mark.parametrize("payload", [
    "not-json",
    "[]",
    "{}",
    '{"id": "invalid"}',
])
def test_rejects_invalid_payload(payload):
    with pytest.raises((KeyError, TypeError, ValueError, json.JSONDecodeError)):
        normalize_state(payload)


def test_output_prefix_is_stable_for_parent_parser():
    assert OUTPUT_PREFIX == "ARM_STATE="
