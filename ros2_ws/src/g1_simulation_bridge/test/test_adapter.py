from pathlib import Path

import pytest

from g1_simulation_bridge.adapter_node import (
    G1SimulationAdapter,
    resolve_simulation_root,
)


class Controls:
    def __init__(self):
        self.arm = None
        self.hand = None

    def set_arm(self, action_id, event):
        self.arm = (action_id, event)

    def set_hand(self, positions, event):
        self.hand = (positions, event)


def adapter():
    instance = G1SimulationAdapter.__new__(G1SimulationAdapter)
    instance.controls = Controls()
    instance.grip_target = [0.48, 0.45, 0.15, 0.20, 0.25, 0.20]
    instance.neutral_target = [0.399, 0.399, 0.05, 0.05, 0.049, 0.05]
    return instance


def test_resolves_explicit_simulation_root():
    expected = Path(__file__).resolve().parents[4] / "simulation"
    assert resolve_simulation_root(expected) == expected


def test_dispatches_reviewed_commands():
    instance = adapter()
    arm_result = instance._arm_action({
        "action_id": 27, "client_id": "test-client"})
    result = instance._hand_grip({"client_id": "test-client"})
    assert instance.controls.arm == (27, "ros:arm:27")
    assert arm_result == {"code": 0}
    assert instance.controls.hand == (
        instance.grip_target, "ros:right_hand:grip")
    assert result is None


@pytest.mark.parametrize("action_id", [0, 28, 100])
def test_rejects_unknown_arm_actions(action_id):
    with pytest.raises(ValueError):
        adapter()._arm_action({
            "action_id": action_id, "client_id": "test-client"})


def test_release_operations_return_safe_targets():
    instance = adapter()
    instance._arm_release({})
    instance._hand_release({})
    assert instance.controls.arm == (99, "ros:arm:99")
    assert instance.controls.hand == (
        instance.neutral_target, "ros:right_hand:release")
