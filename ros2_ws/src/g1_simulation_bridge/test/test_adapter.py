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
        self.card_arm = None

    def set_arm(self, action_id, event):
        self.arm = (action_id, event)

    def set_hand(self, hand, positions, event, ramp=0.2):
        self.hand = (hand, positions, event, ramp)

    def set_arm_motion(self, motion):
        self.card_arm = motion


def adapter():
    instance = G1SimulationAdapter.__new__(G1SimulationAdapter)
    instance.controls = Controls()
    instance.neutral_targets = {
        "right": [0.399, 0.399, 0.05, 0.05, 0.049, 0.05],
        "left": [0.4, 0.4, 0.049, 0.05, 0.049, 0.05],
    }
    return instance


def test_resolves_explicit_simulation_root():
    expected = Path(__file__).resolve().parents[4] / "simulation"
    assert resolve_simulation_root(expected) == expected


def test_dispatches_reviewed_commands():
    instance = adapter()
    arm_result = instance._arm_execute({
        "action_id": 27, "client_id": "test-client"})
    target = [0.48, 0.45, 0.15, 0.20, 0.25, 0.20]
    result = instance._hand_command({
        "client_id": "test-client", "hand": "right",
        "target": target, "ramp_s": 0.2})
    assert instance.controls.arm == (27, "ros:arm:27")
    assert arm_result == {"code": 0}
    assert instance.controls.hand == (
        "right", target, "ros:right_hand:command", 0.2)
    assert result is None


@pytest.mark.parametrize("action_id", [0, 28, 100])
def test_rejects_unknown_arm_actions(action_id):
    with pytest.raises(ValueError):
        adapter()._arm_execute({
            "action_id": action_id, "client_id": "test-client"})


def test_release_operations_return_safe_targets():
    instance = adapter()
    instance._arm_release({})
    instance._hand_neutral({"hand": "right", "ramp_s": 0.2})
    assert instance.controls.arm == (99, "ros:arm:99")
    assert instance.controls.hand == (
        "right", instance.neutral_targets["right"],
        "ros:right_hand:neutral", 0.2)


def test_restricted_motion_accepts_explicit_1_2_second_phases():
    from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES
    motion = {
        'version': 1, 'preparation': dict.fromkeys(ARM_JOINT_NAMES, 0.0),
        'target_offsets': dict.fromkeys(ARM_JOINT_NAMES[:7], 0.0),
        'phase_seconds': [0.3, 0.9], 'recovery': ['preparation', 'baseline'],
    }
    instance = adapter()
    result = instance._arm_motion({
        'motion': motion, 'execution_timeout_s': 8.0})
    assert result == {'mode': 'simulation', 'trajectory_duration': 1.2}
    assert instance.controls.card_arm == motion
