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
    return instance


def test_resolves_explicit_simulation_root():
    expected = Path(__file__).resolve().parents[4] / "simulation"
    assert resolve_simulation_root(expected) == expected


def test_dispatches_reviewed_commands():
    instance = adapter()
    instance._arm_action({"action_id": 27})
    instance._hand_position({"hand": "right", "positions": [0.1] * 6})
    assert instance.controls.arm == (27, "ros:arm:27")
    assert instance.controls.hand == ([0.1] * 6, "ros:right_hand")


@pytest.mark.parametrize("action_id", [0, 28, 100])
def test_rejects_unknown_arm_actions(action_id):
    with pytest.raises(ValueError):
        adapter()._arm_action({"action_id": action_id})


def test_rejects_invalid_hand_commands():
    with pytest.raises(ValueError):
        adapter()._hand_position({"hand": "left", "positions": [0.0] * 6})
    with pytest.raises(ValueError):
        adapter()._hand_position({"hand": "right", "positions": [0.0] * 5})
    with pytest.raises(ValueError):
        adapter()._hand_position({"hand": "right", "positions": [1.1] * 6})
