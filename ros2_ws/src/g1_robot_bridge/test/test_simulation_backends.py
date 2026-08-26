from g1_robot_bridge.simulation_client import (
    SimulationArmActionBackend,
    SimulationBrainCoBackend,
)


class TransportRecorder:
    def __init__(self):
        self.commands = []

    def send(self, command, **payload):
        self.commands.append((command, payload))


def test_simulation_arm_uses_actions_27_and_99(monkeypatch):
    monkeypatch.setattr("g1_robot_bridge.simulation_client.time.sleep", lambda _: None)
    transport = TransportRecorder()
    backend = SimulationArmActionBackend(transport)
    assert backend.execute(27) == 0
    assert backend.execute(99) == 0
    assert transport.commands == [
        ("arm_action", {"action_id": 27}),
        ("arm_action", {"action_id": 99}),
    ]


def test_simulation_hand_uses_hardware_curl_plan(monkeypatch):
    monkeypatch.setattr("g1_robot_bridge.simulation_client.time.sleep", lambda _: None)
    transport = TransportRecorder()
    backend = SimulationBrainCoBackend(transport, 0.12)
    backend.grip()
    backend.release()
    assert transport.commands == [
        ("hand_position", {
            "hand": "right",
            "positions": [0.048, 0.03, 0.12, 0.12, 0.12, 0.12],
        }),
        ("hand_position", {"hand": "right", "positions": [0.0] * 6}),
    ]
