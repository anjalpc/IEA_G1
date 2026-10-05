"""Tests for the hardware-free Sport backend boundary."""

import pytest

from g1_locomotion_control.sport_backend import (
    SportBackend,
    SportBackendError,
)


class FakeSportClient:
    """Record neutral Sport-port calls without SDK or DDS."""

    def __init__(self):
        self.moves = []
        self.stop_count = 0
        self.is_healthy = True

    def move(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        self.moves.append(
            (forward_mps, lateral_mps, yaw_rate_rps)
        )

    def stop_move(self):
        self.stop_count += 1

    def healthy(self):
        return self.is_healthy


def test_activation_checks_health_without_commanding_motion():
    client = FakeSportClient()
    backend = SportBackend(client)

    backend.activate()

    assert backend.active is True
    assert backend.stopped is True
    assert client.moves == []
    assert client.stop_count == 0


def test_unhealthy_client_blocks_activation():
    client = FakeSportClient()
    client.is_healthy = False
    backend = SportBackend(client)

    with pytest.raises(
        SportBackendError,
        match="SPORT_CLIENT_UNAVAILABLE",
    ):
        backend.activate()

    assert backend.active is False
    assert client.moves == []
    assert client.stop_count == 0


def test_repeated_activation_is_idempotent():
    client = FakeSportClient()
    backend = SportBackend(client)

    backend.activate()
    backend.activate()

    assert backend.active is True
    assert client.moves == []
    assert client.stop_count == 0


def test_command_before_activation_is_rejected():
    client = FakeSportClient()
    backend = SportBackend(client)

    with pytest.raises(
        SportBackendError,
        match="SPORT_BACKEND_INACTIVE",
    ):
        backend.command_velocity(0.1, 0.0, 0.0)

    assert client.moves == []


@pytest.mark.parametrize(
    "command",
    [
        (0.2, -0.1, 0.3),
        (-0.4, 0.25, -0.35),
        (0.0, 0.0, 0.0),
    ],
)
def test_forwards_planar_velocity_exactly(command):
    client = FakeSportClient()
    backend = SportBackend(client)
    backend.activate()

    backend.command_velocity(*command)

    assert client.moves == [command]
    assert backend.stopped is (
        command == (0.0, 0.0, 0.0)
    )


def test_repeated_stop_requests_remain_safe():
    client = FakeSportClient()
    backend = SportBackend(client)
    backend.activate()
    backend.command_velocity(0.2, 0.0, 0.0)

    backend.stop()
    backend.stop()

    assert client.stop_count == 2
    assert backend.stopped is True


def test_deactivation_requires_stopped_state():
    client = FakeSportClient()
    backend = SportBackend(client)
    backend.activate()
    backend.command_velocity(0.2, 0.0, 0.0)

    with pytest.raises(
        SportBackendError,
        match="SPORT_BACKEND_NOT_STOPPED",
    ):
        backend.deactivate()

    assert backend.active is True


def test_deactivation_blocks_future_commands():
    client = FakeSportClient()
    backend = SportBackend(client)
    backend.activate()
    backend.stop()

    backend.deactivate()

    assert backend.active is False
    with pytest.raises(
        SportBackendError,
        match="SPORT_BACKEND_INACTIVE",
    ):
        backend.command_velocity(0.1, 0.0, 0.0)
