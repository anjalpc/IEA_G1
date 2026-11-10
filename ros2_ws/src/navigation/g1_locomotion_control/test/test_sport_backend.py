"""Tests for the hardware-free Sport backend boundary."""

import pytest

from g1_locomotion_control.sport_backend import (
    SportBackend,
    SportBackendError,
    decode_sport_request,
    encode_sport_health_request,
    encode_sport_move_request,
    encode_sport_stop_request,
)


def test_sport_move_protocol_round_trip():
    line = encode_sport_move_request(0.12, -0.03, 0.25)

    assert decode_sport_request(line) == {
        "action": "move",
        "forward_mps": 0.12,
        "lateral_mps": -0.03,
        "yaw_rate_rps": 0.25,
    }


def test_sport_stop_protocol_round_trip():
    line = encode_sport_stop_request()

    assert decode_sport_request(line) == {
        "action": "stop",
    }


def test_sport_health_protocol_round_trip():
    line = encode_sport_health_request()

    assert decode_sport_request(line) == {
        "action": "health",
    }


class FakeSportClient:
    """Record neutral Sport-port calls without SDK or DDS."""

    def __init__(self):
        self.moves = []
        self.stop_count = 0
        self.is_healthy = True
        self.move_error = None
        self.stop_error = None
        self.health_error = None

    def move(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        if self.move_error is not None:
            raise self.move_error
        self.moves.append(
            (forward_mps, lateral_mps, yaw_rate_rps)
        )

    def stop_move(self):
        self.stop_count += 1
        if self.stop_error is not None:
            raise self.stop_error

    def healthy(self):
        if self.health_error is not None:
            raise self.health_error
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


@pytest.mark.parametrize(
    "command",
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, -float("inf")),
    ],
)
def test_nonfinite_command_never_reaches_client(command):
    client = FakeSportClient()
    backend = SportBackend(client)
    backend.activate()

    with pytest.raises(
        SportBackendError,
        match="SPORT_MOVE_FAILED",
    ):
        backend.command_velocity(*command)

    assert client.moves == []


def test_move_failure_is_sanitized_and_marks_backend_faulted():
    client = FakeSportClient()
    original = RuntimeError(
        "secret-interface velocity=0.2"
    )
    client.move_error = original
    backend = SportBackend(client)
    backend.activate()

    with pytest.raises(SportBackendError) as caught:
        backend.command_velocity(0.2, -0.1, 0.3)

    assert str(caught.value) == "SPORT_MOVE_FAILED"
    assert caught.value.__cause__ is original
    assert backend.faulted is True
    assert backend.stopped is False
    assert backend.healthy() is False
    assert "0.2" not in str(caught.value)
    assert "secret-interface" not in str(caught.value)


def test_faulted_backend_rejects_commands_without_client_call():
    client = FakeSportClient()
    client.move_error = RuntimeError("first failure")
    backend = SportBackend(client)
    backend.activate()

    with pytest.raises(SportBackendError):
        backend.command_velocity(0.1, 0.0, 0.0)

    client.move_error = None
    with pytest.raises(
        SportBackendError,
        match="SPORT_CLIENT_UNAVAILABLE",
    ):
        backend.command_velocity(0.1, 0.0, 0.0)

    assert client.moves == []


def test_stop_failure_does_not_claim_stopped_state():
    client = FakeSportClient()
    client.stop_error = RuntimeError("private DDS detail")
    backend = SportBackend(client)
    backend.activate()
    backend.command_velocity(0.2, 0.0, 0.0)

    with pytest.raises(SportBackendError) as caught:
        backend.stop()

    assert str(caught.value) == "SPORT_STOP_FAILED"
    assert caught.value.__cause__ is client.stop_error
    assert backend.faulted is True
    assert backend.stopped is False
    assert backend.healthy() is False
    assert "private DDS detail" not in str(caught.value)


def test_health_exception_blocks_activation_without_leaking_detail():
    client = FakeSportClient()
    original = RuntimeError("private network identifier")
    client.health_error = original
    backend = SportBackend(client)

    with pytest.raises(SportBackendError) as caught:
        backend.activate()

    assert str(caught.value) == "SPORT_CLIENT_UNAVAILABLE"
    assert caught.value.__cause__ is original
    assert backend.active is False
    assert "private network identifier" not in str(caught.value)


def test_healthy_reactivation_clears_fault_before_safe_stop():
    client = FakeSportClient()
    client.move_error = RuntimeError("temporary failure")
    backend = SportBackend(client)
    backend.activate()

    with pytest.raises(SportBackendError):
        backend.command_velocity(0.2, 0.0, 0.0)

    client.move_error = None
    client.is_healthy = True
    backend.activate()

    assert backend.active is True
    assert backend.faulted is False
    assert backend.stopped is False
    assert backend.healthy() is True

    backend.stop()
    backend.deactivate()
    assert backend.active is False
