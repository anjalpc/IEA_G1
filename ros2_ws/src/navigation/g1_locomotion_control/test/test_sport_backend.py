"""Tests for the hardware-free Sport backend boundary."""

import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from g1_locomotion_control.sport_backend import (
    SportBackend,
    SportBackendError,
    SportProtocolClient,
    decode_sport_request,
    decode_sport_response,
    encode_sport_health_request,
    encode_sport_move_request,
    encode_sport_stop_request,
    encode_sport_response,
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


def test_sport_response_protocol_round_trip():
    line = encode_sport_response(
        action="stop",
        ok=True,
        stopped=True,
    )

    assert decode_sport_response(line) == {
        "action": "stop",
        "ok": True,
        "stopped": True,
    }


def test_successful_stop_response_must_confirm_stopped():
    with pytest.raises(
        ValueError,
        match="successful Sport STOP must confirm stopped",
    ):
        encode_sport_response(
            action="stop",
            ok=True,
            stopped=False,
        )


def test_protocol_client_health_uses_read_only_exchange():
    requests = []

    def exchange(line):
        requests.append(decode_sport_request(line))
        return encode_sport_response(
            action="health",
            ok=True,
            stopped=True,
        )

    client = SportProtocolClient(exchange)

    assert client.healthy() is True
    assert requests == [{"action": "health"}]


def test_protocol_client_move_requires_matching_acknowledgement():
    requests = []

    def exchange(line):
        requests.append(decode_sport_request(line))
        return encode_sport_response(
            action="move",
            ok=True,
            stopped=False,
        )

    client = SportProtocolClient(exchange)
    client.move(0.12, -0.03, 0.25)

    assert requests == [{
        "action": "move",
        "forward_mps": 0.12,
        "lateral_mps": -0.03,
        "yaw_rate_rps": 0.25,
    }]


def test_protocol_client_stop_requires_confirmed_stop():
    requests = []

    def exchange(line):
        requests.append(decode_sport_request(line))
        return encode_sport_response(
            action="stop",
            ok=True,
            stopped=True,
        )

    client = SportProtocolClient(exchange)
    client.stop_move()

    assert requests == [{"action": "stop"}]


def test_native_helper_maps_stop_only_to_zero_velocity():
    from g1_locomotion_control.sport_helper import (
        execute_sport_request,
    )

    client = Mock()
    client.SetVelocity.return_value = 0

    response = execute_sport_request(
        client,
        {"action": "stop"},
        duration_seconds=0.25,
    )

    assert client.SetVelocity.call_args.args == (
        0.0, 0.0, 0.0, 0.25,
    )
    assert decode_sport_response(response) == {
        "action": "stop",
        "ok": True,
        "stopped": True,
    }
    assert client.Damp.call_count == 0
    assert client.ZeroTorque.call_count == 0
    assert client.Sit.call_count == 0


def test_native_helper_maps_move_to_bounded_velocity_rpc():
    from g1_locomotion_control.sport_helper import (
        execute_sport_request,
    )

    client = Mock()
    client.SetVelocity.return_value = 0
    request = {
        "action": "move",
        "forward_mps": 0.12,
        "lateral_mps": -0.03,
        "yaw_rate_rps": 0.25,
    }

    response = execute_sport_request(
        client, request, duration_seconds=0.25,
    )

    assert client.SetVelocity.call_args.args == (
        0.12, -0.03, 0.25, 0.25,
    )
    assert decode_sport_response(response) == {
        "action": "move",
        "ok": True,
        "stopped": False,
    }
    assert client.Damp.call_count == 0
    assert client.ZeroTorque.call_count == 0
    assert client.Sit.call_count == 0


@pytest.mark.parametrize(
    "duration_seconds",
    [0.0, -0.1, float("inf"), float("nan")],
)
def test_native_helper_rejects_invalid_duration_without_rpc(
    duration_seconds,
):
    from g1_locomotion_control.sport_helper import (
        execute_sport_request,
    )

    client = Mock()
    with pytest.raises(
        ValueError, match="duration_seconds",
    ):
        execute_sport_request(
            client,
            {"action": "stop"},
            duration_seconds=duration_seconds,
        )

    client.SetVelocity.assert_not_called()


def test_native_helper_health_is_read_only():
    from g1_locomotion_control.sport_helper import (
        execute_sport_request,
    )

    client = Mock()
    response = execute_sport_request(
        client,
        {"action": "health"},
        duration_seconds=0.25,
        currently_stopped=False,
    )

    client.SetVelocity.assert_not_called()
    assert decode_sport_response(response) == {
        "action": "health",
        "ok": True,
        "stopped": False,
    }


def test_native_helper_initializes_pinned_g1_client():
    from g1_locomotion_control.sport_helper import (
        create_loco_client,
    )

    channel_factory = Mock()
    client = Mock()

    result = create_loco_client(
        interface="eth0",
        rpc_timeout_seconds=0.1,
        channel_factory=channel_factory,
        client_factory=lambda: client,
    )

    channel_factory.assert_called_once_with(0, "eth0")
    client.SetTimeout.assert_called_once_with(0.1)
    client.Init.assert_called_once_with()
    assert result is client


def test_native_helper_loads_only_official_g1_bindings():
    from g1_locomotion_control.sport_helper import (
        load_native_bindings,
    )

    channel_factory = Mock()
    client_factory = Mock()
    modules = {
        "unitree_sdk2py.core.channel": Mock(
            ChannelFactoryInitialize=channel_factory,
        ),
        "unitree_sdk2py.g1.loco.g1_loco_client": Mock(
            LocoClient=client_factory,
        ),
    }
    importer = Mock(side_effect=modules.__getitem__)

    assert load_native_bindings(importer) == (
        channel_factory, client_factory,
    )
    assert [
        item.args[0] for item in importer.call_args_list
    ] == list(modules)


def test_native_helper_parent_eof_forces_stop():
    from g1_locomotion_control.sport_helper import run_helper

    client = Mock()
    client.SetVelocity.return_value = 0
    input_stream = StringIO(
        encode_sport_move_request(0.12, -0.03, 0.25)
        + chr(10)
    )
    output_stream = StringIO()

    run_helper(
        client,
        input_stream=input_stream,
        output_stream=output_stream,
        duration_seconds=0.25,
    )

    assert [
        item.args for item in client.SetVelocity.call_args_list
    ] == [
        (0.12, -0.03, 0.25, 0.25),
        (0.0, 0.0, 0.0, 0.25),
    ]
    responses = [
        decode_sport_response(line)
        for line in output_stream.getvalue().splitlines()
    ]
    assert responses == [{
        "action": "move",
        "ok": True,
        "stopped": False,
    }]


def test_native_helper_main_wires_official_client_and_eof_stop():
    from g1_locomotion_control.sport_helper import main

    channel_factory = Mock()
    client = Mock()
    client.SetVelocity.return_value = 0
    client_factory = Mock(return_value=client)
    binding_loader = Mock(
        return_value=(channel_factory, client_factory)
    )
    output_stream = StringIO()

    assert main(
        ["eth0", "--duration-seconds", "0.25",
         "--rpc-timeout-seconds", "1.0"],
        input_stream=StringIO(
            encode_sport_health_request() + chr(10)
        ),
        output_stream=output_stream,
        binding_loader=binding_loader,
    ) == 0

    binding_loader.assert_called_once_with()
    channel_factory.assert_called_once_with(0, "eth0")
    client.SetTimeout.assert_called_once_with(1.0)
    client.Init.assert_called_once_with()
    client.SetVelocity.assert_called_once_with(
        0.0, 0.0, 0.0, 0.25
    )


def test_sport_helper_module_executes_main():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "g1_locomotion_control.sport_helper",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_package_installs_native_sport_helper():
    setup_text = (
        Path(__file__).parents[1] / "setup.py"
    ).read_text(encoding="utf-8")

    assert '"g1_sport_helper = "' in setup_text
    assert (
        '"g1_locomotion_control.sport_helper:main"'
        in setup_text
    )


def test_package_installs_explicit_physical_arbiter():
    setup_text = (
        Path(__file__).parents[1] / "setup.py"
    ).read_text(encoding="utf-8")
    assert '"g1_physical_locomotion_arbiter = "' in setup_text
    assert (
        '"g1_locomotion_control.node:physical_main"'
        in setup_text
    )


def test_package_declares_hardware_bridge_runtime():
    package_text = (
        Path(__file__).parents[1] / "package.xml"
    ).read_text(encoding="utf-8")

    assert (
        "<exec_depend>g1_hardware_bridge</exec_depend>"
        in package_text
    )


def test_sport_helper_exchange_starts_isolated_process():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    process = Mock()
    process.stdin = Mock()
    process.stdout = StringIO()
    process.poll.return_value = None
    process_factory = Mock(return_value=process)
    environment = {"ROS_DOMAIN_ID": "0"}
    exchange = SportHelperExchange(
        interface="eth0",
        environment=environment,
        response_timeout_seconds=0.5,
        process_factory=process_factory,
    )

    exchange.start()

    process_factory.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "g1_locomotion_control.sport_helper",
            "eth0",
            "--rpc-timeout-seconds",
            "0.25",
        ],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )


def test_sport_helper_exchange_timeout_closes_parent_stream():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    process = Mock()
    process.stdin = Mock()
    process.stdout = StringIO()
    process.poll.return_value = None
    process_factory = Mock(return_value=process)
    wait_readable = Mock(return_value=False)
    exchange = SportHelperExchange(
        interface="eth0",
        environment={"ROS_DOMAIN_ID": "0"},
        response_timeout_seconds=0.5,
        process_factory=process_factory,
        wait_readable=wait_readable,
    )
    exchange.start()

    with pytest.raises(
        SportBackendError,
        match="SPORT_HELPER_TIMEOUT",
    ):
        exchange(encode_sport_health_request())

    process.stdin.write.assert_called_once_with(
        encode_sport_health_request() + chr(10)
    )
    process.stdin.flush.assert_called_once_with()
    process.stdin.close.assert_called_once_with()
    wait_readable.assert_called_once_with(
        process.stdout, 0.5
    )
    process.terminate.assert_not_called()
    process.wait.assert_called_once_with(timeout=0.5)


@pytest.mark.parametrize(
    "timeout",
    [0.0, -0.1, float("inf"), float("nan")],
)
def test_sport_helper_exchange_rejects_invalid_timeout(timeout):
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    with pytest.raises(
        ValueError,
        match="response_timeout_seconds",
    ):
        SportHelperExchange(
            interface="eth0",
            environment={"ROS_DOMAIN_ID": "0"},
            response_timeout_seconds=timeout,
        )


def test_sport_helper_exchange_rejects_duplicate_start():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    process = Mock()
    process.stdin = Mock()
    process.stdout = Mock()
    process.poll.return_value = None
    process_factory = Mock(return_value=process)
    exchange = SportHelperExchange(
        interface="eth0",
        environment={"ROS_DOMAIN_ID": "0"},
        response_timeout_seconds=0.5,
        process_factory=process_factory,
    )
    exchange.start()

    with pytest.raises(
        RuntimeError,
        match="already running",
    ):
        exchange.start()

    process_factory.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "g1_locomotion_control.sport_helper",
            "eth0",
            "--rpc-timeout-seconds",
            "0.25",
        ],
        env={"ROS_DOMAIN_ID": "0"},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )


def test_sport_helper_exchange_returns_ready_response():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    request = encode_sport_health_request()
    response = encode_sport_response(
        action="health", ok=True, stopped=True
    )
    process = Mock()
    process.stdin = StringIO()
    process.stdout = StringIO(response + chr(10))
    process.poll.return_value = None
    wait_readable = Mock(return_value=True)
    exchange = SportHelperExchange(
        interface="eth0",
        environment={"ROS_DOMAIN_ID": "0"},
        response_timeout_seconds=0.5,
        process_factory=Mock(return_value=process),
        wait_readable=wait_readable,
    )
    exchange.start()

    assert exchange(request) == response + chr(10)
    assert process.stdin.getvalue() == request + chr(10)
    wait_readable.assert_called_once_with(
        process.stdout, 0.5
    )


def test_sport_helper_exchange_reaps_child_without_pipes():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    process = Mock()
    process.stdin = None
    process.stdout = Mock()
    process.poll.return_value = None
    exchange = SportHelperExchange(
        interface="eth0",
        environment={"ROS_DOMAIN_ID": "0"},
        response_timeout_seconds=0.5,
        process_factory=Mock(return_value=process),
    )

    with pytest.raises(
        RuntimeError,
        match="pipes unavailable",
    ):
        exchange.start()

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=0.5)


def test_supervised_sport_backend_owns_exchange_lifecycle():
    from g1_locomotion_control.sport_process import (
        create_supervised_sport_backend,
    )

    environment = {"ROS_DOMAIN_ID": "0"}
    environment_factory = Mock(return_value=environment)
    exchange = Mock()
    exchange_factory = Mock(return_value=exchange)

    backend = create_supervised_sport_backend(
        interface="eth0",
        response_timeout_seconds=1.5,
        rpc_timeout_seconds=1.0,
        environment_factory=environment_factory,
        exchange_factory=exchange_factory,
    )

    assert isinstance(backend, SportBackend)
    environment_factory.assert_called_once_with()
    exchange_factory.assert_called_once_with(
        interface="eth0",
        environment=environment,
        response_timeout_seconds=1.5,
        rpc_timeout_seconds=1.0,
    )
    exchange.start.assert_called_once_with()

    backend.shutdown_transport()

    exchange.stop.assert_called_once_with(timeout=1.5)


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


def test_package_owns_explicit_physical_pc2_profile():
    setup_text = (
        Path(__file__).parents[1] / "setup.py"
    ).read_text(encoding="utf-8")
    assert 'glob("config/*.yaml")' in setup_text

    profile = yaml.safe_load((
        Path(__file__).parents[1]
        / "config"
        / "physical_pc2.yaml"
    ).read_text(encoding="utf-8"))
    parameters = profile[
        "g1_locomotion_arbiter"
    ]["ros__parameters"]

    assert parameters == {
        "backend": "sport",
        "sport_interface": "eth0",
        "sport_response_timeout_seconds": 1.5,
        "sport_rpc_timeout_seconds": 1.0,
        "command_timeout_seconds": 0.25,
        "max_forward_mps": 0.15,
        "max_lateral_mps": 0.01,
        "max_yaw_rate_rps": 0.40,
        "watchdog_rate_hz": 50.0,
        "state_publish_rate_hz": 10.0,
    }


def test_sport_helper_exchange_passes_shorter_rpc_timeout():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    process = Mock()
    process.stdin = Mock()
    process.stdout = StringIO()
    process.poll.return_value = None
    process_factory = Mock(return_value=process)
    exchange = SportHelperExchange(
        interface="eth0",
        environment={"ROS_DOMAIN_ID": "0"},
        response_timeout_seconds=1.5,
        rpc_timeout_seconds=1.0,
        process_factory=process_factory,
    )

    exchange.start()

    command = process_factory.call_args.args[0]
    assert command[-2:] == [
        "--rpc-timeout-seconds",
        "1.0",
    ]


def test_sport_helper_exchange_requires_longer_parent_timeout():
    from g1_locomotion_control.sport_process import (
        SportHelperExchange,
    )

    with pytest.raises(
        ValueError,
        match="greater than rpc_timeout_seconds",
    ):
        SportHelperExchange(
            interface="eth0",
            environment={"ROS_DOMAIN_ID": "0"},
            response_timeout_seconds=1.0,
            rpc_timeout_seconds=1.0,
        )


def test_pc2_build_selects_physical_locomotion_packages():
    build_script = (
        Path(__file__).parents[5] / "scripts" / "build.sh"
    ).read_text(encoding="utf-8")

    assert "g1_locomotion_interfaces" in build_script
    assert "g1_locomotion_control" in build_script
