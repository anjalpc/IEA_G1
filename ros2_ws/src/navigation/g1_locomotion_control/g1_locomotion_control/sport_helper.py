"""Map strict Sport requests to pinned Unitree G1 calls."""

import argparse
import importlib
import math
import sys

from g1_locomotion_control.sport_backend import (
    decode_sport_request,
    decode_sport_response,
    encode_sport_response,
)


def run_helper(
    client, *, input_stream, output_stream,
    duration_seconds,
):
    """Serve strict requests and stop on parent EOF."""
    currently_stopped = True
    try:
        for line in input_stream:
            request = decode_sport_request(line)
            if request is None:
                continue
            response = execute_sport_request(
                client,
                request,
                duration_seconds=duration_seconds,
                currently_stopped=currently_stopped,
            )
            acknowledged = decode_sport_response(response)
            if acknowledged["ok"]:
                currently_stopped = acknowledged["stopped"]
            output_stream.write(response + chr(10))
            output_stream.flush()
    finally:
        execute_sport_request(
            client,
            {"action": "stop"},
            duration_seconds=duration_seconds,
            currently_stopped=currently_stopped,
        )


def load_native_bindings(
    importer=importlib.import_module,
):
    """Load only the official Unitree G1 bindings lazily."""
    channel_module = importer(
        "unitree_sdk2py.core.channel"
    )
    client_module = importer(
        "unitree_sdk2py.g1.loco.g1_loco_client"
    )
    return (
        channel_module.ChannelFactoryInitialize,
        client_module.LocoClient,
    )


def create_loco_client(
    *, interface, rpc_timeout_seconds,
    channel_factory, client_factory,
):
    """Initialize the pinned G1 client through injected bindings."""
    rpc_timeout_seconds = float(rpc_timeout_seconds)
    if (
        not math.isfinite(rpc_timeout_seconds)
        or rpc_timeout_seconds <= 0.0
    ):
        raise ValueError(
            "rpc_timeout_seconds must be finite and positive"
        )

    channel_factory(0, str(interface))
    client = client_factory()
    client.SetTimeout(rpc_timeout_seconds)
    client.Init()
    return client


def main(
    argv=None, *, input_stream=None, output_stream=None,
    binding_loader=None,
):
    """Run the native Sport helper over line-delimited streams."""
    parser = argparse.ArgumentParser()
    parser.add_argument("interface")
    parser.add_argument(
        "--duration-seconds", type=float, default=0.25
    )
    parser.add_argument(
        "--rpc-timeout-seconds", type=float, default=1.0
    )
    arguments = parser.parse_args(argv)

    if binding_loader is None:
        binding_loader = load_native_bindings
    channel_factory, client_factory = binding_loader()
    client = create_loco_client(
        interface=arguments.interface,
        rpc_timeout_seconds=arguments.rpc_timeout_seconds,
        channel_factory=channel_factory,
        client_factory=client_factory,
    )
    run_helper(
        client,
        input_stream=(
            sys.stdin if input_stream is None else input_stream
        ),
        output_stream=(
            sys.stdout if output_stream is None else output_stream
        ),
        duration_seconds=arguments.duration_seconds,
    )
    return 0


def execute_sport_request(
    client, request, *, duration_seconds,
    currently_stopped=True,
):
    """Execute one validated helper request."""
    duration_seconds = float(duration_seconds)
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds <= 0.0
    ):
        raise ValueError(
            "duration_seconds must be finite and positive"
        )
    action = request.get("action")
    if request == {"action": "health"}:
        return encode_sport_response(
            action="health",
            ok=True,
            stopped=bool(currently_stopped),
        )
    if request == {"action": "stop"}:
        velocity = (0.0, 0.0, 0.0)
    elif (
        action == "move"
        and set(request) == {
            "action",
            "forward_mps",
            "lateral_mps",
            "yaw_rate_rps",
        }
    ):
        velocity = (
            request["forward_mps"],
            request["lateral_mps"],
            request["yaw_rate_rps"],
        )
    else:
        raise ValueError("unsupported Sport action")

    code = client.SetVelocity(
        *velocity, float(duration_seconds),
    )
    ok = code == 0
    stopped = ok and velocity == (0.0, 0.0, 0.0)
    return encode_sport_response(
        action=action,
        ok=ok,
        stopped=stopped,
    )



if __name__ == "__main__":
    raise SystemExit(main())
