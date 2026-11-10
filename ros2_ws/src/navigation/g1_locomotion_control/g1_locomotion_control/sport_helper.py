"""Map strict Sport requests to pinned Unitree G1 calls."""

import importlib
import math

from g1_locomotion_control.sport_backend import (
    encode_sport_response,
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

