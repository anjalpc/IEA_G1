"""Hardware-free boundary for a future Unitree Sport adapter."""

import math
import json
from typing import Protocol


SPORT_REQUEST_PREFIX = "SPORT_REQUEST="
_MOVE_FIELDS = {
    "action",
    "forward_mps",
    "lateral_mps",
    "yaw_rate_rps",
}


def _move_payload(forward_mps, lateral_mps, yaw_rate_rps):
    payload = {
        "action": "move",
        "forward_mps": float(forward_mps),
        "lateral_mps": float(lateral_mps),
        "yaw_rate_rps": float(yaw_rate_rps),
    }
    if not all(
        math.isfinite(payload[field])
        for field in _MOVE_FIELDS - {"action"}
    ):
        raise ValueError("Sport velocity must be finite")
    return payload


def encode_sport_move_request(
    forward_mps,
    lateral_mps,
    yaw_rate_rps,
):
    """Encode one validated helper MOVE request."""
    return SPORT_REQUEST_PREFIX + json.dumps(
        _move_payload(
            forward_mps,
            lateral_mps,
            yaw_rate_rps,
        ),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def encode_sport_stop_request():
    """Encode one helper STOP request."""
    return SPORT_REQUEST_PREFIX + json.dumps(
        {"action": "stop"},
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_sport_request(line):
    """Decode one strict helper request line."""
    stripped = str(line).strip()
    if not stripped.startswith(SPORT_REQUEST_PREFIX):
        return None
    payload = json.loads(
        stripped[len(SPORT_REQUEST_PREFIX):]
    )
    if payload == {"action": "stop"}:
        return payload
    if (
        not isinstance(payload, dict)
        or set(payload) != _MOVE_FIELDS
        or payload.get("action") != "move"
    ):
        raise ValueError("invalid Sport request")
    return _move_payload(
        payload["forward_mps"],
        payload["lateral_mps"],
        payload["yaw_rate_rps"],
    )


class SportBackendError(RuntimeError):
    """Stable error raised at the Sport backend boundary."""


class SportClientPort(Protocol):
    """Project-owned neutral interface for a native Sport client."""

    def move(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Forward one planar velocity command."""

    def stop_move(self):
        """Request stopped locomotion."""

    def healthy(self):
        """Return whether the native client is available."""


class SportBackend:
    """Adapt a neutral Sport client to the locomotion backend contract."""

    def __init__(self, client):
        self._client = client
        self.active = False
        self.stopped = True
        self.faulted = False

    def activate(self):
        """Activate or recover without commanding motion."""
        if self.active and not self.faulted:
            return

        try:
            client_healthy = bool(self._client.healthy())
        except Exception as exc:
            self.faulted = True
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            ) from exc

        if not client_healthy:
            self.faulted = True
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            )

        was_active = self.active
        self.active = True
        if not was_active:
            self.stopped = True
        self.faulted = False

    def command_velocity(
        self,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
    ):
        """Forward an already-bounded planar command exactly."""
        if not self.active:
            raise SportBackendError(
                "SPORT_BACKEND_INACTIVE"
            )
        if self.faulted:
            raise SportBackendError(
                "SPORT_CLIENT_UNAVAILABLE"
            )

        try:
            command = (
                float(forward_mps),
                float(lateral_mps),
                float(yaw_rate_rps),
            )
        except (TypeError, ValueError) as exc:
            raise SportBackendError(
                "SPORT_MOVE_FAILED"
            ) from exc

        if not all(math.isfinite(value) for value in command):
            raise SportBackendError(
                "SPORT_MOVE_FAILED"
            )

        try:
            self._client.move(*command)
        except Exception as exc:
            self.faulted = True
            self.stopped = False
            raise SportBackendError(
                "SPORT_MOVE_FAILED"
            ) from exc

        self.stopped = command == (0.0, 0.0, 0.0)

    def stop(self):
        """Request stop whenever the backend is active."""
        if not self.active:
            return

        try:
            self._client.stop_move()
        except Exception as exc:
            self.faulted = True
            self.stopped = False
            raise SportBackendError(
                "SPORT_STOP_FAILED"
            ) from exc

        self.stopped = True

    def healthy(self):
        """Return fail-closed backend and client health."""
        if not self.active or self.faulted:
            return False
        try:
            return bool(self._client.healthy())
        except Exception:
            return False

    def deactivate(self):
        """Deactivate locally only after a confirmed stop."""
        if not self.active:
            return
        if not self.stopped:
            raise SportBackendError(
                "SPORT_BACKEND_NOT_STOPPED"
            )
        self.active = False
