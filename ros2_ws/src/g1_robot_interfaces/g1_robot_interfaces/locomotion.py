"""Shared validation for backend-neutral locomotion commands."""

from dataclasses import dataclass
import math


OWNER_NONE = 0
OWNER_MANUAL = 1
OWNER_NAVIGATION = 2
OWNER_STAIR_POLICY = 3

COMMAND_OWNERS = frozenset({
    OWNER_MANUAL,
    OWNER_NAVIGATION,
    OWNER_STAIR_POLICY,
})

UNSUPPORTED_AXIS_TOLERANCE = 1e-9


def _finite_float(value, name):
    """Return a finite float with a deterministic validation error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PlanarVelocityLimits:
    """Positive hard limits for a planar locomotion command."""

    forward_mps: float
    lateral_mps: float
    yaw_rate_rps: float

    def __post_init__(self):
        for field_name in (
            "forward_mps",
            "lateral_mps",
            "yaw_rate_rps",
        ):
            try:
                value = _finite_float(
                    getattr(self, field_name), field_name
                )
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be finite and positive"
                ) from exc
            if value <= 0.0:
                raise ValueError(
                    f"{field_name} must be finite and positive"
                )
            object.__setattr__(self, field_name, value)


def validate_locomotion_owner(value, allow_none=False):
    """Return a supported locomotion owner."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("unsupported locomotion owner")
    if value == OWNER_NONE:
        if allow_none:
            return value
        raise ValueError("active locomotion owner cannot be NONE")
    if value not in COMMAND_OWNERS:
        raise ValueError("unsupported locomotion owner")
    return value


def validate_lease_id(value):
    """Return a nonempty opaque lease identifier unchanged."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lease identifier must be nonempty")
    return value


def validate_planar_velocity(
    *,
    linear_x,
    linear_y,
    linear_z,
    angular_x,
    angular_y,
    angular_z,
    limits,
):
    """Validate and return forward, lateral, and yaw velocity."""
    if not isinstance(limits, PlanarVelocityLimits):
        raise ValueError("limits must be PlanarVelocityLimits")

    components = {
        "linear_x": _finite_float(linear_x, "linear_x"),
        "linear_y": _finite_float(linear_y, "linear_y"),
        "linear_z": _finite_float(linear_z, "linear_z"),
        "angular_x": _finite_float(angular_x, "angular_x"),
        "angular_y": _finite_float(angular_y, "angular_y"),
        "angular_z": _finite_float(angular_z, "angular_z"),
    }

    unsupported = (
        components["linear_z"],
        components["angular_x"],
        components["angular_y"],
    )
    if any(
        abs(value) > UNSUPPORTED_AXIS_TOLERANCE
        for value in unsupported
    ):
        raise ValueError("unsupported locomotion velocity axis")

    bounded = (
        ("linear_x", limits.forward_mps),
        ("linear_y", limits.lateral_mps),
        ("angular_z", limits.yaw_rate_rps),
    )
    for field_name, limit in bounded:
        if abs(components[field_name]) > limit:
            raise ValueError(f"{field_name} exceeds configured hard limit")

    return (
        components["linear_x"],
        components["linear_y"],
        components["angular_z"],
    )
