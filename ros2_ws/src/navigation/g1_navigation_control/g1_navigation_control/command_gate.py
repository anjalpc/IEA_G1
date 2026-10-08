"""Pure lease-aware validation for Nav2 velocity commands."""

from dataclasses import dataclass

from g1_robot_interfaces.locomotion import (
    OWNER_NAVIGATION,
    PlanarVelocityLimits,
    validate_lease_id,
    validate_planar_velocity,
)


LATERAL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class NavigationCommand:
    """Validated planar command carrying navigation authority."""

    owner: int
    lease_id: str
    linear_x: float
    linear_y: float
    angular_z: float


class NavigationCommandGate:
    """Translate Nav2 velocities only while a lease is active."""

    def __init__(self, *, limits, allow_lateral=False):
        if not isinstance(limits, PlanarVelocityLimits):
            raise ValueError(
                "limits must be PlanarVelocityLimits"
            )
        if not isinstance(allow_lateral, bool):
            raise ValueError("allow_lateral must be boolean")

        self._limits = limits
        self._allow_lateral = allow_lateral
        self._lease_id = None

    @property
    def active(self):
        """Return whether navigation currently owns a lease."""
        return self._lease_id is not None

    def grant(self, lease_id):
        """Activate one opaque navigation lease."""
        validate_lease_id(lease_id)
        if self.active:
            raise RuntimeError("navigation lease already active")
        self._lease_id = lease_id

    def clear(self):
        """Forget the active lease and remove command authority."""
        self._lease_id = None

    def translate(
        self,
        *,
        linear_x,
        linear_y,
        linear_z,
        angular_x,
        angular_y,
        angular_z,
    ):
        """Validate a Nav2 velocity under the active lease."""
        if not self.active:
            return None

        forward, lateral, yaw = validate_planar_velocity(
            linear_x=linear_x,
            linear_y=linear_y,
            linear_z=linear_z,
            angular_x=angular_x,
            angular_y=angular_y,
            angular_z=angular_z,
            limits=self._limits,
        )

        if (
            not self._allow_lateral
            and abs(lateral) > LATERAL_TOLERANCE
        ):
            raise ValueError(
                "lateral navigation velocity is disabled"
            )

        if not self._allow_lateral:
            lateral = 0.0

        return NavigationCommand(
            owner=OWNER_NAVIGATION,
            lease_id=self._lease_id,
            linear_x=forward,
            linear_y=lateral,
            angular_z=yaw,
        )

    def stop(self):
        """Create an explicit zero command under the active lease."""
        return self.translate(
            linear_x=0.0,
            linear_y=0.0,
            linear_z=0.0,
            angular_x=0.0,
            angular_y=0.0,
            angular_z=0.0,
        )
