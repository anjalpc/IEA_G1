"""Simulation-only planar carrier for the harnessed G1."""

from dataclasses import dataclass
import math


LATERAL_TOLERANCE = 1e-9
ANGULAR_TOLERANCE = 1e-12


def _finite(value, name):
    """Return one finite float."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value, name):
    """Return one finite positive float."""
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(
            f"{name} must be finite and positive"
        )
    return result


def _normalize_yaw(value):
    """Normalize yaw to the half-open minus-pi to pi range."""
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PlanarVelocity:
    """Body-frame velocity of the simulation carrier."""

    forward_mps: float
    lateral_mps: float
    yaw_rate_rps: float


@dataclass(frozen=True)
class PlanarPose:
    """World-frame pose of the simulation carrier."""

    x: float
    y: float
    yaw: float

    def __post_init__(self):
        object.__setattr__(self, "x", _finite(self.x, "x"))
        object.__setattr__(self, "y", _finite(self.y, "y"))
        object.__setattr__(
            self,
            "yaw",
            _normalize_yaw(_finite(self.yaw, "yaw")),
        )


class PlanarCarrier:
    """Integrate bounded body-frame velocity in simulation time."""

    def __init__(
        self,
        *,
        command_timeout_seconds,
        max_forward_mps,
        max_lateral_mps,
        max_yaw_rate_rps,
        allow_lateral,
        initial_pose=PlanarPose(0.0, 0.0, 0.0),
    ):
        if not isinstance(initial_pose, PlanarPose):
            raise ValueError(
                "initial_pose must be PlanarPose"
            )
        if not isinstance(allow_lateral, bool):
            raise ValueError("allow_lateral must be boolean")

        self._timeout = _positive(
            command_timeout_seconds,
            "command_timeout_seconds",
        )
        self._forward_limit = _positive(
            max_forward_mps,
            "max_forward_mps",
        )
        self._lateral_limit = _positive(
            max_lateral_mps,
            "max_lateral_mps",
        )
        self._yaw_limit = _positive(
            max_yaw_rate_rps,
            "max_yaw_rate_rps",
        )
        self._allow_lateral = allow_lateral
        self._pose = initial_pose
        self._forward = 0.0
        self._lateral = 0.0
        self._yaw_rate = 0.0
        self._last_time = None
        self._command_time = None
        self._stopped = True

    @property
    def pose(self):
        """Return the current immutable planar pose."""
        return self._pose

    @property
    def stopped(self):
        """Return whether the carrier velocity is zero."""
        return self._stopped

    @property
    def velocity(self):
        """Return the current body-frame planar velocity."""
        return PlanarVelocity(
            forward_mps=self._forward,
            lateral_mps=self._lateral,
            yaw_rate_rps=self._yaw_rate,
        )

    def _validate_time(self, simulation_time):
        value = _finite(
            simulation_time,
            "simulation_time",
        )
        if self._last_time is not None and value < self._last_time:
            raise ValueError("simulation time moved backwards")
        return value

    def command(
        self,
        *,
        forward_mps,
        lateral_mps,
        yaw_rate_rps,
        simulation_time,
    ):
        """Accept one bounded simulation-only velocity."""
        try:
            forward = _finite(
                forward_mps,
                "forward_mps",
            )
            lateral = _finite(
                lateral_mps,
                "lateral_mps",
            )
            yaw_rate = _finite(
                yaw_rate_rps,
                "yaw_rate_rps",
            )
            if abs(forward) > self._forward_limit:
                raise ValueError(
                    "forward_mps exceeds configured limit"
                )
            if abs(lateral) > self._lateral_limit:
                raise ValueError(
                    "lateral_mps exceeds configured limit"
                )
            if abs(yaw_rate) > self._yaw_limit:
                raise ValueError(
                    "yaw_rate_rps exceeds configured limit"
                )
            if (
                not self._allow_lateral
                and abs(lateral) > LATERAL_TOLERANCE
            ):
                raise ValueError(
                    "lateral simulation velocity is disabled"
                )
        except ValueError:
            self.stop()
            raise

        now = self._validate_time(simulation_time)
        self.advance(simulation_time=now)
        self._forward = forward
        self._lateral = (
            lateral if self._allow_lateral else 0.0
        )
        self._yaw_rate = yaw_rate
        self._command_time = now
        self._stopped = (
            self._forward == 0.0
            and self._lateral == 0.0
            and self._yaw_rate == 0.0
        )

    def _integrate(self, duration):
        if duration <= 0.0 or self._stopped:
            return

        x = self._pose.x
        y = self._pose.y
        yaw = self._pose.yaw
        yaw_rate = self._yaw_rate

        if abs(yaw_rate) <= ANGULAR_TOLERANCE:
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            x += duration * (
                self._forward * cos_yaw
                - self._lateral * sin_yaw
            )
            y += duration * (
                self._forward * sin_yaw
                + self._lateral * cos_yaw
            )
            new_yaw = yaw
        else:
            new_yaw = yaw + yaw_rate * duration
            x += (
                self._forward
                / yaw_rate
                * (math.sin(new_yaw) - math.sin(yaw))
            )
            x += (
                self._lateral
                / yaw_rate
                * (math.cos(new_yaw) - math.cos(yaw))
            )
            y += (
                self._forward
                / yaw_rate
                * (math.cos(yaw) - math.cos(new_yaw))
            )
            y += (
                self._lateral
                / yaw_rate
                * (math.sin(new_yaw) - math.sin(yaw))
            )

        self._pose = PlanarPose(
            x=x,
            y=y,
            yaw=new_yaw,
        )

    def advance(self, *, simulation_time):
        """Advance to a monotonic simulation timestamp."""
        now = self._validate_time(simulation_time)

        if self._last_time is None:
            self._last_time = now
            return self._pose

        motion_end = now
        if self._command_time is not None:
            deadline = self._command_time + self._timeout
            motion_end = min(now, deadline)

        duration = max(0.0, motion_end - self._last_time)
        self._integrate(duration)
        self._last_time = now

        if (
            self._command_time is not None
            and now >= self._command_time + self._timeout
        ):
            self.stop()

        return self._pose

    def stop(self):
        """Set zero velocity immediately and idempotently."""
        self._forward = 0.0
        self._lateral = 0.0
        self._yaw_rate = 0.0
        self._stopped = True
