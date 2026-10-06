"""Pure simulation-time conversion and publication scheduling."""

from dataclasses import dataclass
import math
from numbers import Real


_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class SimulationTimestamp:
    """ROS-compatible timestamp without a ROS runtime dependency."""

    sec: int
    nanosec: int


@dataclass(frozen=True)
class SimulationPublicationDecision:
    """Publication work due after one completed physics step."""

    clock_stamp: SimulationTimestamp
    publish_state: bool


def _finite_seconds(value, *, name, allow_zero):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        qualifier = (
            "finite and non-negative"
            if allow_zero
            else "finite and positive"
        )
        raise ValueError(f"{name} must be {qualifier}")

    if allow_zero:
        valid = value >= 0.0
        qualifier = "finite and non-negative"
    else:
        valid = value > 0.0
        qualifier = "finite and positive"

    if not valid:
        raise ValueError(f"{name} must be {qualifier}")

    nanoseconds = round(
        float(value) * _NANOSECONDS_PER_SECOND
    )
    if not allow_zero and nanoseconds <= 0:
        raise ValueError(
            f"{name} must be finite and positive"
        )

    return nanoseconds


def _timestamp_from_nanoseconds(total_nanoseconds):
    sec, nanosec = divmod(
        total_nanoseconds,
        _NANOSECONDS_PER_SECOND,
    )
    return SimulationTimestamp(
        sec=sec,
        nanosec=nanosec,
    )


def simulation_timestamp(simulation_time):
    """Convert finite non-negative simulation seconds to a timestamp."""

    total_nanoseconds = _finite_seconds(
        simulation_time,
        name="simulation_time",
        allow_zero=True,
    )
    return _timestamp_from_nanoseconds(total_nanoseconds)


class SimulationPublicationSchedule:
    """Schedule clock and raw-state publication from simulation time."""

    def __init__(self, *, state_period_seconds):
        self._state_period_nanoseconds = _finite_seconds(
            state_period_seconds,
            name="state_period_seconds",
            allow_zero=False,
        )
        self._next_state_nanoseconds = (
            self._state_period_nanoseconds
        )
        self._last_step_nanoseconds = None

    def after_step(self, simulation_time):
        """Return publication work due after a completed physics step."""

        step_nanoseconds = _finite_seconds(
            simulation_time,
            name="simulation_time",
            allow_zero=True,
        )

        if (
            self._last_step_nanoseconds is not None
            and step_nanoseconds <= self._last_step_nanoseconds
        ):
            raise ValueError(
                "simulation_time must be strictly increasing"
            )

        publish_state = (
            step_nanoseconds >= self._next_state_nanoseconds
        )
        if publish_state:
            while (
                self._next_state_nanoseconds
                <= step_nanoseconds
            ):
                self._next_state_nanoseconds += (
                    self._state_period_nanoseconds
                )

        self._last_step_nanoseconds = step_nanoseconds

        return SimulationPublicationDecision(
            clock_stamp=_timestamp_from_nanoseconds(
                step_nanoseconds
            ),
            publish_state=publish_state,
        )
