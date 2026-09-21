"""Reward-independent stair-climbing outcome evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Iterable


class Outcome(str, Enum):
    """Terminal episode outcomes."""

    SUCCESS = "success"
    INVALID_SIMULATION_STATE = "invalid_simulation_state"
    FALL = "fall"
    TORQUE_VIOLATION = "torque_violation"
    VELOCITY_VIOLATION = "velocity_violation"
    JOINT_VIOLATION = "joint_violation"
    FORBIDDEN_CONTACT = "forbidden_contact"
    WRONG_DIRECTION = "wrong_direction"
    SLIP = "slip"
    STALL = "stall"
    TIMEOUT = "timeout"


FORBIDDEN_CONTACTS = frozenset(
    {
        "torso",
        "pelvis",
        "left_knee",
        "right_knee",
        "left_forearm",
        "right_forearm",
        "left_hand",
        "right_hand",
    }
)


@dataclass(frozen=True)
class StairOutcomeConfig:
    """Thresholds defining success and failure independently of reward."""

    approach_length_m: float = 2.0
    start_x_m: float = 1.0
    tread_depth_m: float = 0.25
    step_count: int = 3
    required_platform_continuation_m: float = 2.0

    minimum_upright_base_height_m: float = 0.45
    maximum_absolute_roll_rad: float = 0.8
    maximum_absolute_pitch_rad: float = 0.8
    required_upright_dwell_s: float = 1.0

    wrong_direction_tolerance_m: float = 0.25
    stall_duration_s: float = 2.0
    progress_epsilon_m: float = 0.02

    forbidden_contacts: frozenset[str] = FORBIDDEN_CONTACTS

    def validate(self) -> None:
        positive_values = (
            self.approach_length_m,
            self.tread_depth_m,
            self.required_platform_continuation_m,
            self.minimum_upright_base_height_m,
            self.maximum_absolute_roll_rad,
            self.maximum_absolute_pitch_rad,
            self.required_upright_dwell_s,
            self.stall_duration_s,
            self.progress_epsilon_m,
        )

        if any(value <= 0.0 for value in positive_values):
            raise ValueError(
                "all evaluator thresholds must be positive"
            )

        if self.step_count < 1:
            raise ValueError("step_count must be at least one")

        if self.start_x_m >= self.approach_length_m:
            raise ValueError(
                "start_x_m must be before the first riser"
            )

    @property
    def first_riser_x_m(self) -> float:
        return self.approach_length_m

    @property
    def platform_start_x_m(self) -> float:
        return (
            self.approach_length_m
            + self.step_count * self.tread_depth_m
        )

    @property
    def completion_x_m(self) -> float:
        return (
            self.platform_start_x_m
            + self.required_platform_continuation_m
        )


@dataclass(frozen=True)
class StepSample:
    """Physical measurements for one environment at one control step."""

    time_s: float
    base_x_m: float
    base_height_m: float
    roll_rad: float
    pitch_rad: float
    step_index: int

    contacts: frozenset[str] = frozenset()

    joint_violation: bool = False
    velocity_violation: bool = False
    torque_violation: bool = False
    slip_violation: bool = False
    timed_out: bool = False

    def numeric_values(self) -> tuple[float, ...]:
        return (
            self.time_s,
            self.base_x_m,
            self.base_height_m,
            self.roll_rad,
            self.pitch_rad,
        )

    @classmethod
    def create(
        cls,
        *,
        time_s: float,
        base_x_m: float,
        base_height_m: float,
        roll_rad: float,
        pitch_rad: float,
        step_index: int,
        contacts: Iterable[str] = (),
        joint_violation: bool = False,
        velocity_violation: bool = False,
        torque_violation: bool = False,
        slip_violation: bool = False,
        timed_out: bool = False,
    ) -> "StepSample":
        return cls(
            time_s=time_s,
            base_x_m=base_x_m,
            base_height_m=base_height_m,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            step_index=step_index,
            contacts=frozenset(contacts),
            joint_violation=joint_violation,
            velocity_violation=velocity_violation,
            torque_violation=torque_violation,
            slip_violation=slip_violation,
            timed_out=timed_out,
        )


@dataclass(frozen=True)
class EpisodeResult:
    """Final outcome plus preserved secondary evidence."""

    outcome: Outcome
    success: bool
    termination_time_s: float
    primary_reason: str
    secondary_flags: tuple[str, ...]
    metrics: dict[str, float | int | bool]


@dataclass
class StairOutcomeEvaluator:
    """Stateful, reward-independent evaluator for one episode."""

    config: StairOutcomeConfig = field(
        default_factory=StairOutcomeConfig
    )

    maximum_x_m: float = field(init=False)
    maximum_base_height_m: float = field(init=False)
    maximum_step_reached: int = field(init=False, default=0)

    platform_reached: bool = field(init=False, default=False)
    approach_completed: bool = field(init=False, default=False)

    upright_dwell_s: float = field(init=False, default=0.0)
    completion_zone_active: bool = field(
        init=False,
        default=False,
    )

    last_time_s: float | None = field(init=False, default=None)
    last_progress_time_s: float = field(init=False, default=0.0)
    progress_reference_x_m: float = field(init=False)

    secondary_flags: set[str] = field(
        init=False,
        default_factory=set,
    )

    result: EpisodeResult | None = field(
        init=False,
        default=None,
    )

    def __post_init__(self) -> None:
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        """Reset all episode state."""

        self.maximum_x_m = self.config.start_x_m
        self.maximum_base_height_m = 0.0
        self.maximum_step_reached = 0

        self.platform_reached = False
        self.approach_completed = False

        self.upright_dwell_s = 0.0
        self.completion_zone_active = False

        self.last_time_s = None
        self.last_progress_time_s = 0.0
        self.progress_reference_x_m = self.config.start_x_m

        self.secondary_flags = set()
        self.result = None

    def update(
        self,
        sample: StepSample,
    ) -> EpisodeResult | None:
        """Consume one sample and return a result once terminal."""

        if self.result is not None:
            raise RuntimeError(
                "episode is already terminal; call reset before reuse"
            )

        if not all(
            math.isfinite(value)
            for value in sample.numeric_values()
        ):
            self.secondary_flags.add(
                Outcome.INVALID_SIMULATION_STATE.value
            )
            return self._finish(
                Outcome.INVALID_SIMULATION_STATE,
                sample,
            )

        if sample.time_s < 0.0:
            raise ValueError("sample time cannot be negative")

        if (
            self.last_time_s is not None
            and sample.time_s < self.last_time_s
        ):
            raise ValueError("sample time must be monotonic")

        first_sample = self.last_time_s is None

        delta_time = (
            0.0
            if first_sample
            else sample.time_s - self.last_time_s
        )

        self.last_time_s = sample.time_s

        if first_sample:
            self.last_progress_time_s = sample.time_s
            self.progress_reference_x_m = max(
                self.config.start_x_m,
                sample.base_x_m,
            )
        elif (
            sample.base_x_m
            >= self.progress_reference_x_m
            + self.config.progress_epsilon_m
        ):
            self.last_progress_time_s = sample.time_s
            self.progress_reference_x_m = sample.base_x_m

        self.maximum_x_m = max(
            self.maximum_x_m,
            sample.base_x_m,
        )

        self.maximum_base_height_m = max(
            self.maximum_base_height_m,
            sample.base_height_m,
        )

        self.maximum_step_reached = max(
            self.maximum_step_reached,
            sample.step_index,
        )

        self.approach_completed = (
            self.approach_completed
            or sample.base_x_m >= self.config.first_riser_x_m
        )

        self.platform_reached = (
            self.platform_reached
            or sample.base_x_m >= self.config.platform_start_x_m
        )

        forbidden = (
            sample.contacts
            & self.config.forbidden_contacts
        )

        if forbidden:
            self.secondary_flags.add(
                Outcome.FORBIDDEN_CONTACT.value
            )

        if sample.joint_violation:
            self.secondary_flags.add(
                Outcome.JOINT_VIOLATION.value
            )

        if sample.velocity_violation:
            self.secondary_flags.add(
                Outcome.VELOCITY_VIOLATION.value
            )

        if sample.torque_violation:
            self.secondary_flags.add(
                Outcome.TORQUE_VIOLATION.value
            )

        if sample.slip_violation:
            self.secondary_flags.add(
                Outcome.SLIP.value
            )

        fallen = (
            sample.base_height_m
            < self.config.minimum_upright_base_height_m
            or abs(sample.roll_rad)
            > self.config.maximum_absolute_roll_rad
            or abs(sample.pitch_rad)
            > self.config.maximum_absolute_pitch_rad
            or bool(
                {"torso", "pelvis"}
                & sample.contacts
            )
        )

        if fallen:
            self.secondary_flags.add(
                Outcome.FALL.value
            )

        wrong_direction = (
            sample.base_x_m
            < self.config.start_x_m
            - self.config.wrong_direction_tolerance_m
        )

        if wrong_direction:
            self.secondary_flags.add(
                Outcome.WRONG_DIRECTION.value
            )

        stalled = (
            sample.time_s >= self.config.stall_duration_s
            and sample.time_s - self.last_progress_time_s
            >= self.config.stall_duration_s
        )

        if stalled:
            self.secondary_flags.add(
                Outcome.STALL.value
            )

        primary_failure = self._primary_failure()

        if primary_failure is not None:
            return self._finish(
                primary_failure,
                sample,
            )

        in_completion_zone = (
            sample.base_x_m >= self.config.completion_x_m
            and sample.step_index >= self.config.step_count
        )

        upright = (
            sample.base_height_m
            >= self.config.minimum_upright_base_height_m
            and abs(sample.roll_rad)
            <= self.config.maximum_absolute_roll_rad
            and abs(sample.pitch_rad)
            <= self.config.maximum_absolute_pitch_rad
        )

        if in_completion_zone and upright:
            if self.completion_zone_active:
                self.upright_dwell_s += delta_time
            else:
                self.completion_zone_active = True
                self.upright_dwell_s = 0.0
        else:
            self.completion_zone_active = False
            self.upright_dwell_s = 0.0

        if (
            self.approach_completed
            and self.platform_reached
            and self.maximum_step_reached
            >= self.config.step_count
            and self.upright_dwell_s
            >= self.config.required_upright_dwell_s
        ):
            return self._finish(
                Outcome.SUCCESS,
                sample,
            )

        if sample.timed_out:
            self.secondary_flags.add(
                Outcome.TIMEOUT.value
            )
            return self._finish(
                Outcome.TIMEOUT,
                sample,
            )

        return None

    def _primary_failure(self) -> Outcome | None:
        """Choose exactly one primary failure by fixed priority."""

        priority = (
            Outcome.INVALID_SIMULATION_STATE,
            Outcome.FALL,
            Outcome.TORQUE_VIOLATION,
            Outcome.VELOCITY_VIOLATION,
            Outcome.JOINT_VIOLATION,
            Outcome.FORBIDDEN_CONTACT,
            Outcome.WRONG_DIRECTION,
            Outcome.STALL,
        )

        for outcome in priority:
            if outcome.value in self.secondary_flags:
                return outcome

        return None

    def _finish(
        self,
        outcome: Outcome,
        sample: StepSample,
    ) -> EpisodeResult:
        """Create and preserve the final episode result."""

        continuation_m = max(
            0.0,
            self.maximum_x_m
            - self.config.platform_start_x_m,
        )

        metrics: dict[str, float | int | bool] = {
            "maximum_x_m": self.maximum_x_m,
            "maximum_base_height_m": (
                self.maximum_base_height_m
            ),
            "maximum_step_reached": (
                self.maximum_step_reached
            ),
            "approach_completed": (
                self.approach_completed
            ),
            "platform_reached": (
                self.platform_reached
            ),
            "platform_continuation_m": continuation_m,
            "upright_dwell_s": self.upright_dwell_s,
        }

        self.result = EpisodeResult(
            outcome=outcome,
            success=outcome is Outcome.SUCCESS,
            termination_time_s=sample.time_s,
            primary_reason=outcome.value,
            secondary_flags=tuple(
                sorted(self.secondary_flags)
            ),
            metrics=metrics,
        )

        return self.result