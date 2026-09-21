"""Physical state detector for G1 single-step ascent.

This module contains no Isaac Lab dependencies.  It converts physical
measurements into monotonic step-up states that can later be used both for
evaluation and one-time transition rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class StepUpState(IntEnum):
    APPROACH = 0
    FIRST_SUPPORT = 1
    BASE_ABOVE = 2
    BOTH_FEET_SUPPORTED = 3
    COMPLETED = 4


@dataclass(frozen=True)
class FootSupportObservation:
    """Measurements required to decide whether one foot has stable upper support."""

    in_upper_region: bool
    height_ok: bool
    vertical_force_n: float
    speed_mps: float


@dataclass(frozen=True)
class StepUpObservation:
    """Physical measurements for one control step."""

    left: FootSupportObservation
    right: FootSupportObservation

    upright_gravity_z: float

    # These are geometric/physical checks computed by the Isaac adapter.
    base_past_riser: bool
    base_height_ok: bool

    # True for shin, knee, torso, hand, etc. contact that invalidates progress.
    forbidden_contact: bool

    # Distance travelled upright after both feet have reached the upper level.
    upper_forward_continuation_m: float


@dataclass(frozen=True)
class StepUpDetectorConfig:
    minimum_support_duration_s: float = 0.15
    maximum_supported_foot_speed_mps: float = 0.15
    minimum_upright_gravity_z: float = 0.80

    # Must be calibrated from flat-ground stance data before training.
    minimum_vertical_support_force_n: float = 1.0

    completion_forward_distance_m: float = 0.40


@dataclass(frozen=True)
class StepUpTransition:
    """Result from one detector update."""

    state: StepUpState

    left_supported: bool
    right_supported: bool

    entered_first_support: bool = False
    entered_base_above: bool = False
    entered_both_feet_supported: bool = False
    entered_completed: bool = False


class StepUpStateDetector:
    """Monotonic per-episode state machine for one step-up."""

    def __init__(
        self,
        config: StepUpDetectorConfig,
        *,
        dt: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        if config.minimum_support_duration_s <= 0.0:
            raise ValueError(
                "minimum_support_duration_s must be positive"
            )

        if config.minimum_vertical_support_force_n < 0.0:
            raise ValueError(
                "minimum_vertical_support_force_n cannot be negative"
            )

        self.config = config
        self.dt = float(dt)

        self.reset()

    def reset(
        self,
        *,
        initial_state: StepUpState = StepUpState.APPROACH,
    ) -> None:
        self.state = StepUpState(initial_state)

        self._left_support_time_s = 0.0
        self._right_support_time_s = 0.0

    def _update_support_timer(
        self,
        observation: FootSupportObservation,
        previous_time_s: float,
        *,
        robot_upright: bool,
        forbidden_contact: bool,
    ) -> float:
        candidate = (
            observation.in_upper_region
            and observation.height_ok
            and observation.vertical_force_n
            >= self.config.minimum_vertical_support_force_n
            and observation.speed_mps
            <= self.config.maximum_supported_foot_speed_mps
            and robot_upright
            and not forbidden_contact
        )

        if not candidate:
            return 0.0

        return previous_time_s + self.dt

    def update(
        self,
        observation: StepUpObservation,
    ) -> StepUpTransition:
        upright = (
            observation.upright_gravity_z
            >= self.config.minimum_upright_gravity_z
        )

        self._left_support_time_s = self._update_support_timer(
            observation.left,
            self._left_support_time_s,
            robot_upright=upright,
            forbidden_contact=observation.forbidden_contact,
        )

        self._right_support_time_s = self._update_support_timer(
            observation.right,
            self._right_support_time_s,
            robot_upright=upright,
            forbidden_contact=observation.forbidden_contact,
        )

        left_supported = (
            self._left_support_time_s
            + 1e-12
            >= self.config.minimum_support_duration_s
        )
        right_supported = (
            self._right_support_time_s
            + 1e-12
            >= self.config.minimum_support_duration_s
        )

        entered_first_support = False
        entered_base_above = False
        entered_both_feet_supported = False
        entered_completed = False

        valid_progress = upright and not observation.forbidden_contact

        if (
            self.state == StepUpState.APPROACH
            and valid_progress
            and (left_supported or right_supported)
        ):
            self.state = StepUpState.FIRST_SUPPORT
            entered_first_support = True

        if (
            self.state == StepUpState.FIRST_SUPPORT
            and valid_progress
            and observation.base_past_riser
            and observation.base_height_ok
        ):
            self.state = StepUpState.BASE_ABOVE
            entered_base_above = True

        if (
            self.state == StepUpState.BASE_ABOVE
            and valid_progress
            and left_supported
            and right_supported
        ):
            self.state = StepUpState.BOTH_FEET_SUPPORTED
            entered_both_feet_supported = True

        if (
            self.state == StepUpState.BOTH_FEET_SUPPORTED
            and valid_progress
            and left_supported
            and right_supported
            and observation.upper_forward_continuation_m
            >= self.config.completion_forward_distance_m
        ):
            self.state = StepUpState.COMPLETED
            entered_completed = True

        return StepUpTransition(
            state=self.state,
            left_supported=left_supported,
            right_supported=right_supported,
            entered_first_support=entered_first_support,
            entered_base_above=entered_base_above,
            entered_both_feet_supported=entered_both_feet_supported,
            entered_completed=entered_completed,
        )
