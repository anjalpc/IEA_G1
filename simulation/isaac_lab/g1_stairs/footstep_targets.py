"""Geometry-derived sequential footstep targets for stair-policy experiments.

Pure geometry and progression logic. It neither imports Isaac Lab nor modifies
legacy acquisition/reward/evaluation state. Coordinates are local to course origin.
The planner describes goals; contact-qualified completion is a separate contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin


@dataclass(frozen=True)
class StairCourse:
    first_riser_x: float
    tread_depth: float
    riser_height: float
    step_count: int
    centre_y: float = 0.0
    foot_separation: float = 0.16
    tread_margin: float = 0.02

    def __post_init__(self) -> None:
        if self.step_count < 1:
            raise ValueError('step_count must be positive')
        if self.tread_depth <= 0 or self.riser_height <= 0:
            raise ValueError('stair dimensions must be positive')
        if not 0 < 2*self.tread_margin < self.tread_depth:
            raise ValueError('tread_margin must leave a landing interval')
        if self.foot_separation <= 0:
            raise ValueError('foot_separation must be positive')


@dataclass(frozen=True)
class FootstepTarget:
    sequence_index: int
    foot: int  # 0 left, 1 right; matches explicit planner convention, not arbitrary link order
    x: float
    y: float
    z: float
    yaw: float
    terrain_level: int


def planned_targets(course: StairCourse, first_foot: int) -> tuple[FootstepTarget, ...]:
    """One planned ascent touchdown per stair, then both feet on platform.

    The near/far platform targets preserve alternation. This is a reference
    plan, not an assertion that three steps have physically been completed.
    """
    if first_foot not in (0, 1):
        raise ValueError('first_foot must be 0 (left) or 1 (right)')
    out = []
    count = course.step_count + 2
    for index in range(count):
        foot = (first_foot + index) % 2
        level = min(index + 1, course.step_count)
        x = (course.first_riser_x + (index + .5) * course.tread_depth)
        y = course.centre_y + (.5 if foot == 0 else -.5)*course.foot_separation
        out.append(FootstepTarget(index, foot, x, y, level*course.riser_height, 0.0, level))
    return tuple(out)


def relative_target(target: FootstepTarget, root_x: float, root_y: float, root_yaw: float) -> tuple[float, float, float, float]:
    """SE(2) robot-relative XY, world-height Z, and wrapped relative yaw."""
    dx, dy = target.x - root_x, target.y - root_y
    x = cos(root_yaw)*dx + sin(root_yaw)*dy
    y = -sin(root_yaw)*dx + cos(root_yaw)*dy
    from math import atan2
    yaw = atan2(sin(target.yaw-root_yaw), cos(target.yaw-root_yaw))
    return (x, y, target.z, yaw)


@dataclass
class FootstepProgress:
    """Explicit task state. Strict evaluator remains fully independent."""
    targets: tuple[FootstepTarget, ...]
    next_index: int = 0
    attempted: int = 0
    completed: int = 0
    wrong_foot: int = 0
    same_level_replants: int = 0
    _last_level: int = -1

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError('targets cannot be empty')

    def next_two(self) -> tuple[FootstepTarget | None, FootstepTarget | None]:
        return tuple(self.targets[i] if i < len(self.targets) else None for i in (self.next_index, self.next_index+1))

    def touchdown(self, *, foot: int, level: int, within_target: bool, supported: bool) -> bool:
        """Advance only for assigned foot, correct level, supported touchdown.

        Caller must supply actual support/contact and landing validation; this
        method cannot manufacture contacts from positions or reset jumps.
        """
        if foot not in (0, 1):
            raise ValueError('foot must be 0 or 1')
        if not supported or self.next_index >= len(self.targets):
            return False
        self.attempted += 1
        target = self.targets[self.next_index]
        if within_target and foot != target.foot:
            self.wrong_foot += 1
        if level == self._last_level:
            self.same_level_replants += 1
        self._last_level = level
        valid = within_target and level == target.terrain_level and foot == target.foot
        if valid:
            self.next_index += 1
            self.completed += 1
        return valid

    def reset(self) -> None:
        self.next_index = self.attempted = self.completed = 0
        self.wrong_foot = self.same_level_replants = 0
        self._last_level = -1


METRIC_CONTRACT = 'footstep_target_v1'


def metrics(progress: FootstepProgress) -> dict[str, int | float | str | None]:
    """Contract-versioned, explicit denominators; never masquerade as CLEAN_STEP_OVER."""
    return {
        'metric_contract': METRIC_CONTRACT,
        'footstep_target_attempts': progress.attempted,
        'footstep_target_completions': progress.completed,
        'footstep_target_completion_rate': (progress.completed/progress.attempted if progress.attempted else None),
        'footstep_wrong_foot_landings': progress.wrong_foot,
        'footstep_same_level_replants': progress.same_level_replants,
        'footstep_next_target_index': progress.next_index,
    }
