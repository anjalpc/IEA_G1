"""Simulator-independent geometry contract for the stair-ascent task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """Axis-aligned box expressed by its centre and full extents."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class StaircaseSpec:
    """A forward (+X) staircase followed by a continuous top platform."""

    riser_height: float = 0.20
    tread_depth: float = 0.25
    step_count: int = 3
    width: float = 1.2
    approach_length: float = 2.0
    platform_length: float = 3.0
    foundation_depth: float = 0.10
    start_clearance: float = 1.0

    def validate(self) -> None:
        values = (
            self.riser_height,
            self.tread_depth,
            self.width,
            self.approach_length,
            self.platform_length,
            self.foundation_depth,
            self.start_clearance,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all staircase dimensions must be positive")
        if self.step_count < 1:
            raise ValueError("step_count must be at least one")
        if self.start_clearance >= self.approach_length:
            raise ValueError("start_clearance must be shorter than approach_length")

    @property
    def total_rise(self) -> float:
        return self.step_count * self.riser_height

    @property
    def stair_run(self) -> float:
        return self.step_count * self.tread_depth

    @property
    def course_length(self) -> float:
        return self.approach_length + self.stair_run + self.platform_length

    @property
    def fixed_start_xy_yaw(self) -> tuple[float, float, float]:
        """Base start pose: one metre before the first riser, facing +X."""
        self.validate()
        return (self.approach_length - self.start_clearance, 0.0, 0.0)

    def collision_boxes(self) -> tuple[Box, ...]:
        """Return non-overlapping boxes with exact vertical riser faces.

        The course begins at x=0.  The first riser is at ``approach_length``;
        the top platform starts immediately after the third tread.
        """
        self.validate()
        boxes = [
            Box(
                center=(self.approach_length / 2.0, 0.0, -self.foundation_depth / 2.0),
                size=(self.approach_length, self.width, self.foundation_depth),
            )
        ]
        for index in range(self.step_count):
            height = (index + 1) * self.riser_height
            x0 = self.approach_length + index * self.tread_depth
            boxes.append(
                Box(
                    center=(x0 + self.tread_depth / 2.0, 0.0, height / 2.0),
                    size=(self.tread_depth, self.width, height),
                )
            )
        platform_x0 = self.approach_length + self.stair_run
        boxes.append(
            Box(
                center=(platform_x0 + self.platform_length / 2.0, 0.0, self.total_rise / 2.0),
                size=(self.platform_length, self.width, self.total_rise),
            )
        )
        return tuple(boxes)


def nominal_staircase() -> StaircaseSpec:
    """Return the agreed 20 cm rise, 25 cm tread, three-step course."""
    return StaircaseSpec()
