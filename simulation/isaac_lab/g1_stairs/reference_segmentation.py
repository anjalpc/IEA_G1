"""Segment the frozen stair-ascent reference into useful stair phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_motion import ReferenceMotion


@dataclass(frozen=True)
class StairTransition:
    name: str
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    foot: str
    start_height_m: float
    end_height_m: float


@dataclass(frozen=True)
class ReferenceSegmentation:
    transitions: tuple[StairTransition, ...]
    cut_frame: int
    cut_time_s: float


def _select_distinct_peaks(
    z: np.ndarray,
    *,
    frequency_hz: float,
    window_s: float = 0.4,
    minimum_rise_m: float = 0.12,
    minimum_separation_s: float = 0.6,
) -> list[int]:
    window = max(1, int(round(window_s * frequency_hz)))
    separation = max(1, int(round(minimum_separation_s * frequency_hz)))

    dz = z[window:] - z[:-window]
    candidates = np.argsort(dz)[::-1]

    selected: list[int] = []

    for frame in candidates:
        frame = int(frame)

        if dz[frame] < minimum_rise_m:
            break

        if any(abs(frame - old) < separation for old in selected):
            continue

        selected.append(frame)

    return sorted(selected)


def segment_first_three_steps(
    motion: ReferenceMotion,
) -> ReferenceSegmentation:
    """Identify the first three useful stair-rise events.

    The reference alternates feet while ascending.  We use ankle vertical
    trajectories only for segmentation.  No motion values are modified here.
    """
    left = motion.left_ankle[:, 2]
    right = motion.right_ankle[:, 2]

    left_peaks = _select_distinct_peaks(
        left,
        frequency_hz=motion.frequency_hz,
    )
    right_peaks = _select_distinct_peaks(
        right,
        frequency_hz=motion.frequency_hz,
    )

    window = int(round(0.4 * motion.frequency_hz))

    events = []

    for foot, z, peaks in (
        ("left", left, left_peaks),
        ("right", right, right_peaks),
    ):
        for frame in peaks:
            events.append(
                (
                    frame,
                    foot,
                    float(z[frame]),
                    float(z[min(frame + window, len(z) - 1)]),
                )
            )

    events.sort(key=lambda item: item[0])

    # Ignore the small early pre-stair lift and keep the first three
    # substantial stair ascent events.
    substantial = [
        event
        for event in events
        if (event[3] - event[2]) >= 0.20
    ]

    if len(substantial) < 3:
        raise ValueError(
            f"expected at least 3 stair transitions, found {len(substantial)}"
        )

    first_three = substantial[:3]

    transitions = []

    for index, (frame, foot, z0, z1) in enumerate(first_three, start=1):
        end_frame = min(frame + window, motion.frame_count - 1)

        transitions.append(
            StairTransition(
                name=f"step_{index}",
                frame_start=frame,
                frame_end=end_frame,
                time_start_s=frame / motion.frequency_hz,
                time_end_s=end_frame / motion.frequency_hz,
                foot=foot,
                start_height_m=z0,
                end_height_m=z1,
            )
        )

    # Leave a short settling period after the third detected transition.
    settle_frames = int(round(0.7 * motion.frequency_hz))
    cut_frame = min(
        transitions[-1].frame_end + settle_frames,
        motion.frame_count - 1,
    )

    return ReferenceSegmentation(
        transitions=tuple(transitions),
        cut_frame=cut_frame,
        cut_time_s=cut_frame / motion.frequency_hz,
    )
