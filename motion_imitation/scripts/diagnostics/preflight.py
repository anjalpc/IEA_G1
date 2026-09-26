#!/usr/bin/env python3
"""CPU-only preflight for the canonical G1 stair imitation reference."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


MOTION_IMITATION_ROOT = Path(__file__).resolve().parents[2]
if str(MOTION_IMITATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_IMITATION_ROOT))

from g1_stair_imitation.paths import REFERENCE_FILE  # noqa: E402
from g1_stair_imitation.reference import (  # noqa: E402
    load_g1_stair_imitation_reference,
)


FIRST_RISER_X_M = 1.0
RISER_M = 0.20
TREAD_M = 0.25
SPHERE_RADIUS_M = 0.005


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ref = load_g1_stair_imitation_reference(REFERENCE_FILE)

    print("=" * 78)
    print(" G1 IMITATION PREFLIGHT")
    print("=" * 78)
    print("reference:", ref.path)
    print("sha256   :", sha256(ref.path))
    print("frames   :", ref.frame_count)
    print(
        "source   :",
        f"{ref.source_frame_start}->{ref.source_frame_end}",
    )
    print("q_ref    :", ref.q_ref.shape)
    print("sole     :", ref.left_sole_points.shape)
    print("taskspace:", int(np.sum(ref.taskspace_valid)), "/", ref.frame_count)
    print("pose DOF :", int(np.sum(ref.pose_mask > 0.0)), "/ 29")

    max_support_z_error = 0.0
    minimum_tread_margin = float("inf")
    support_samples = 0

    for index in range(1, ref.frame_count):
        source_frame = ref.source_frame_start + index

        for side in ("left", "right"):
            if side == "left":
                persistent = bool(ref.left_contact[index])
                step = int(ref.left_step[index])
                sole = ref.left_sole_points[index]
            else:
                persistent = bool(ref.right_contact[index])
                step = int(ref.right_step[index])
                sole = ref.right_sole_points[index]

            if not persistent:
                continue

            if step < 0:
                raise RuntimeError(
                    f"frame {source_frame} {side}: contact=True but step is unset"
                )

            expected_z = step * RISER_M
            bottom_z = sole[:, 2] - SPHERE_RADIUS_M
            z_error = float(np.max(np.abs(bottom_z - expected_z)))
            max_support_z_error = max(max_support_z_error, z_error)
            support_samples += 1

            if z_error > 0.002:
                raise RuntimeError(
                    f"frame {source_frame} {side}: support Z error "
                    f"{z_error * 1000.0:.3f} mm"
                )

            if step > 0:
                course_x = sole[:, 0] + FIRST_RISER_X_M
                tread_start = FIRST_RISER_X_M + (step - 1) * TREAD_M
                tread_end = tread_start + TREAD_M
                margin = min(
                    float(np.min(course_x) - tread_start),
                    float(tread_end - np.max(course_x)),
                )
                minimum_tread_margin = min(minimum_tread_margin, margin)
                if margin < -1.0e-6:
                    raise RuntimeError(
                        f"frame {source_frame} {side}: support sole leaves tread; "
                        f"margin={margin * 1000.0:.3f} mm"
                    )

    print()
    print("SUPPORT CONTRACT")
    print("support samples checked:", support_samples)
    print(f"max support Z error    : {max_support_z_error * 1000.0:.6f} mm")
    print(f"minimum tread margin   : {minimum_tread_margin * 1000.0:.3f} mm")

    print()
    print("CRITICAL CONTACT STATES")
    for source_frame in (152, 179, 207, 239, 269, 277):
        index = ref.reference_index_for_source_frame(source_frame)
        print(
            f"{source_frame}: "
            f"L={bool(ref.left_contact[index])}/{int(ref.left_step[index])} "
            f"R={bool(ref.right_contact[index])}/{int(ref.right_step[index])}"
        )

    print()
    print("=" * 78)
    print(" IMITATION PREFLIGHT PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
