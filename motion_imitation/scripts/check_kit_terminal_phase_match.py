#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import c3d


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")
C3D = ROOT / "01_source/KIT/motion_256/WSUF01.c3d"


MARKERS = [
    "LASI", "RASI",
    "LHIP", "RHIP",

    "LKNE", "RKNE",
    "LANK", "RANK",
    "LTOE", "RTOE",
    "LHEE", "RHEE",

    "LSHO", "RSHO",
    "LAEL", "RAEL",
    "LWTS", "RWTS",

    "C7", "STRN",
]


def clean_label(name):
    name = str(name).strip()
    if ":" in name:
        name = name.split(":")[-1]
    return name.strip()


def load_c3d(path):
    with open(path, "rb") as f:
        reader = c3d.Reader(f)

        labels = [
            clean_label(x)
            for x in reader.point_labels
        ]

        frames = []

        for _, points, _ in reader.read_frames(copy=True):
            frames.append(
                np.asarray(points[:, :3], dtype=float)
            )

    return labels, np.asarray(frames)


labels, xyz = load_c3d(C3D)

indices = [
    labels.index(name)
    for name in MARKERS
]

poses = xyz[:, indices, :]


def pelvis_center(frame):
    ids = [
        MARKERS.index("LASI"),
        MARKERS.index("RASI"),
        MARKERS.index("LHIP"),
        MARKERS.index("RHIP"),
    ]

    return np.mean(
        frame[ids],
        axis=0,
    )


def normalize_pose(frame):

    frame = frame.copy()

    pelvis = pelvis_center(frame)
    frame -= pelvis

    # Body lateral direction from left/right ASIS.
    lasi = frame[
        MARKERS.index("LASI")
    ]

    rasi = frame[
        MARKERS.index("RASI")
    ]

    lateral = lasi[:2] - rasi[:2]
    lateral /= np.linalg.norm(lateral)

    # Horizontal forward direction.
    forward = np.array([
        -lateral[1],
        lateral[0],
    ])

    # Construct body-local basis:
    # x = forward
    # y = lateral
    # z = vertical
    R = np.array([
        [forward[0], lateral[0], 0.0],
        [forward[1], lateral[1], 0.0],
        [0.0,        0.0,        1.0],
    ])

    return frame @ R


norm = np.asarray([
    normalize_pose(frame)
    for frame in poses
])


# Search near R->step3 landing and R->step5 landing.
range_step3 = range(470, 486)
range_step5 = range(608, 626)

results = []

for a in range_step3:
    for b in range_step5:

        diff = norm[a] - norm[b]

        per_marker = np.linalg.norm(
            diff,
            axis=1,
        )

        rms = np.sqrt(
            np.mean(
                np.sum(diff * diff, axis=1)
            )
        )

        results.append(
            (
                rms,
                a,
                b,
                per_marker.max(),
                per_marker.mean(),
            )
        )


results.sort()

print("========================================")
print(" KIT TERMINAL PHASE COMPATIBILITY")
print("========================================")

print()
print("Expected contact anchors:")
print("R step3 begins : frame 477")
print("R step5 begins : frame 613")

print()
print("===== BEST FRAME PAIRS =====")

for rms, a, b, mx, mean in results[:15]:
    print(
        f"step3={a:03d} "
        f"step5={b:03d} "
        f"RMS={rms:7.2f} mm "
        f"mean={mean:7.2f} mm "
        f"max={mx:7.2f} mm"
    )


# Exact nominal pair
A = 477
B = 613

diff = norm[A] - norm[B]
err = np.linalg.norm(diff, axis=1)

rms = np.sqrt(
    np.mean(
        np.sum(diff * diff, axis=1)
    )
)

print()
print("===== NOMINAL 477 vs 613 =====")

print(
    f"RMS pose difference: {rms:.2f} mm"
)

for name, value in sorted(
    zip(MARKERS, err),
    key=lambda x: x[1],
    reverse=True,
):
    print(
        f"{name:6s} {value:7.2f} mm"
    )


print()
print("========================================")
print(" CHECK COMPLETE")
print("========================================")
