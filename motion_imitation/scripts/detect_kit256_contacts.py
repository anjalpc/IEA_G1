#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import c3d


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")
KIT = ROOT / "01_source/KIT/motion_256"

MOTION = KIT / "WSUF01.c3d"
STAIRS = KIT / "STAIRS.c3d"


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
        valid = []

        for _, points, _ in reader.read_frames(copy=True):
            points = np.asarray(points)

            xyz = np.asarray(
                points[:, :3],
                dtype=float,
            )

            ok = np.isfinite(xyz).all(axis=1)

            if points.shape[1] >= 4:
                residual = np.asarray(
                    points[:, 3],
                    dtype=float,
                )
                ok &= residual >= 0.0

            frames.append(xyz)
            valid.append(ok)

        return {
            "fps": float(reader.point_rate),
            "labels": labels,
            "xyz": np.asarray(frames),
            "valid": np.asarray(valid),
        }


def marker(data, name):
    i = data["labels"].index(name)

    if not np.all(data["valid"][:, i]):
        raise RuntimeError(
            f"Invalid samples in marker {name}"
        )

    return data["xyz"][:, i, :]


def foot_center(data, side):
    names = [
        f"{side}HEE",
        f"{side}TOE",
        f"{side}MT1",
        f"{side}MT5",
    ]

    return np.mean(
        np.stack(
            [marker(data, n) for n in names],
            axis=0,
        ),
        axis=0,
    )


def speed(x, fps):
    out = np.zeros(len(x))

    out[1:] = (
        np.linalg.norm(
            np.diff(x, axis=0),
            axis=1,
        )
        * fps
    )

    return out


def vertical_speed(x, fps):
    out = np.zeros(len(x))

    out[1:] = (
        np.abs(np.diff(x[:, 2]))
        * fps
    )

    return out


def contiguous_regions(mask):
    regions = []

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        elif not value and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append(
            (start, len(mask) - 1)
        )

    return regions


def merge_regions(regions, max_gap=4):
    if not regions:
        return []

    merged = [list(regions[0])]

    for start, end in regions[1:]:

        if start - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [
        tuple(x)
        for x in merged
    ]


motion = load_c3d(MOTION)
stairs = load_c3d(STAIRS)

fps = motion["fps"]

print("========================================")
print(" KIT256 FOOT CONTACT DETECTION")
print("========================================")

print("FPS   :", fps)
print("Frames:", len(motion["xyz"]))


# ---------------------------------------------------------
# Stair coordinate frame
# ---------------------------------------------------------

pairs = [
    ("Marker1", "Marker12"),
    ("Marker2", "Marker11"),
    ("Marker3", "Marker10"),
    ("Marker4", "Marker9"),
    ("Marker5", "Marker8"),
    ("Marker6", "Marker7"),
]

centers = []

for a, b in pairs:

    ia = stairs["labels"].index(a)
    ib = stairs["labels"].index(b)

    pa = np.median(
        stairs["xyz"][
            stairs["valid"][:, ia],
            ia,
            :
        ],
        axis=0,
    )

    pb = np.median(
        stairs["xyz"][
            stairs["valid"][:, ib],
            ib,
            :
        ],
        axis=0,
    )

    centers.append(
        0.5 * (pa + pb)
    )

centers = np.asarray(centers)

forward = (
    centers[4, :2]
    - centers[0, :2]
)

forward /= np.linalg.norm(forward)

lateral = np.array([
    -forward[1],
    forward[0],
])

origin_xy = centers[0, :2]

rise = np.diff(
    centers[:5, 2]
)

mean_rise = float(
    np.mean(rise)
)

floor_z = (
    centers[0, 2]
    - mean_rise
)


def stair_frame(points):
    xy = points[:, :2] - origin_xy

    return np.column_stack([
        xy @ forward,
        xy @ lateral,
        points[:, 2] - floor_z,
    ])


left_world = foot_center(
    motion,
    "L",
)

right_world = foot_center(
    motion,
    "R",
)

left = stair_frame(left_world)
right = stair_frame(right_world)


left_speed = speed(
    left_world,
    fps,
)

right_speed = speed(
    right_world,
    fps,
)

left_vz = vertical_speed(
    left_world,
    fps,
)

right_vz = vertical_speed(
    right_world,
    fps,
)


# ---------------------------------------------------------
# Estimate marker offset above support surface.
#
# Use first 100 frames and find the low-speed samples.
# This means we do not assume the foot-center marker is
# exactly at physical floor height.
# ---------------------------------------------------------

def estimate_floor_offset(
    foot,
    spd,
    vz,
):
    mask = (
        (spd[:150] < 120.0)
        &
        (vz[:150] < 60.0)
    )

    vals = foot[:150, 2][mask]

    if len(vals) < 5:
        raise RuntimeError(
            "Could not estimate floor foot height"
        )

    return float(
        np.median(vals)
    )


left_floor_height = estimate_floor_offset(
    left,
    left_speed,
    left_vz,
)

right_floor_height = estimate_floor_offset(
    right,
    right_speed,
    right_vz,
)


print()
print("===== FOOT REFERENCE HEIGHT =====")

print(
    "Left floor foot-center :",
    f"{left_floor_height:.2f} mm"
)

print(
    "Right floor foot-center:",
    f"{right_floor_height:.2f} mm"
)

print(
    "Mean stair rise        :",
    f"{mean_rise:.2f} mm"
)


# ---------------------------------------------------------
# Contact candidates
# ---------------------------------------------------------

SPEED_LIMIT = 180.0
VERTICAL_SPEED_LIMIT = 80.0

left_contact = (
    (left_speed < SPEED_LIMIT)
    &
    (left_vz < VERTICAL_SPEED_LIMIT)
)

right_contact = (
    (right_speed < SPEED_LIMIT)
    &
    (right_vz < VERTICAL_SPEED_LIMIT)
)


def extract_contacts(
    side,
    foot,
    spd,
    vz,
    contact,
    floor_height,
):

    regions = contiguous_regions(
        contact
    )

    regions = merge_regions(
        regions,
        max_gap=4,
    )

    # Require at least 80 ms.
    regions = [
        (a, b)
        for a, b in regions
        if (b - a + 1) >= 8
    ]

    print()
    print(
        f"===== {side} CONTACT PLATEAUS ====="
    )

    output = []

    for a, b in regions:

        med = np.median(
            foot[a:b+1],
            axis=0,
        )

        relative_height = (
            med[2] - floor_height
        )

        level = int(
            np.rint(
                relative_height
                / mean_rise
            )
        )

        level = max(0, level)

        output.append(
            {
                "start": a,
                "end": b,
                "level": level,
                "s": med[0],
                "y": med[1],
                "z": med[2],
            }
        )

        print(
            f"{a:03d}->{b:03d} "
            f"t={a/fps:5.2f}->{b/fps:5.2f}s "
            f"dur={(b-a+1)/fps:4.2f}s "
            f"level={level} "
            f"s={med[0]:8.1f} "
            f"y={med[1]:7.1f} "
            f"z={med[2]:7.1f} "
            f"speed_med={np.median(spd[a:b+1]):6.1f}"
        )

    return output


left_events = extract_contacts(
    "LEFT",
    left,
    left_speed,
    left_vz,
    left_contact,
    left_floor_height,
)

right_events = extract_contacts(
    "RIGHT",
    right,
    right_speed,
    right_vz,
    right_contact,
    right_floor_height,
)


# ---------------------------------------------------------
# Combine stair contacts only
# ---------------------------------------------------------

events = []

for e in left_events:
    if e["level"] >= 1:
        events.append(
            (
                e["start"],
                "L",
                e["level"],
                e["end"],
                e["s"],
                e["z"],
            )
        )

for e in right_events:
    if e["level"] >= 1:
        events.append(
            (
                e["start"],
                "R",
                e["level"],
                e["end"],
                e["s"],
                e["z"],
            )
        )

events.sort()


print()
print("========================================")
print(" ASCENDING CONTACT SEQUENCE")
print("========================================")

for start, side, level, end, s, z in events:

    print(
        f"{side} "
        f"level={level} "
        f"frame={start:03d}->{end:03d} "
        f"time={start/fps:5.2f}s "
        f"s={s:8.1f} "
        f"z={z:7.1f}"
    )


print()
print("========================================")
print(" CONTACT DETECTION COMPLETE")
print("========================================")
