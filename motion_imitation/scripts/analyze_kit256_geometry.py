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
            "rate": float(reader.point_rate),
            "labels": labels,
            "xyz": np.asarray(frames),
            "valid": np.asarray(valid),
        }


def marker(data, name):
    i = data["labels"].index(name)

    if not np.all(data["valid"][:, i]):
        raise RuntimeError(
            f"Marker {name} contains invalid samples"
        )

    return data["xyz"][:, i, :]


print("========================================")
print(" KIT256 GEOMETRY / MOTION ANALYSIS")
print("========================================")

motion = load_c3d(MOTION)
stairs = load_c3d(STAIRS)

fps = motion["rate"]

print("Motion FPS   :", fps)
print("Motion frames:", len(motion["xyz"]))
print(
    "Duration     :",
    (len(motion["xyz"]) - 1) / fps,
    "s",
)

# ---------------------------------------------------------
# Stair geometry
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
widths = []

print()
print("===== STAIR EDGE PAIRS =====")

for level, (a, b) in enumerate(pairs, start=1):

    ia = stairs["labels"].index(a)
    ib = stairs["labels"].index(b)

    ma = np.median(
        stairs["xyz"][stairs["valid"][:, ia], ia, :],
        axis=0,
    )

    mb = np.median(
        stairs["xyz"][stairs["valid"][:, ib], ib, :],
        axis=0,
    )

    center = 0.5 * (ma + mb)
    width = np.linalg.norm(
        ma[:2] - mb[:2]
    )

    centers.append(center)
    widths.append(width)

    print(
        f"{level}: "
        f"{a:8s}<->{b:8s} "
        f"center=({center[0]:8.2f}, "
        f"{center[1]:8.2f}, "
        f"{center[2]:8.2f}) mm "
        f"width={width:8.2f} mm"
    )

centers = np.asarray(centers)
widths = np.asarray(widths)

# Use the first five stair noses to define ascent direction.
forward_xy = (
    centers[4, :2]
    - centers[0, :2]
)

forward_xy /= np.linalg.norm(
    forward_xy
)

lateral_xy = np.array([
    -forward_xy[1],
    forward_xy[0],
])

origin_xy = centers[0, :2]

s = (
    centers[:, :2] - origin_xy
) @ forward_xy

z = centers[:, 2]

# Rise between consecutive stair levels.
rise = np.diff(z[:5])

# Estimate floor level from first tread using typical measured rise.
mean_rise = float(np.mean(rise))
floor_z = z[0] - mean_rise

# Tread depth between consecutive stair noses.
tread = np.diff(s[:5])

print()
print("===== STAIR FRAME =====")
print(
    "Forward XY :",
    np.round(forward_xy, 6),
)
print(
    "Lateral XY :",
    np.round(lateral_xy, 6),
)

print()
print("===== STAIR GEOMETRY =====")

for i in range(5):
    print(
        f"step {i+1}: "
        f"s={s[i]:8.2f} mm "
        f"z={z[i]-floor_z:8.2f} mm"
    )

print()
print(
    "Mean riser :",
    f"{mean_rise:.2f} mm"
)
print(
    "Risers     :",
    np.round(rise, 2).tolist()
)

print(
    "Mean tread :",
    f"{np.mean(tread):.2f} mm"
)
print(
    "Treads     :",
    np.round(tread, 2).tolist()
)

print(
    "Mean width :",
    f"{np.mean(widths):.2f} mm"
)

# ---------------------------------------------------------
# Human root and feet
# ---------------------------------------------------------

LASI = marker(motion, "LASI")
RASI = marker(motion, "RASI")
LHIP = marker(motion, "LHIP")
RHIP = marker(motion, "RHIP")

pelvis = (
    LASI + RASI + LHIP + RHIP
) / 4.0


def foot_center(side):
    names = [
        f"{side}HEE",
        f"{side}TOE",
        f"{side}MT1",
        f"{side}MT5",
    ]

    pts = [
        marker(motion, x)
        for x in names
    ]

    return np.mean(
        np.stack(pts, axis=0),
        axis=0,
    )


left_foot = foot_center("L")
right_foot = foot_center("R")


def to_stair_frame(points):
    xy = points[:, :2] - origin_xy

    forward = xy @ forward_xy
    lateral = xy @ lateral_xy
    height = points[:, 2] - floor_z

    return np.column_stack([
        forward,
        lateral,
        height,
    ])


pelvis_s = to_stair_frame(pelvis)
left_s = to_stair_frame(left_foot)
right_s = to_stair_frame(right_foot)


def speed(points):
    v = np.zeros(len(points))

    v[1:] = (
        np.linalg.norm(
            np.diff(points, axis=0),
            axis=1,
        )
        * fps
    )

    return v


left_speed = speed(left_foot)
right_speed = speed(right_foot)


print()
print("===== HUMAN MOTION RANGE =====")

print(
    "Pelvis forward :",
    f"{pelvis_s[:,0].min():.1f}",
    "->",
    f"{pelvis_s[:,0].max():.1f}",
    "mm"
)

print(
    "Pelvis height  :",
    f"{pelvis_s[:,2].min():.1f}",
    "->",
    f"{pelvis_s[:,2].max():.1f}",
    "mm"
)

print()
print("===== INITIAL STATE =====")

print(
    "Pelvis:",
    np.round(pelvis_s[0], 2)
)

print(
    "Left foot :",
    np.round(left_s[0], 2)
)

print(
    "Right foot:",
    np.round(right_s[0], 2)
)

print()
print("===== FINAL STATE =====")

print(
    "Pelvis:",
    np.round(pelvis_s[-1], 2)
)

print(
    "Left foot :",
    np.round(left_s[-1], 2),
    f"speed={left_speed[-1]:.2f} mm/s"
)

print(
    "Right foot:",
    np.round(right_s[-1], 2),
    f"speed={right_speed[-1]:.2f} mm/s"
)

# ---------------------------------------------------------
# Final standing phase
# ---------------------------------------------------------

LOW_SPEED = 80.0  # mm/s

both_still = (
    (left_speed < LOW_SPEED)
    &
    (right_speed < LOW_SPEED)
)

# Find continuous low-speed suffix.
start = len(both_still) - 1

while (
    start > 0
    and both_still[start - 1]
):
    start -= 1

print()
print("===== FINAL STABLE PHASE =====")

print(
    "Low-speed threshold:",
    LOW_SPEED,
    "mm/s"
)

print(
    "Start frame:",
    start
)

print(
    "Start time :",
    f"{start / fps:.3f} s"
)

print(
    "End frame  :",
    len(both_still) - 1
)

print(
    "Duration   :",
    f"{(len(both_still)-start) / fps:.3f} s"
)

print()
print("===== LAST 20 FRAMES =====")

for i in range(
    max(0, len(left_speed)-20),
    len(left_speed),
):
    print(
        f"{i:03d} "
        f"t={i/fps:6.3f} "
        f"L_v={left_speed[i]:8.2f} "
        f"R_v={right_speed[i]:8.2f} "
        f"L_z={left_s[i,2]:8.2f} "
        f"R_z={right_s[i,2]:8.2f}"
    )

print()
print("========================================")
print(" KIT256 GEOMETRY ANALYSIS COMPLETE")
print("========================================")
