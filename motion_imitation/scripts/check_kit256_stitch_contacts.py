#!/usr/bin/env python3

from pathlib import Path

import c3d
import numpy as np


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")

NPZ = (
    ROOT
    / "outputs/kit256_3step_stitch/"
      "WSUF01_3step_stitched_markers.npz"
)

STAIRS = (
    ROOT
    / "01_source/KIT/motion_256/STAIRS.c3d"
)


def clean_label(name):
    name = str(name).strip()

    if ":" in name:
        name = name.split(":")[-1]

    return name.strip()


def load_stairs(path):

    with open(path, "rb") as f:
        reader = c3d.Reader(f)

        labels = [
            clean_label(x)
            for x in reader.point_labels
        ]

        xyz = []
        valid = []

        for _, points, _ in reader.read_frames(copy=True):

            points = np.asarray(points)

            p = np.asarray(
                points[:, :3],
                dtype=float,
            )

            ok = np.isfinite(p).all(axis=1)

            if points.shape[1] >= 4:
                residual = np.asarray(
                    points[:, 3],
                    dtype=float,
                )

                ok &= residual >= 0.0

            xyz.append(p)
            valid.append(ok)

    return (
        labels,
        np.asarray(xyz),
        np.asarray(valid),
    )


def marker_ids(labels, names):

    return [
        labels.index(name)
        for name in names
    ]


def foot_center(
    xyz,
    labels,
    side,
):

    ids = marker_ids(
        labels,
        [
            f"{side}HEE",
            f"{side}TOE",
            f"{side}MT1",
            f"{side}MT5",
        ],
    )

    return np.mean(
        xyz[:, ids, :],
        axis=1,
    )


print("========================================")
print(" KIT256 STITCH CONTACT CHECK")
print("========================================")


# ---------------------------------------------------------
# Load stitched marker motion
# ---------------------------------------------------------

d = np.load(
    NPZ,
    allow_pickle=False,
)

xyz = np.asarray(
    d["xyz_mm"],
    dtype=float,
)

labels = [
    str(x)
    for x in d["labels"].tolist()
]

fps = float(d["fps"])


print("FPS   :", fps)
print("Frames:", len(xyz))

assert xyz.shape[0] == 555
assert xyz.shape[1] == len(labels)


# ---------------------------------------------------------
# Reconstruct KIT stair frame
# ---------------------------------------------------------

stair_labels, stair_xyz, stair_valid = (
    load_stairs(STAIRS)
)

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

    ia = stair_labels.index(a)
    ib = stair_labels.index(b)

    pa = np.median(
        stair_xyz[
            stair_valid[:, ia],
            ia,
            :
        ],
        axis=0,
    )

    pb = np.median(
        stair_xyz[
            stair_valid[:, ib],
            ib,
            :
        ],
        axis=0,
    )

    centers.append(
        0.5 * (pa + pb)
    )

centers = np.asarray(centers)

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

s_nose = (
    centers[:, :2]
    - origin_xy
) @ forward_xy

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

    xy = (
        points[:, :2]
        - origin_xy
    )

    return np.column_stack([
        xy @ forward_xy,
        xy @ lateral_xy,
        points[:, 2] - floor_z,
    ])


# ---------------------------------------------------------
# Feet
# ---------------------------------------------------------

left_world = foot_center(
    xyz,
    labels,
    "L",
)

right_world = foot_center(
    xyz,
    labels,
    "R",
)

left = stair_frame(
    left_world
)

right = stair_frame(
    right_world
)


def speed(points):

    out = np.zeros(
        len(points)
    )

    out[1:] = (
        np.linalg.norm(
            np.diff(
                points,
                axis=0,
            ),
            axis=1,
        )
        * fps
    )

    return out


left_speed = speed(
    left_world
)

right_speed = speed(
    right_world
)


# ---------------------------------------------------------
# Estimate marker height above floor from initial stance
# ---------------------------------------------------------

left_floor_offset = float(
    np.median(
        left[:150, 2][
            left_speed[:150] < 120.0
        ]
    )
)

right_floor_offset = float(
    np.median(
        right[:150, 2][
            right_speed[:150] < 120.0
        ]
    )
)


print()
print("===== STAIR GEOMETRY =====")

print(
    "Mean rise:",
    f"{mean_rise:.2f} mm"
)

print(
    "Step noses:",
    np.round(
        s_nose[:5],
        2,
    ).tolist()
)

print(
    "L floor marker offset:",
    f"{left_floor_offset:.2f} mm"
)

print(
    "R floor marker offset:",
    f"{right_floor_offset:.2f} mm"
)


# ---------------------------------------------------------
# Right support foot through splice
#
# Original:
#   right step-3 contact begins at source frame 477
#
# Output:
#   original 0..474
#   blend    475..495
#   terminal 636..694 -> output 496..554
#
# Therefore the right support foot should remain planted
# continuously from about output 477 to the end.
# ---------------------------------------------------------

R_START = 477
R_END = 554

r = right[
    R_START:R_END + 1
]

r_ref = np.median(
    r,
    axis=0,
)

r_error = np.linalg.norm(
    r - r_ref,
    axis=1,
)

r_forward_range = np.ptp(
    r[:, 0]
)

r_lateral_range = np.ptp(
    r[:, 1]
)

r_vertical_range = np.ptp(
    r[:, 2]
)


print()
print("========================================")
print(" RIGHT SUPPORT FOOT THROUGH STITCH")
print("========================================")

print(
    "Frames:",
    f"{R_START}->{R_END}"
)

print(
    "Duration:",
    f"{(R_END-R_START+1)/fps:.3f} s"
)

print(
    "Reference [s,y,z]:",
    np.round(
        r_ref,
        2,
    )
)

print(
    "Forward range :",
    f"{r_forward_range:.2f} mm"
)

print(
    "Lateral range :",
    f"{r_lateral_range:.2f} mm"
)

print(
    "Vertical range:",
    f"{r_vertical_range:.2f} mm"
)

print(
    "Max 3D drift  :",
    f"{r_error.max():.2f} mm"
)

print(
    "Median speed  :",
    f"{np.median(right_speed[R_START:R_END+1]):.2f} mm/s"
)

print(
    "Max speed     :",
    f"{np.max(right_speed[R_START:R_END+1]):.2f} mm/s"
)


print()
print("===== RIGHT FOOT AROUND BLEND =====")

for i in range(
    470,
    506,
    2,
):
    print(
        f"{i:03d} "
        f"t={i/fps:5.2f} "
        f"s={right[i,0]:8.2f} "
        f"y={right[i,1]:8.2f} "
        f"z={right[i,2]:8.2f} "
        f"v={right_speed[i]:8.2f}"
    )


# ---------------------------------------------------------
# Final left-foot landing
#
# Terminal source L top contact:
#   678..694
#
# Terminal output mapping:
#   source 636 -> output 496
#
# therefore:
#   678 -> 538
#   694 -> 554
# ---------------------------------------------------------

L_START = 538
L_END = 554

l = left[
    L_START:L_END + 1
]

l_ref = np.median(
    l,
    axis=0,
)

l_error = np.linalg.norm(
    l - l_ref,
    axis=1,
)


print()
print("========================================")
print(" FINAL LEFT-FOOT CONTACT")
print("========================================")

print(
    "Frames:",
    f"{L_START}->{L_END}"
)

print(
    "Forward range :",
    f"{np.ptp(l[:,0]):.2f} mm"
)

print(
    "Lateral range :",
    f"{np.ptp(l[:,1]):.2f} mm"
)

print(
    "Vertical range:",
    f"{np.ptp(l[:,2]):.2f} mm"
)

print(
    "Max 3D drift  :",
    f"{l_error.max():.2f} mm"
)

print(
    "Median speed  :",
    f"{np.median(left_speed[L_START:L_END+1]):.2f} mm/s"
)


# ---------------------------------------------------------
# Final feet relative to physical step 3
# ---------------------------------------------------------

step3_front = float(
    s_nose[2]
)

step3_back = float(
    s_nose[3]
)

step3_top = float(
    centers[2, 2]
    - floor_z
)

lf = left[-1]
rf = right[-1]


print()
print("========================================")
print(" FINAL POSITION ON STEP 3")
print("========================================")

print(
    "Step-3 tread s:",
    f"{step3_front:.2f}",
    "->",
    f"{step3_back:.2f}",
    "mm"
)

print(
    "Step-3 top z:",
    f"{step3_top:.2f} mm"
)

print()

print(
    "Left final  [s,y,z]:",
    np.round(
        lf,
        2,
    )
)

print(
    "Right final [s,y,z]:",
    np.round(
        rf,
        2,
    )
)

left_inside = (
    step3_front
    <= lf[0]
    <= step3_back
)

right_inside = (
    step3_front
    <= rf[0]
    <= step3_back
)

print()
print(
    "Left inside step-3 tread :",
    "PASS"
    if left_inside
    else "FAIL"
)

print(
    "Right inside step-3 tread:",
    "PASS"
    if right_inside
    else "FAIL"
)


# Expected marker-center height if the foot geometry
# were identical to initial flat stance.
left_expected_z = (
    step3_top
    + left_floor_offset
)

right_expected_z = (
    step3_top
    + right_floor_offset
)

print()
print(
    "Left marker-height error :",
    f"{lf[2]-left_expected_z:+.2f} mm"
)

print(
    "Right marker-height error:",
    f"{rf[2]-right_expected_z:+.2f} mm"
)


print()
print("========================================")
print(" CONTACT CHECK COMPLETE")
print("========================================")
