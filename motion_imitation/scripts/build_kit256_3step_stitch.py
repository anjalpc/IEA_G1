#!/usr/bin/env python3

from pathlib import Path
import math

import c3d
import numpy as np


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")

SRC = (
    ROOT
    / "01_source/KIT/motion_256/WSUF01.c3d"
)

OUT_DIR = (
    ROOT
    / "outputs/kit256_3step_stitch"
)

OUT = (
    OUT_DIR
    / "WSUF01_3step_stitched_markers.npz"
)


# ------------------------------------------------------------
# Stitch contract
# ------------------------------------------------------------

SRC_BLEND_START = 475
SRC_BLEND_END = 495

TERM_BLEND_START = 615
TERM_BLEND_END = 635

SRC_ANCHOR = 485
TERM_ANCHOR = 625

TERM_CONTINUE = 636

FPS_EXPECTED = 100.0


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

        xyz = []
        valid = []

        fps = float(reader.point_rate)

        for _, points, _ in reader.read_frames(
            copy=True
        ):
            points = np.asarray(points)

            p = np.asarray(
                points[:, :3],
                dtype=float,
            )

            ok = np.isfinite(
                p
            ).all(axis=1)

            if points.shape[1] >= 4:
                residual = np.asarray(
                    points[:, 3],
                    dtype=float,
                )

                ok &= residual >= 0.0

            xyz.append(p)
            valid.append(ok)

    return (
        fps,
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
    frame,
    side,
):
    names = [
        f"{side}HEE",
        f"{side}TOE",
        f"{side}MT1",
        f"{side}MT5",
    ]

    ids = marker_ids(
        labels,
        names,
    )

    return np.mean(
        xyz[frame, ids, :],
        axis=0,
    )


def body_lateral(
    xyz,
    labels,
    frame,
):
    lasi = xyz[
        frame,
        labels.index("LASI"),
        :
    ]

    rasi = xyz[
        frame,
        labels.index("RASI"),
        :
    ]

    v = lasi[:2] - rasi[:2]

    return v / np.linalg.norm(v)


def rotation_z(theta):

    c = math.cos(theta)
    s = math.sin(theta)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


print("========================================")
print(" KIT256 3-STEP MOTION STITCH")
print("========================================")

fps, labels, xyz, valid = load_c3d(
    SRC
)

print("Source :", SRC)
print("FPS    :", fps)
print("Frames :", len(xyz))
print("Markers:", len(labels))

assert abs(
    fps - FPS_EXPECTED
) < 1e-9

assert len(xyz) == 695

# All markers needed for the actual retarget are valid.
required = [
    "LASI", "RASI",
    "LHIP", "RHIP",

    "LKNE", "RKNE",
    "LANK", "RANK",

    "LTOE", "RTOE",
    "LHEE", "RHEE",

    "LMT1", "LMT5",
    "RMT1", "RMT5",

    "LSHO", "RSHO",
    "LAEL", "RAEL",
    "LWTS", "RWTS",

    "C7", "STRN",
]

for name in required:

    idx = labels.index(name)

    if not np.all(
        valid[:, idx]
    ):
        raise RuntimeError(
            f"Required marker has invalid "
            f"samples: {name}"
        )

print("[01] required marker validity: PASS")


# ------------------------------------------------------------
# Determine terminal rigid-body transform
# ------------------------------------------------------------

lat_src = body_lateral(
    xyz,
    labels,
    SRC_ANCHOR,
)

lat_term = body_lateral(
    xyz,
    labels,
    TERM_ANCHOR,
)

cross = (
    lat_term[0] * lat_src[1]
    - lat_term[1] * lat_src[0]
)

dot = np.dot(
    lat_term,
    lat_src,
)

theta = math.atan2(
    cross,
    dot,
)

R = rotation_z(theta)


src_support = foot_center(
    xyz,
    labels,
    SRC_ANCHOR,
    "R",
)

term_support = foot_center(
    xyz,
    labels,
    TERM_ANCHOR,
    "R",
)

term_support_rot = (
    R @ term_support
)

translation = (
    src_support
    - term_support_rot
)


print()
print("===== TERMINAL TRANSFORM =====")

print(
    "Yaw correction:",
    f"{math.degrees(theta):.4f} deg"
)

print(
    "Translation:",
    np.round(
        translation,
        3,
    ),
    "mm",
)

print(
    "Translation magnitude:",
    f"{np.linalg.norm(translation):.2f} mm"
)


def transform(points):
    return (
        points @ R.T
        + translation
    )


terminal_xyz = transform(
    xyz
)


# Right stance foot must now coincide at the anchor.
mapped_support = foot_center(
    terminal_xyz,
    labels,
    TERM_ANCHOR,
    "R",
)

support_error = np.linalg.norm(
    mapped_support
    - src_support
)

print(
    "R-foot anchor error:",
    f"{support_error:.6f} mm"
)

assert support_error < 1e-6

print("[02] support-foot alignment: PASS")


# ------------------------------------------------------------
# Check phase similarity after rigid alignment
# ------------------------------------------------------------

body_ids = marker_ids(
    labels,
    required,
)


def pose_rms(
    frame_a,
    frame_b,
):
    d = (
        xyz[frame_a, body_ids, :]
        - terminal_xyz[
            frame_b,
            body_ids,
            :
        ]
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    d * d,
                    axis=1,
                )
            )
        )
    )


print()
print("===== ALIGNED PHASE DIFFERENCE =====")

for a, b in [
    (475, 615),
    (480, 620),
    (485, 625),
    (490, 630),
    (495, 635),
]:
    print(
        f"{a:03d} <-> {b:03d} "
        f"RMS={pose_rms(a,b):7.2f} mm"
    )


# ------------------------------------------------------------
# Construct 21-frame smooth overlap
# ------------------------------------------------------------

n_blend = (
    SRC_BLEND_END
    - SRC_BLEND_START
    + 1
)

assert n_blend == (
    TERM_BLEND_END
    - TERM_BLEND_START
    + 1
)

blend = []

blend_source_frames = []
blend_terminal_frames = []

for k in range(n_blend):

    src_frame = (
        SRC_BLEND_START + k
    )

    term_frame = (
        TERM_BLEND_START + k
    )

    u = (
        k / (n_blend - 1)
    )

    # cubic smoothstep
    w = (
        3.0 * u * u
        - 2.0 * u * u * u
    )

    frame = (
        (1.0 - w)
        * xyz[src_frame]
        +
        w
        * terminal_xyz[term_frame]
    )

    blend.append(frame)

    blend_source_frames.append(
        src_frame
    )

    blend_terminal_frames.append(
        term_frame
    )

blend = np.asarray(blend)


# ------------------------------------------------------------
# Assemble
#
# 0..474               original
# 475..495             blended
# 636..694 transformed terminal
# ------------------------------------------------------------

pre = xyz[
    :SRC_BLEND_START
]

post = terminal_xyz[
    TERM_CONTINUE:
]

stitched = np.concatenate(
    [
        pre,
        blend,
        post,
    ],
    axis=0,
)


print()
print("===== OUTPUT CONTRACT =====")

print(
    "Original pre frames :",
    len(pre)
)

print(
    "Blend frames        :",
    len(blend)
)

print(
    "Terminal frames     :",
    len(post)
)

print(
    "Output frames       :",
    len(stitched)
)

print(
    "Output duration     :",
    f"{(len(stitched)-1)/fps:.3f} s"
)


# ------------------------------------------------------------
# Continuity diagnostics
# ------------------------------------------------------------

frame_delta = np.linalg.norm(
    np.diff(
        stitched[:, body_ids, :],
        axis=0,
    ),
    axis=2,
)

frame_rms = np.sqrt(
    np.mean(
        frame_delta ** 2,
        axis=1,
    )
)

blend_out_start = len(pre)

blend_out_end = (
    blend_out_start
    + len(blend)
    - 1
)

region_start = max(
    0,
    blend_out_start - 10,
)

region_end = min(
    len(frame_rms),
    blend_out_end + 10,
)

local = frame_rms[
    region_start:region_end
]


print()
print("===== CONTINUITY =====")

print(
    "Whole-motion max frame RMS:",
    f"{frame_rms.max():.2f} mm/frame"
)

print(
    "Splice-region max frame RMS:",
    f"{local.max():.2f} mm/frame"
)

print(
    "Splice-region mean RMS:",
    f"{local.mean():.2f} mm/frame"
)

print()
print("Largest splice-region transitions:")

indices = np.argsort(
    local
)[::-1][:10]

for idx in indices:

    global_idx = (
        region_start + idx
    )

    print(
        f"{global_idx:03d}"
        f"->{global_idx+1:03d} "
        f"RMS={frame_rms[global_idx]:7.2f} "
        f"mm"
    )


# ------------------------------------------------------------
# Final feet
# ------------------------------------------------------------

def output_foot_center(
    frame,
    side,
):
    names = [
        f"{side}HEE",
        f"{side}TOE",
        f"{side}MT1",
        f"{side}MT5",
    ]

    ids = marker_ids(
        labels,
        names,
    )

    return np.mean(
        stitched[frame, ids],
        axis=0,
    )


left_final = output_foot_center(
    len(stitched)-1,
    "L",
)

right_final = output_foot_center(
    len(stitched)-1,
    "R",
)


print()
print("===== FINAL FOOT STATE =====")

print(
    "Left :",
    np.round(
        left_final,
        2,
    )
)

print(
    "Right:",
    np.round(
        right_final,
        2,
    )
)

print(
    "Final L/R height difference:",
    f"{abs(left_final[2]-right_final[2]):.2f} mm"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    OUT,

    fps=np.asarray(
        fps,
        dtype=float,
    ),

    labels=np.asarray(
        labels,
        dtype=str,
    ),

    xyz_mm=stitched,

    source_file=np.asarray(
        str(SRC),
        dtype=str,
    ),

    source_anchor=np.asarray(
        SRC_ANCHOR,
        dtype=int,
    ),

    terminal_anchor=np.asarray(
        TERM_ANCHOR,
        dtype=int,
    ),

    yaw_correction_deg=np.asarray(
        math.degrees(theta),
        dtype=float,
    ),

    terminal_translation_mm=translation,

    blend_source_frames=np.asarray(
        blend_source_frames,
        dtype=int,
    ),

    blend_terminal_frames=np.asarray(
        blend_terminal_frames,
        dtype=int,
    ),
)


print()
print("Saved:")
print(OUT)

print()
print("========================================")
print(" KIT256 3-STEP STITCH COMPLETE")
print("========================================")
