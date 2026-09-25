#!/usr/bin/env python3

from pathlib import Path
import subprocess

import c3d
import matplotlib.pyplot as plt
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

OUT_DIR = (
    ROOT
    / "outputs/kit256_3step_stitch/preview_frames"
)

VIDEO = (
    ROOT
    / "outputs/kit256_3step_stitch/"
      "WSUF01_3step_stitched_preview.mp4"
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


# ---------------------------------------------------------
# Load stitched motion
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


# ---------------------------------------------------------
# Stair coordinate frame
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

rise = np.diff(
    centers[:, 2]
)

mean_rise = float(
    np.mean(rise)
)

floor_z = (
    centers[0, 2]
    - mean_rise
)

nose_s = (
    centers[:, :2]
    - origin_xy
) @ forward_xy

top_z = (
    centers[:, 2]
    - floor_z
)


def transform_points(p):

    xy = (
        p[..., :2]
        - origin_xy
    )

    s = (
        xy[..., 0] * forward_xy[0]
        +
        xy[..., 1] * forward_xy[1]
    )

    y = (
        xy[..., 0] * lateral_xy[0]
        +
        xy[..., 1] * lateral_xy[1]
    )

    z = (
        p[..., 2]
        - floor_z
    )

    return np.stack(
        [s, y, z],
        axis=-1,
    )


motion = transform_points(
    xyz
)


# ---------------------------------------------------------
# Skeleton connections
# ---------------------------------------------------------

edges = [
    # pelvis
    ("LASI", "RASI"),
    ("LASI", "LHIP"),
    ("RASI", "RHIP"),
    ("LHIP", "RHIP"),

    # torso
    ("LASI", "STRN"),
    ("RASI", "STRN"),
    ("STRN", "C7"),

    # left leg
    ("LHIP", "LKNE"),
    ("LKNE", "LANK"),
    ("LANK", "LHEE"),
    ("LHEE", "LTOE"),

    # right leg
    ("RHIP", "RKNE"),
    ("RKNE", "RANK"),
    ("RANK", "RHEE"),
    ("RHEE", "RTOE"),

    # shoulders / arms
    ("C7", "LSHO"),
    ("C7", "RSHO"),
    ("LSHO", "LAEL"),
    ("LAEL", "LWTS"),
    ("RSHO", "RAEL"),
    ("RAEL", "RWTS"),
]

edge_ids = [
    (
        labels.index(a),
        labels.index(b),
    )
    for a, b in edges
]

show_names = [
    "LASI", "RASI",
    "LHIP", "RHIP",
    "LKNE", "RKNE",
    "LANK", "RANK",
    "LHEE", "RHEE",
    "LTOE", "RTOE",
    "STRN", "C7",
    "LSHO", "RSHO",
    "LAEL", "RAEL",
    "LWTS", "RWTS",
]

show_ids = [
    labels.index(x)
    for x in show_names
]


# ---------------------------------------------------------
# Render every second source frame.
#
# Motion remains 100 Hz.
# Preview is 50 FPS only to reduce rendering time/file size.
# ---------------------------------------------------------

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for old in OUT_DIR.glob("*.png"):
    old.unlink()

frame_ids = list(
    range(
        0,
        len(motion),
        2,
    )
)

preview_fps = (
    fps / 2.0
)


def draw_stairs(ax):

    # floor
    ax.plot(
        [-1900, 0],
        [0, 0],
        linewidth=2,
    )

    # step 1
    ax.plot(
        [0, nose_s[1]],
        [top_z[0], top_z[0]],
        linewidth=3,
    )

    ax.plot(
        [0, 0],
        [0, top_z[0]],
        linewidth=3,
    )

    # step 2
    ax.plot(
        [nose_s[1], nose_s[2]],
        [top_z[1], top_z[1]],
        linewidth=3,
    )

    ax.plot(
        [nose_s[1], nose_s[1]],
        [top_z[0], top_z[1]],
        linewidth=3,
    )

    # step 3 / upper surface
    ax.plot(
        [nose_s[2], 1450],
        [top_z[2], top_z[2]],
        linewidth=3,
    )

    ax.plot(
        [nose_s[2], nose_s[2]],
        [top_z[1], top_z[2]],
        linewidth=3,
    )


print("========================================")
print(" KIT256 3-STEP PREVIEW RENDER")
print("========================================")

print("Motion frames :", len(motion))
print("Source FPS    :", fps)
print("Preview FPS   :", preview_fps)
print("Render frames :", len(frame_ids))

for out_i, frame_i in enumerate(frame_ids):

    p = motion[frame_i]

    fig, ax = plt.subplots(
        figsize=(12.8, 7.2),
        dpi=100,
    )

    draw_stairs(ax)

    # skeleton in side view: forward vs height
    for ia, ib in edge_ids:

        ax.plot(
            [
                p[ia, 0],
                p[ib, 0],
            ],
            [
                p[ia, 2],
                p[ib, 2],
            ],
            linewidth=2,
        )

    ax.scatter(
        p[show_ids, 0],
        p[show_ids, 2],
        s=18,
    )

    ax.set_xlim(
        -1900,
        1450,
    )

    ax.set_ylim(
        -100,
        2100,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(True)

    ax.set_xlabel(
        "Stair-forward position [mm]"
    )

    ax.set_ylabel(
        "Height [mm]"
    )

    ax.set_title(
        "KIT WSUF01 → stitched 3-step reference\n"
        f"source frame {frame_i:03d}   "
        f"time {frame_i/fps:.2f} s"
    )

    fig.tight_layout()

    path = (
        OUT_DIR
        / f"frame_{out_i:04d}.png"
    )

    fig.savefig(path)

    plt.close(fig)

    if (
        out_i % 50 == 0
        or out_i == len(frame_ids) - 1
    ):
        print(
            f"{out_i+1}/{len(frame_ids)}"
        )


print()
print("Encoding MP4...")

cmd = [
    "ffmpeg",
    "-y",
    "-framerate",
    f"{preview_fps:g}",
    "-i",
    str(
        OUT_DIR / "frame_%04d.png"
    ),
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "18",
    str(VIDEO),
]

subprocess.run(
    cmd,
    check=True,
)

print()
print("Saved:")
print(VIDEO)

print()
print("========================================")
print(" PREVIEW COMPLETE")
print("========================================")
