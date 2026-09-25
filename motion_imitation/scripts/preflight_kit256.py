#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import c3d


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")

KIT = (
    ROOT
    / "01_source/KIT/motion_256"
)

MOTION = KIT / "WSUF01.c3d"
STAIRS = KIT / "STAIRS.c3d"
FORCE = KIT / "WSUF01.csv"
VIDEO = KIT / "WSUF01_final.avi"


def clean_label(name):
    name = str(name).strip()

    if ":" in name:
        name = name.split(":")[-1]

    return name.strip()


def load_c3d(path):

    with open(path, "rb") as f:
        reader = c3d.Reader(f)

        labels_raw = [
            str(x).strip()
            for x in reader.point_labels
        ]

        labels = [
            clean_label(x)
            for x in labels_raw
        ]

        rate = float(reader.point_rate)
        analog_rate = float(reader.analog_rate)

        xyz_frames = []
        valid_frames = []
        frame_numbers = []

        for frame_no, points, analog in reader.read_frames(
            copy=True
        ):
            points = np.asarray(points)

            xyz = np.asarray(
                points[:, :3],
                dtype=float,
            )

            valid = np.isfinite(xyz).all(axis=1)

            if points.shape[1] >= 4:
                residual = np.asarray(
                    points[:, 3],
                    dtype=float,
                )

                valid &= residual >= 0.0

            xyz_frames.append(xyz)
            valid_frames.append(valid)
            frame_numbers.append(frame_no)

    return {
        "labels_raw": labels_raw,
        "labels": labels,
        "rate": rate,
        "analog_rate": analog_rate,
        "xyz": np.asarray(xyz_frames),
        "valid": np.asarray(valid_frames),
        "frame_numbers": frame_numbers,
    }


def read_force_csv(path):

    rows = []

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    for line in text.splitlines():

        parts = [
            x.strip()
            for x in line.split(";")
        ]

        if len(parts) < 7:
            continue

        try:
            vals = [
                float(parts[i].replace(",", "."))
                for i in range(7)
            ]
        except ValueError:
            continue

        rows.append(vals)

    return np.asarray(rows, dtype=float)


print("========================================")
print(" KIT MOTION 256 PREFLIGHT")
print("========================================")

for p in [MOTION, STAIRS, FORCE, VIDEO]:
    print(
        f"{p.name:20s}",
        "PASS" if p.is_file() else "MISSING"
    )

assert MOTION.is_file()
assert STAIRS.is_file()
assert FORCE.is_file()
assert VIDEO.is_file()

print()
print("========================================")
print(" WSUF01 C3D")
print("========================================")

motion = load_c3d(MOTION)

N = motion["xyz"].shape[0]

print("Point rate    :", motion["rate"], "Hz")
print("Analog rate   :", motion["analog_rate"], "Hz")
print("Frames        :", N)
print(
    "Duration      :",
    (N - 1) / motion["rate"],
    "s",
)
print("Markers       :", len(motion["labels"]))

print()
print("===== MARKER LABELS =====")

for i, name in enumerate(motion["labels"]):
    print(
        f"{i:02d}: "
        f"{name}"
    )

required = [
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",

    "LHIP",
    "RHIP",

    "LKNE",
    "RKNE",

    "LANK",
    "RANK",

    "LTOE",
    "RTOE",

    "LHEE",
    "RHEE",

    "LSHO",
    "RSHO",

    "LAEL",
    "RAEL",

    "LWTS",
    "RWTS",

    "C7",
    "STRN",
]

print()
print("===== REQUIRED BODY MARKERS =====")

missing = []

for name in required:

    if name not in motion["labels"]:
        print(
            f"{name:8s} MISSING"
        )

        missing.append(name)

        continue

    idx = motion["labels"].index(name)

    valid_ratio = (
        motion["valid"][:, idx].mean()
        * 100.0
    )

    print(
        f"{name:8s} "
        f"PASS "
        f"valid={valid_ratio:6.2f}%"
    )

if missing:
    print()
    print("Missing markers:", missing)
else:
    print()
    print("REQUIRED MARKER CONTRACT: PASS")


print()
print("========================================")
print(" STAIRS C3D")
print("========================================")

stairs = load_c3d(STAIRS)

print("Point rate :", stairs["rate"], "Hz")
print("Frames     :", stairs["xyz"].shape[0])
print("Markers    :", len(stairs["labels"]))

print()
print("===== STATIC STAIR MARKER MEDIANS =====")

for j, label in enumerate(stairs["labels"]):

    mask = stairs["valid"][:, j]

    if not np.any(mask):
        print(
            f"{label:12s} NO VALID DATA"
        )

        continue

    p = np.median(
        stairs["xyz"][mask, j, :],
        axis=0,
    )

    print(
        f"{label:12s} "
        f"x={p[0]:10.3f} "
        f"y={p[1]:10.3f} "
        f"z={p[2]:10.3f}"
    )


print()
print("========================================")
print(" FORCE INSOLE CSV")
print("========================================")

force = read_force_csv(FORCE)

assert force.ndim == 2
assert force.shape[1] == 7

t = force[:, 0]

left_force = force[:, 1]
right_force = force[:, 4]

dt = np.median(
    np.diff(t)
)

force_rate = 1.0 / dt

print("Samples       :", len(force))
print("Median dt     :", dt, "s")
print("Sample rate   :", force_rate, "Hz")
print(
    "Time range    :",
    t[0],
    "->",
    t[-1],
    "s",
)
print(
    "Duration      :",
    t[-1] - t[0],
    "s",
)

print()
print("===== FORCE RANGE =====")

print(
    "Left :",
    f"{left_force.min():.2f}",
    "->",
    f"{left_force.max():.2f}",
    "N",
)

print(
    "Right:",
    f"{right_force.min():.2f}",
    "->",
    f"{right_force.max():.2f}",
    "N",
)


print()
print("========================================")
print(" RATE / DURATION CHECK")
print("========================================")

c3d_duration = (
    motion["xyz"].shape[0] - 1
) / motion["rate"]

force_duration = (
    t[-1] - t[0]
)

print(
    "C3D motion :",
    motion["rate"],
    "Hz,",
    f"{c3d_duration:.3f}",
    "s",
)

print(
    "Force CSV  :",
    f"{force_rate:.1f}",
    "Hz,",
    f"{force_duration:.3f}",
    "s",
)

if abs(
    c3d_duration - force_duration
) > 0.1:

    print()
    print(
        "NOTE: C3D and force CSV durations differ."
    )

    print(
        "Contact data must be time-synchronized "
        "before we use force labels."
    )

print()
print("========================================")
print(" KIT256 RAW PREFLIGHT COMPLETE")
print("========================================")
