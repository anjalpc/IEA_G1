from pathlib import Path

import yaml

from simulation.isaac_lab.g1_stairs.reference_motion import (
    load_reference_motion,
)


ROOT = Path(__file__).resolve().parents[5]

CONFIG = (
    ROOT
    / "experiments"
    / "g1_stairs"
    / "configs"
    / "026_reference_baseline.yaml"
)

with CONFIG.open("r", encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)


metadata = ROOT / cfg["reference"]["metadata"]
motion = load_reference_motion(metadata)

task_pos = motion.positions_in_task_frame()

contacts = cfg["validated_reference_contacts"]
tread_ranges = cfg["terrain"]["tread_x_ranges_from_start_m"]

body_for_foot = {
    "left": "left_ankle_roll_link",
    "right": "right_ankle_roll_link",
}


print("=" * 78)
print("026 REFERENCE BASELINE")
print("=" * 78)

all_passed = True

for step_name in ("step_1", "step_2", "step_3"):
    contact = contacts[step_name]

    foot = contact["foot"]
    body_name = body_for_foot[foot]
    body_idx = motion.body_index(body_name)

    f0, f1 = contact["support_frame_range"]

    # Use the middle of the validated stable-support interval.
    frame = (f0 + f1) // 2

    x = float(task_pos[frame, body_idx, 0])
    z = float(task_pos[frame, body_idx, 2])

    xmin, xmax = tread_ranges[step_name]

    inside = xmin <= x <= xmax

    print()
    print(
        f"{step_name}: {foot.upper()} foot "
        f"support frame={frame}"
    )
    print(f"  X: {x:.3f} m")
    print(f"  Z: {z:.3f} m")
    print(
        f"  target tread X: "
        f"[{xmin:.3f}, {xmax:.3f}] m"
    )
    print(f"  inside tread: {inside}")

    all_passed &= inside


print()
print("-" * 78)

pelvis_idx = motion.body_index("pelvis")

boundary = int(
    cfg["reference"]["adaptation_boundary"]["frame"]
)

print(
    f"adaptation boundary: frame {boundary} "
    f"({boundary / motion.frequency_hz:.2f} s)"
)

print(
    "pelvis at boundary: "
    f"X={task_pos[boundary,pelvis_idx,0]:.3f} "
    f"Z={task_pos[boundary,pelvis_idx,2]:.3f}"
)

print()
print(
    "RESULT:",
    "PASS" if all_passed else "FAIL",
)

if not all_passed:
    raise SystemExit(1)
