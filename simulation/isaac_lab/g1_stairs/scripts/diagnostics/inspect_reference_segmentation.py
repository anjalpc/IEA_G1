from pathlib import Path

from simulation.isaac_lab.g1_stairs.reference_motion import (
    load_reference_motion,
)
from simulation.isaac_lab.g1_stairs.reference_segmentation import (
    segment_first_three_steps,
)


ROOT = Path(__file__).resolve().parents[5]

metadata = (
    ROOT
    / "experiments"
    / "g1_stairs"
    / "reference_data"
    / "stair_ascent_reference.yaml"
)

motion = load_reference_motion(metadata)
seg = segment_first_three_steps(motion)

print("REFERENCE SEGMENTATION")
print("=" * 70)

for transition in seg.transitions:
    print(
        f"{transition.name}: "
        f"{transition.foot:5s} "
        f"frames {transition.frame_start:3d}->{transition.frame_end:3d} "
        f"time {transition.time_start_s:5.2f}->{transition.time_end_s:5.2f}s "
        f"z {transition.start_height_m:.3f}->{transition.end_height_m:.3f} m"
    )

print()
print("cut_frame:", seg.cut_frame)
print("cut_time_s:", f"{seg.cut_time_s:.2f}")
