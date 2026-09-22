from pathlib import Path

import numpy as np

from simulation.isaac_lab.g1_stairs.reference_motion import (
    load_reference_motion,
)


ROOT = Path(__file__).resolve().parents[4]

METADATA = (
    ROOT
    / "experiments"
    / "g1_stairs"
    / "reference_data"
    / "stair_ascent_reference.yaml"
)


def test_reference_motion_contract():
    motion = load_reference_motion(METADATA)

    assert motion.frame_count == 499
    assert motion.frequency_hz == 50.0
    assert motion.dof_pos.shape == (499, 29)
    assert motion.dof_vel.shape == (499, 29)
    assert motion.body_pos_w.shape == (499, 14, 3)


def test_reference_motion_task_frame_moves_forward_positive_x():
    motion = load_reference_motion(METADATA)

    task_pos = motion.positions_in_task_frame()
    pelvis = task_pos[:, motion.body_index("pelvis")]

    forward_delta = pelvis[-1, 0] - pelvis[0, 0]

    assert forward_delta > 3.0
    assert np.isclose(forward_delta, 3.0371, atol=0.01)


def test_reference_motion_ankles_are_distinct():
    motion = load_reference_motion(METADATA)

    assert not np.allclose(
        motion.left_ankle,
        motion.right_ankle,
    )
