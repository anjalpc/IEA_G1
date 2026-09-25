#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/ubuntu/workspace/IEA_G1/motion_imitation")
MAHR = ROOT / "MAHR"

sys.path.insert(0, str(MAHR))

from config.MocapConfig.CMUAsfAmcConfig import (
    CMU_AsfAmc_WholeBody_Config,
)
from config.RobotConfig.UnitreeG1 import (
    Unitree_G1_WholeBody_Config,
)


ASF = (
    MAHR
    / "mahr/rsc/all_asfamc/subjects/14/14.asf"
)

AMC = (
    MAHR
    / "mahr/rsc/all_asfamc/subjects/14/"
      "14_21_ascent_170_540.amc"
)

DEFAULT_OUT = (
    ROOT
    / "outputs/cmu_to_g1_trial21_clean15_20fps"
)


class Unitree_G1_CMU15_Config(Unitree_G1_WholeBody_Config):
    """
    G1 graph matched to MAHR's original 15-node CMU graph.

    Thumb markers are intentionally excluded because this MAHR
    G1 URDF has fixed wrist DOFs.
    """

    def __init__(self, norm_pose_type="stand", fk_backend="mujoco"):
        super().__init__(
            norm_pose_type=norm_pose_type,
            fk_backend=fk_backend,
        )

        self.key_joints = [
            "pelvis_marker",

            "left_knee_marker",
            "left_ankle_marker",
            "left_toe_marker",

            "right_knee_marker",
            "right_ankle_marker",
            "right_toe_marker",

            "head_marker",

            "left_shoulder_marker",
            "left_elbow_marker",
            "left_hand_marker",

            "right_shoulder_marker",
            "right_elbow_marker",
            "right_hand_marker",

            "ground",
        ]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="store_true",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    print("========================================")
    print(" CMU -> G1 CLEAN 15-NODE PREFLIGHT")
    print("========================================")

    print("ROOT :", ROOT)
    print("MAHR :", MAHR)
    print("ASF  :", ASF)
    print("AMC  :", AMC)
    print("OUT  :", args.out_dir)

    if not ASF.is_file():
        raise FileNotFoundError(ASF)

    if not AMC.is_file():
        raise FileNotFoundError(AMC)

    print("\n[01] source files: PASS")

    # --------------------------------------------------------
    # Source mocap
    # --------------------------------------------------------

    mocap = CMU_AsfAmc_WholeBody_Config(
        norm_pose_type="stand"
    )

    mocap.load_motion(
        str(ASF),
        str(AMC),
        120.0,
    )

    nframes = mocap.uni_repr.num_frames
    fps = float(mocap.uni_repr.fps)

    print("[02] source frames:", nframes)
    print("[03] source FPS   :", fps)

    if nframes != 371:
        raise RuntimeError(
            f"Expected 371 frames, got {nframes}"
        )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    robot = Unitree_G1_CMU15_Config(
        norm_pose_type="stand",
        fk_backend="mujoco",
    )

    print()
    print("===== GRAPH SIZE =====")
    print("CMU:", len(mocap.key_bodies))
    print("G1 :", len(robot.key_joints))

    if len(mocap.key_bodies) != 15:
        raise RuntimeError(
            f"Expected CMU 15 nodes, got {len(mocap.key_bodies)}"
        )

    if len(robot.key_joints) != 15:
        raise RuntimeError(
            f"Expected G1 15 nodes, got {len(robot.key_joints)}"
        )

    print("[04] 15 <-> 15 graph contract: PASS")

    # --------------------------------------------------------
    # Exact correspondence
    # --------------------------------------------------------

    print()
    print("===== GRAPH CORRESPONDENCE =====")

    for i, (human, g1) in enumerate(
        zip(mocap.key_bodies, robot.key_joints)
    ):
        print(
            f"{i:02d}: "
            f"{human:12s} -> {g1}"
        )

    # --------------------------------------------------------
    # Output joint contract
    # --------------------------------------------------------

    actuated = list(
        robot.robot.actuated_joint_names
    )

    missing = [
        j for j in actuated
        if j not in robot.index_mapping
    ]

    print()
    print("===== OUTPUT CONTRACT =====")
    print("Actuated joints :", len(actuated))
    print("Index mappings  :", len(robot.index_mapping))
    print("Missing mappings:", missing)

    if len(actuated) != 23:
        raise RuntimeError(
            f"Expected 23 actuated joints, got {len(actuated)}"
        )

    if missing:
        raise RuntimeError(
            f"Missing mappings: {missing}"
        )

    print("[05] 23-joint output contract: PASS")

    # --------------------------------------------------------
    # Source root yaw diagnostic
    # --------------------------------------------------------

    d = mocap.uni_repr.to_dict(
        is_local=False
    )

    root_quats = np.asarray([
        frame[0]
        for frame in d["global_rotations"]
    ])

    yaw = Rotation.from_quat(
        root_quats
    ).as_euler(
        "xyz",
        degrees=True,
    )[:, 2]

    yaw_unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(yaw)
        )
    )

    print()
    print("===== SOURCE ROOT YAW =====")
    print(
        f"start : {yaw_unwrapped[0]: .3f} deg"
    )
    print(
        f"end   : {yaw_unwrapped[-1]: .3f} deg"
    )
    print(
        f"change: "
        f"{yaw_unwrapped[-1] - yaw_unwrapped[0]: .3f} deg"
    )
    print(
        f"range : "
        f"{yaw_unwrapped.min(): .3f} "
        f"to {yaw_unwrapped.max(): .3f} deg"
    )

    # --------------------------------------------------------
    # Planned dense test
    # --------------------------------------------------------

    expected = math.ceil(
        nframes / args.frame_skip
    )

    print()
    print("===== TEST PLAN =====")
    print("Frame skip :", args.frame_skip)
    print("Output FPS :", fps / args.frame_skip)
    print("Frames     :", expected)

    if not args.run:

        print()
        print("========================================")
        print(" CLEAN15 PREFLIGHT PASS")
        print(" No optimization started.")
        print("========================================")

        return

    # Fresh-cache safety check
    if args.out_dir.exists():
        existing = list(
            args.out_dir.iterdir()
        )

        if existing:
            raise RuntimeError(
                "Output directory is not empty. "
                "Clean15 must use a fresh cache: "
                f"{args.out_dir}"
            )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    from motion_retarget_graph_distance import (
        _build_edge_filters,
        _setup_logger,
        motion_retarget_graph_distance,
    )

    _setup_logger()

    edge_filters = _build_edge_filters(
        len(mocap.key_bodies)
    )

    print()
    print("========================================")
    print(" START CLEAN15 RETARGETING")
    print("========================================")

    motion_retarget_graph_distance(
        str(args.out_dir),
        mocap,
        robot,

        flag_force_generate_motion_dict=False,
        flag_force_generate_graph=False,

        flag_filter_edges_when_generating_graph=True,
        edge_filters=edge_filters,

        flag_normalize_edge_weights_using_T_pose=True,
        flag_weighted_optimization_by_closeness=True,

        height_map_png=None,

        frame_skip=args.frame_skip,
    )

    print()
    print("========================================")
    print(" CLEAN15 RETARGETING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
