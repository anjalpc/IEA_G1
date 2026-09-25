#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path


MOTION_ROOT = Path(__file__).resolve().parents[1]
MAHR_ROOT = MOTION_ROOT / "MAHR"

if not MAHR_ROOT.is_dir():
    raise RuntimeError(f"MAHR directory missing: {MAHR_ROOT}")

sys.path.insert(0, str(MAHR_ROOT))


from config.MocapConfig.CMUAsfAmcConfig import (
    CMU_AsfAmc_WholeBody_Config,
)
from config.RobotConfig.UnitreeG1 import (
    Unitree_G1_WholeBody_Config,
)


ASF = (
    MAHR_ROOT
    / "mahr/rsc/all_asfamc/subjects/14/14.asf"
)

AMC = (
    MAHR_ROOT
    / "mahr/rsc/all_asfamc/subjects/14/"
    / "14_21_ascent_170_540.amc"
)

DEFAULT_OUT = (
    MOTION_ROOT
    / "outputs/cmu_to_g1_trial21_smoke"
)


class CMU_G1_WholeBody_Config(CMU_AsfAmc_WholeBody_Config):
    """
    CMU whole-body graph adapted to MAHR's Unitree G1 graph.

    The stock MAHR CMU config has 15 graph nodes.
    Unitree G1 has 17.

    CMU already contains lthumb/rthumb, so expose them as
    graph nodes to produce a 17 <-> 17 correspondence.
    """

    def __init__(self, norm_pose_type="stand"):
        super().__init__(norm_pose_type=norm_pose_type)

        self.key_bodies = [
            # Root
            "root",

            # Left leg
            "lfemur",
            "ltibia",
            "lfoot",

            # Right leg
            "rfemur",
            "rtibia",
            "rfoot",

            # Trunk
            "thorax",

            # Left arm
            "lclavicle",
            "lhumerus",
            "lwrist",
            "lthumb",

            # Right arm
            "rclavicle",
            "rhumerus",
            "rwrist",
            "rthumb",

            # Ground
            "ground",
        ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run MAHR optimization. Without this flag only preflight is performed.",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=60,
        help="Process every Nth source frame. 60 gives a small smoke test.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    print("========================================")
    print(" CMU -> UNITREE G1 PREFLIGHT")
    print("========================================")

    print(f"MAHR root : {MAHR_ROOT}")
    print(f"ASF       : {ASF}")
    print(f"AMC       : {AMC}")

    if not ASF.is_file():
        raise FileNotFoundError(ASF)

    if not AMC.is_file():
        raise FileNotFoundError(AMC)

    print("\n[01] source files: PASS")

    # ---------------------------------------------------------
    # Mocap
    # ---------------------------------------------------------

    mocap = CMU_G1_WholeBody_Config(
        norm_pose_type="stand"
    )

    print("[02] loading CMU ASF/AMC ...")

    mocap.load_motion(
        str(ASF),
        str(AMC),
        120.0,
    )

    num_frames = mocap.uni_repr.num_frames
    fps = mocap.uni_repr.fps

    print(f"     frames : {num_frames}")
    print(f"     fps    : {fps}")

    if num_frames != 371:
        raise RuntimeError(
            f"Expected 371 frames, got {num_frames}"
        )

    print("[03] CMU motion load: PASS")

    # ---------------------------------------------------------
    # Verify our additional nodes actually exist
    # ---------------------------------------------------------

    motion_names = set(
        mocap.uni_repr.to_dict()["names"]
    )

    required_mocap_nodes = [
        x for x in mocap.key_bodies
        if x != "ground"
    ]

    missing_mocap_nodes = [
        x for x in required_mocap_nodes
        if x not in motion_names
    ]

    print(
        "[04] required CMU graph nodes:",
        len(mocap.key_bodies),
    )

    if missing_mocap_nodes:
        raise RuntimeError(
            "Missing CMU nodes: "
            + str(missing_mocap_nodes)
        )

    print("[05] lthumb/rthumb availability: PASS")

    # ---------------------------------------------------------
    # Robot
    # ---------------------------------------------------------

    robot = Unitree_G1_WholeBody_Config(
        norm_pose_type="stand",
        fk_backend="mujoco",
    )

    print(
        "[06] robot FK backend:",
        type(robot.robot).__name__,
    )

    print(
        "[07] CMU graph nodes :",
        len(mocap.key_bodies),
    )

    print(
        "[08] G1 graph nodes  :",
        len(robot.key_joints),
    )

    if len(mocap.key_bodies) != len(robot.key_joints):
        raise RuntimeError(
            "CMU/G1 graph-size mismatch"
        )

    print("[09] graph-size contract: PASS")

    # ---------------------------------------------------------
    # Explicit graph correspondence
    # ---------------------------------------------------------

    print("\n===== GRAPH CORRESPONDENCE =====")

    for i, (human, g1) in enumerate(
        zip(mocap.key_bodies, robot.key_joints)
    ):
        print(f"{i:02d}: {human:12s} -> {g1}")

    # ---------------------------------------------------------
    # Actuated-joint contract
    # ---------------------------------------------------------

    actuated = list(
        robot.robot.actuated_joint_names
    )

    missing_mapping = [
        j for j in actuated
        if j not in robot.index_mapping
    ]

    print("\n===== G1 OUTPUT CONTRACT =====")
    print("Actuated joints :", len(actuated))
    print(
        "Index mappings  :",
        len(robot.index_mapping),
    )
    print(
        "Missing mapping :",
        missing_mapping,
    )

    if missing_mapping:
        raise RuntimeError(
            "Incomplete G1 output mapping"
        )

    print("[10] G1 joint mapping: PASS")

    expected_output_frames = math.ceil(
        num_frames / args.frame_skip
    )

    print("\n===== SMOKE TEST PLAN =====")
    print(
        "Source frames        :",
        num_frames,
    )
    print(
        "Frame skip           :",
        args.frame_skip,
    )
    print(
        "Expected output frames:",
        expected_output_frames,
    )
    print(
        "Output FPS           :",
        fps / args.frame_skip,
    )

    if not args.run:
        print("\n========================================")
        print(" PREFLIGHT PASS")
        print(" No optimization was started.")
        print("========================================")
        return

    # ---------------------------------------------------------
    # Actual MAHR retargeting
    # ---------------------------------------------------------

    from motion_retarget_graph_distance import (
        _build_edge_filters,
        _setup_logger,
        motion_retarget_graph_distance,
    )

    _setup_logger()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_filters = _build_edge_filters(
        len(mocap.key_bodies)
    )

    print("\n========================================")
    print(" STARTING CMU -> G1 RETARGETING")
    print("========================================")
    print("Output:", args.out_dir)

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

    print("\n========================================")
    print(" RETARGETING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
