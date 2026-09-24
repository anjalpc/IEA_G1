"""Project-owned Unitree G1 asset configuration for stair training.

The pinned Unitree RL Lab configuration is reused unchanged except for the
asset source. The upstream checkout contains placeholder model paths, so this
project explicitly loads the pinned mode-5 G1 URDF from unitree_ros.
"""

from __future__ import annotations

import os
from pathlib import Path

from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_CFG,
    UnitreeUrdfFileCfg,
)

UNITREE_ROS_ROOT = Path(
    os.environ.get("UNITREE_ROS_ROOT", "/workspace/unitree_ros")
)

G1_29DOF_URDF = (
    Path(__file__).resolve().parent
    / "assets"
    / "g1_29dof_rev_1_0_stair_collision.urdf"
)

G1_STAIRS_ROBOT_CFG = UNITREE_G1_29DOF_CFG.replace(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(G1_29DOF_URDF),
    )
)
