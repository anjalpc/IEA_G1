"""Isaac Lab overrides for fixed-start G1 stair-ascent training.

This deliberately reuses Unitree's pinned G1 velocity environment. Import it
only after Isaac Lab's AppLauncher has started the simulation application.
"""

from __future__ import annotations

import copy
import importlib

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .contracts import (
    BODY_JOINTS,
    FIXED_FORWARD_COMMAND_MPS,
    NOMINAL_RISER_HEIGHT_M,
)
from .isaac_terrain import ThreeStepStairsCfg
from .robot_cfg import G1_STAIRS_ROBOT_CFG
from . import rewards as stair_rewards
from . import reference_tracking as reference_rewards
from . import terrain_observations
from . import unified_rewards


_UPSTREAM = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)

# Isaac Lab @configclass fields are instance attributes after decoration.
# Keep one upstream environment instance as the immutable template for
# scene, command, and reward terms that we deep-copy below.
_UPSTREAM_CFG = _UPSTREAM.RobotEnvCfg()

COURSE_SIZE = (5.75, 1.2)
STAIR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=COURSE_SIZE,
    border_width=1.0,
    num_rows=8,
    num_cols=16,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "three_step_ascent": ThreeStepStairsCfg(
            proportion=1.0,
            size=COURSE_SIZE,
            # Begin with the exact thesis staircase. The 5–20 cm
            # curriculum will be introduced only after the nominal task
            # passes its smoke and evaluation gates.
            riser_height_range=(
                NOMINAL_RISER_HEIGHT_M,
                NOMINAL_RISER_HEIGHT_M,
            ),
            tread_depth=0.25,
            step_count=3,
            approach_length=2.0,
            platform_length=3.0,
            start_clearance=1.0,
        )
    },
)


@configclass
class StairSceneCfg(_UPSTREAM.RobotSceneCfg):
    """Unitree G1 scene using the exact nominal three-step terrain."""

    terrain = copy.deepcopy(_UPSTREAM_CFG.scene.terrain)
    terrain.terrain_generator = STAIR_TERRAIN_CFG
    terrain.max_init_terrain_level = 0

    # Project-owned mode-5 G1 asset. Keep the upstream actuator, initial-state,
    # SDK-joint, and limit configuration, but load the pinned unitree_ros URDF
    # instead of Unitree RL Lab's placeholder model path.
    robot = G1_STAIRS_ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Visual-only material for clearer evaluation recordings.
    # Collision geometry remains unchanged.
    terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.34, 0.36, 0.39),
        roughness=0.82,
        metallic=0.0,
    )


@configclass
class StairActionsCfg(_UPSTREAM.ActionsCfg):
    """Control the 29 mode-5 body joints, never the BrainCo fingers."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(BODY_JOINTS),
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class StairObservationsCfg(_UPSTREAM.ObservationsCfg):
    """Expose local stair heights to both actor and privileged critic."""

    @configclass
    class PolicyCfg(_UPSTREAM.ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            history_length=1,
        )

        stair_geometry = ObsTerm(
            func=terrain_observations.stair_geometry,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            },
            clip=(-1.0, 1.0),
            history_length=1,
        )

        # Dynamically enabled only for reference-conditioned experiments.
        reference_joint_position_error = None
        reference_joint_velocity_error = None
        reference_phase = None
        reference_root_position_error = None
        reference_root_orientation_error = None
        reference_root_linear_velocity_error = None
        reference_root_angular_velocity_error = None

        def __post_init__(self):
            super().__post_init__()

            # Do not apply one history length to every observation term.
            # Keep five frames of proprioception, but only the current
            # terrain scan.
            self.history_length = None
            self.flatten_history_dim = True
            self.enable_corruption = False

            self.base_ang_vel.history_length = 5
            self.projected_gravity.history_length = 5
            self.velocity_commands.history_length = 5
            self.joint_pos_rel.history_length = 5
            self.joint_vel_rel.history_length = 5
            self.last_action.history_length = 5
            self.height_scan.history_length = 1
            self.stair_geometry.history_length = 1

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(_UPSTREAM.ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            history_length=1,
        )

        stair_geometry = ObsTerm(
            func=terrain_observations.stair_geometry,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            },
            clip=(-1.0, 1.0),
            history_length=1,
        )

        # Dynamically enabled only for reference-conditioned experiments.
        reference_joint_position_error = None
        reference_joint_velocity_error = None
        reference_phase = None
        reference_root_position_error = None
        reference_root_orientation_error = None
        reference_root_linear_velocity_error = None
        reference_root_angular_velocity_error = None

        def __post_init__(self):
            super().__post_init__()

            # Privileged critic keeps five frames of proprioception and
            # linear velocity, but only the current terrain scan.
            self.history_length = None
            self.flatten_history_dim = True

            self.base_lin_vel.history_length = 5
            self.base_ang_vel.history_length = 5
            self.projected_gravity.history_length = 5
            self.velocity_commands.history_length = 5
            self.joint_pos_rel.history_length = 5
            self.joint_vel_rel.history_length = 5
            self.last_action.history_length = 5
            self.height_scan.history_length = 1
            self.stair_geometry.history_length = 1

    critic: CriticCfg = CriticCfg()


@configclass
class StairCommandsCfg(_UPSTREAM.CommandsCfg):
    """Use one deterministic forward command during nominal learning."""

    base_velocity = copy.deepcopy(_UPSTREAM_CFG.commands.base_velocity)
    base_velocity.ranges.lin_vel_x = (
        FIXED_FORWARD_COMMAND_MPS,
        FIXED_FORWARD_COMMAND_MPS,
    )
    base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    base_velocity.limit_ranges = copy.deepcopy(base_velocity.ranges)
    base_velocity.rel_standing_envs = 0.0
    base_velocity.rel_heading_envs = 0.0
    base_velocity.heading_command = False


@configclass
class StairEventsCfg(_UPSTREAM.EventCfg):
    """Begin with a deterministic pose and no pushes or broad randomization."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reference_state_reset = None

    base_external_force_torque = None
    physics_material = None
    add_base_mass = None
    push_robot = None


@configclass
class StairCurriculumCfg(_UPSTREAM.CurriculumCfg):
    """Disable all curricula during the exact nominal training phase."""

    terrain_levels = None
    lin_vel_cmd_levels = None


@configclass
class StairRewardsCfg:
    """Phase 12 reward catalog; the resolved experiment selects active terms."""

    reference_joint_position_tracking = RewTerm(
        func=reference_rewards.reference_joint_position_tracking,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "metadata_path": (
                "experiments/g1_stairs/reference_data/"
                "stair_ascent_reference.yaml"
            ),
            "start_frame": 0,
            "end_frame": 260,
            "std_rad": 0.35,
        },
    )

    reference_joint_velocity_tracking = RewTerm(
        func=reference_rewards.reference_joint_velocity_tracking,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "metadata_path": (
                "experiments/g1_stairs/reference_data/"
                "stair_ascent_reference.yaml"
            ),
            "start_frame": 0,
            "end_frame": 260,
            "std_radps": 2.0,
        },
    )

    reference_root_position_tracking = RewTerm(
        func=reference_rewards.reference_root_position_tracking,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "metadata_path": (
                "experiments/g1_stairs/reference_data/"
                "stair_ascent_reference.yaml"
            ),
            "start_frame": 0,
            "end_frame": 246,
            "std_m": 0.15,
        },
    )

    reference_root_orientation_tracking = RewTerm(
        func=reference_rewards.reference_root_orientation_tracking,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "metadata_path": (
                "experiments/g1_stairs/reference_data/"
                "stair_ascent_reference.yaml"
            ),
            "start_frame": 0,
            "end_frame": 246,
            "std_rad": 0.35,
        },
    )

    reference_root_linear_velocity_tracking = RewTerm(
        func=reference_rewards.reference_root_linear_velocity_tracking,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "metadata_path": (
                "experiments/g1_stairs/reference_data/"
                "stair_ascent_reference.yaml"
            ),
            "start_frame": 0,
            "end_frame": 246,
            "std_mps": 0.50,
        },
    )

    reference_root_angular_velocity_tracking = RewTerm(
        func=reference_rewards.reference_root_angular_velocity_tracking,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "metadata_path": (
                "experiments/g1_stairs/reference_data/"
                "stair_ascent_reference.yaml"
            ),
            "start_frame": 0,
            "end_frame": 246,
            "std_radps": 1.0,
        },
    )

    forward_progress = RewTerm(
        func=stair_rewards.forward_progress,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stair_progress = RewTerm(
        func=stair_rewards.stair_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "step_edges_m": (1.0, 1.25, 1.50),
        },
    )
    upright = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Pretrained locomotion-preservation terms.
    # These are available to experiment configs but are not globally active.
    track_lin_vel_xy = copy.deepcopy(
        _UPSTREAM_CFG.rewards.track_lin_vel_xy
    )
    track_ang_vel_z = copy.deepcopy(
        _UPSTREAM_CFG.rewards.track_ang_vel_z
    )
    joint_acc = copy.deepcopy(
        _UPSTREAM_CFG.rewards.joint_acc
    )
    dof_pos_limits = copy.deepcopy(
        _UPSTREAM_CFG.rewards.dof_pos_limits
    )
    feet_slide = copy.deepcopy(
        _UPSTREAM_CFG.rewards.feet_slide
    )
    undesired_contacts = copy.deepcopy(
        _UPSTREAM_CFG.rewards.undesired_contacts
    )
    action_rate = copy.deepcopy(
        _UPSTREAM_CFG.rewards.action_rate
    )
    waist_deviation = copy.deepcopy(
        _UPSTREAM_CFG.rewards.joint_deviation_waists
    )

    # 035+: geometry-driven stair adaptation.
    # Whichever foot is physically in swing may earn this reward.
    terrain_relative_swing_clearance = RewTerm(
        func=unified_rewards.terrain_relative_swing_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            ),
            "nominal_ankle_height_m": 0.10,
            "lookahead_distance_m": 0.35,
            "lip_distance_m": 0.05,
            "clearance_margin_m": 0.04,
            "forward_speed_scale_mps": 0.15,
        },
    )

    terrain_relative_clearance_shortfall = RewTerm(
        func=unified_rewards.terrain_relative_clearance_shortfall,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            ),
            "nominal_ankle_height_m": 0.10,
            "lookahead_distance_m": 0.35,
            "lip_distance_m": 0.03,
            "clearance_margin_m": 0.04,
        },
    )

    terrain_riser_face_strike = RewTerm(
        func=unified_rewards.terrain_riser_face_strike,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "nominal_ankle_height_m": 0.10,
            "face_window_m": 0.035,
            "clearance_margin_m": 0.025,
            "min_forward_speed_mps": 0.02,
        },
    )

    terrain_tread_touchdown = RewTerm(
        func=unified_rewards.terrain_tread_touchdown,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            ),
            "nominal_ankle_height_m": 0.10,
            "x_margin_m": 0.02,
            "z_tolerance_m": 0.035,
            "upright_min": 0.80,
        },
    )

    # Historical stair-specific swing-foot clearance.
    # Retained for reproducibility of earlier experiments.
    stair_swing_clearance = RewTerm(
        func=stair_rewards.stair_swing_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "riser_edges_m": (1.0, 1.25, 1.50),
            "tread_heights_m": (0.20, 0.40, 0.60),
            "activation_distance_m": 0.20,
            "lip_distance_m": 0.05,
            "clearance_margin_m": 0.05,
            "std": 0.01,
            "tanh_mult": 2.0,
        },
    )


    # Pretrained-C: dense anticipatory swing-foot shaping.
    anticipatory_swing_clearance = RewTerm(
        func=stair_rewards.anticipatory_stair_swing_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "riser_edges_m": (1.0, 1.25, 1.50),
            "tread_heights_m": (0.20, 0.40, 0.60),
            "lookahead_distance_m": 0.50,
            "lip_distance_m": 0.05,
            "clearance_margin_m": 0.05,
            "tanh_mult": 2.0,
        },
    )


    # Pretrained-D: contact-gated, correctly referenced ankle clearance.
    contact_gated_stair_clearance = RewTerm(
        func=stair_rewards.contact_gated_stair_clearance,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            ),
            "riser_edges_m": (1.0, 1.25, 1.50),
            "tread_heights_m": (0.20, 0.40, 0.60),
            "nominal_ankle_height_m": 0.10,
            "lookahead_distance_m": 0.35,
            "lip_distance_m": 0.05,
            "clearance_margin_m": 0.05,
        },
    )


    # 022: terrain-aware dense shaping for normal-start curriculum.
    curriculum_riser_clearance = RewTerm(
        func=stair_rewards.curriculum_riser_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "lookahead_distance_m": 0.45,
            "forward_margin_m": 0.06,
            "clearance_margin_m": 0.025,
            "height_sigma_m": 0.035,
            "minimum_horizontal_speed_mps": 0.05,
        },
    )

    curriculum_foot_over_tread = RewTerm(
        func=stair_rewards.curriculum_foot_over_tread,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "x_margin_m": 0.01,
            "clearance_margin_m": 0.01,
            "upright_min": 0.80,
        },
    )

    pre_riser_foot_lift = RewTerm(
        func=stair_rewards.pre_riser_foot_lift,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "lookahead_m": 0.45,
            "safety_clearance_m": 0.06,
            "min_forward_speed_mps": 0.03,
            "vertical_speed_scale_mps": 0.30,
        },
    )

    distance_target_foot_lift = RewTerm(
        func=stair_rewards.distance_target_foot_lift,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "lookahead_m": 0.45,
            "safety_clearance_m": 0.06,
            "min_forward_speed_mps": 0.03,
            "vertical_speed_scale_mps": 0.30,
        },
    )

    riser_crossing_clearance = RewTerm(
        func=stair_rewards.riser_crossing_clearance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "crossing_before_m": 0.04,
            "crossing_after_m": 0.06,
            "safety_clearance_m": 0.06,
            "min_forward_speed_mps": 0.03,
        },
    )

    riser_face_strike_proxy = RewTerm(
        func=stair_rewards.riser_face_strike_proxy,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "face_window_m": 0.035,
            "minimum_clearance_m": 0.025,
            "min_forward_speed_mps": 0.03,
        },
    )

    validated_tread_contact = RewTerm(
        func=stair_rewards.validated_tread_contact,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            ),
            "first_riser_x_m": 1.0,
            "nominal_ankle_height_m": 0.10,
            "x_margin_m": 0.02,
            "z_tolerance_m": 0.04,
            "upright_min": 0.80,
        },
    )

    base_above_transition = RewTerm(
        func=stair_rewards.halfway_step_transition,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=".*ankle_roll_link"
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll_link"
            ),
            "forbidden_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=(
                    "pelvis|torso_link|.*knee_link|.*elbow_link|"
                    ".*rubber_hand"
                ),
            ),
            "transition": "base_above",
        },
    )
    both_feet_supported_transition = RewTerm(
        func=stair_rewards.halfway_step_transition,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=".*ankle_roll_link"
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll_link"
            ),
            "forbidden_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=(
                    "pelvis|torso_link|.*knee_link|.*elbow_link|"
                    ".*rubber_hand"
                ),
            ),
            "transition": "both_feet_supported",
        },
    )
    completed_transition = RewTerm(
        func=stair_rewards.halfway_step_transition,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=".*ankle_roll_link"
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*ankle_roll_link"
            ),
            "forbidden_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=(
                    "pelvis|torso_link|.*knee_link|.*elbow_link|"
                    ".*rubber_hand"
                ),
            ),
            "transition": "completed",
        },
    )

    platform_entry = RewTerm(
        func=stair_rewards.progress_threshold_event,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold_m": 1.75,
        },
    )
    task_completion = RewTerm(
        func=stair_rewards.progress_threshold_event,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold_m": 3.75,
        },
    )


@configclass
class StairEnvCfg(_UPSTREAM.RobotEnvCfg):
    """Deterministic training configuration for the nominal staircase."""

    scene: InteractiveSceneCfg = StairSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: StairObservationsCfg = StairObservationsCfg()
    actions: StairActionsCfg = StairActionsCfg()
    commands: StairCommandsCfg = StairCommandsCfg()
    rewards: StairRewardsCfg = StairRewardsCfg()
    events: StairEventsCfg = StairEventsCfg()
    curriculum: StairCurriculumCfg = StairCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 18.0


@configclass
class StairPlayEnvCfg(StairEnvCfg):
    """Small rendered configuration for cloud smoke tests and evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 16
