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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
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
from . import reference_tracking as reference_rewards
from . import specialist_rewards as stair_rewards
from . import terrain_observations
from . import geometry_locomotion_rewards
from . import geometry_acquisition_rewards


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
            self.last_action.clip = (-5.0, 5.0)
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
            self.last_action.clip = (-5.0, 5.0)
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
    """Active reward surface for specialist and reference-guided training.

    Experiment YAML selects the exact runtime set. Historical numbered reward
    implementations live in the pre-specialist archive and are intentionally
    absent from this active catalog.
    """

    # Optional reference-guided terms remain available for future specialist
    # imitation/tracking experiments. They are disabled unless selected by a
    # reference-tracking config.
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

    # General locomotion terms used by the geometry-conditioned policy.
    # These are copied directly from Unitree's pinned velocity environment so
    # their functions/parameters remain identical to the pretrained policy's
    # native objective. Strict experiment configs select their final weights.
    track_lin_vel_xy = copy.deepcopy(
        _UPSTREAM_CFG.rewards.track_lin_vel_xy
    )
    track_ang_vel_z = copy.deepcopy(
        _UPSTREAM_CFG.rewards.track_ang_vel_z
    )
    track_stair_axis_velocity = RewTerm(
        func=geometry_locomotion_rewards.track_stair_axis_velocity,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_velocity_mps": FIXED_FORWARD_COMMAND_MPS,
            "std_mps": 0.25,
            "command_name": None,
        },
    )
    stair_centerline_error = RewTerm(
        func=geometry_locomotion_rewards.stair_centerline_error,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scale_m": 0.25,
        },
    )
    finish_planar_velocity_error = RewTerm(
        func=geometry_locomotion_rewards.finish_planar_velocity_error,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "cruise_speed_mps": FIXED_FORWARD_COMMAND_MPS,
            "scale_mps": 0.25,
        },
    )
    finish_centerline_error = RewTerm(
        func=geometry_locomotion_rewards.finish_centerline_error,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "cruise_speed_mps": FIXED_FORWARD_COMMAND_MPS,
            "scale_m": 0.25,
        },
    )
    finish_arm_target_return = RewTerm(
        func=geometry_locomotion_rewards.finish_arm_target_return,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    name
                    for name in BODY_JOINTS
                    if any(
                        token in name
                        for token in (
                            "_shoulder_",
                            "_elbow_",
                            "_wrist_",
                        )
                    )
                ],
            ),
            "command_name": "base_velocity",
            "cruise_speed_mps": FIXED_FORWARD_COMMAND_MPS,
        },
    )

    # Specialist foundation regularizers. The resolved YAML strict runtime set
    # selects exactly which of these are manager-visible. Keep the balance
    # terms terrain-compatible: no absolute base-height or vertical-velocity
    # target is used because those would fight the required body rise.
    alive = copy.deepcopy(_UPSTREAM_CFG.rewards.alive)
    upright = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_angular_velocity = copy.deepcopy(
        _UPSTREAM_CFG.rewards.base_angular_velocity
    )
    feet_slide = copy.deepcopy(_UPSTREAM_CFG.rewards.feet_slide)
    joint_acc = copy.deepcopy(_UPSTREAM_CFG.rewards.joint_acc)
    dof_pos_limits = copy.deepcopy(_UPSTREAM_CFG.rewards.dof_pos_limits)
    undesired_contacts = copy.deepcopy(_UPSTREAM_CFG.rewards.undesired_contacts)
    action_rate = copy.deepcopy(_UPSTREAM_CFG.rewards.action_rate)
    failure_termination_event = RewTerm(
        func=stair_rewards.failure_termination_event,
        weight=-1.0,
    )
    geometry_failure_termination_event = RewTerm(
        func=geometry_locomotion_rewards.geometry_failure_termination_event,
        weight=0.0,
    )

    # Geometry-driven specialist task terms.  Parameters remain bounded and
    # config-overridable through reward_parameters in the resolved YAML.
    approach_progress = RewTerm(
        func=stair_rewards.approach_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "preparation_distance_m": 0.18,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    swing_waypoint_progress = RewTerm(
        func=stair_rewards.swing_waypoint_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "swing_left": True,
            "support_force_bodyweight_fraction": 0.40,
            "swing_unload_bodyweight_fraction": 0.20,
            "phase0_bottleneck": False,
            "phase0_require_grounded_swing": False,
            "phase0_swing_height_free_m": 0.010,
            "phase0_swing_height_zero_m": 0.030,
            "pelvis_support_tolerance_m": 0.09,
            "pelvis_support_zero_m": 0.20,
            "support_anchor_tolerance_m": 0.015,
            "support_anchor_zero_m": 0.070,
            "early_x_fraction": 0.35,
            "clearance_x_fraction": 0.60,
            "early_lift_riser_fraction": 0.55,
            "clearance_margin_m": 0.025,
            "hover_margin_m": 0.025,
            "waypoint_x_tolerance_m": 0.030,
            "waypoint_x_zero_m": 0.120,
            "waypoint_z_tolerance_m": 0.015,
            "waypoint_z_zero_m": 0.055,
            "lateral_lane_tolerance_m": 0.030,
            "lateral_lane_zero_m": 0.080,
            "foot_roll_tolerance_rad": 0.12,
            "foot_roll_zero_rad": 0.35,
            "foot_yaw_tolerance_rad": 0.12,
            "foot_yaw_zero_rad": 0.35,
            "root_roll_tolerance_rad": 0.10,
            "root_roll_zero_rad": 0.30,
            "root_pitch_tolerance_rad": 0.20,
            "root_pitch_zero_rad": 0.45,
            "root_yaw_tolerance_rad": 0.10,
            "root_yaw_zero_rad": 0.30,
            "maximum_lower_support_foot_speed_mps": 0.20,
            "transition_dwell_s": 0.06,
            "touchdown_dwell_s": 0.08,
            "maximum_touchdown_vertical_speed_mps": 0.25,
            "reversible_touchdown_credit": False,
            "continuous_touchdown_credit": False,
            "touchdown_commitment_full_score": 0.75,
            "touchdown_commitment_zero_score": 0.25,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "upper_support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    swing_body_advance_progress = RewTerm(
        func=stair_rewards.swing_body_advance_progress,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "swing_left": True,
            "target_forward_from_support_m": 0.080,
            "clearance_x_fraction": 0.60,
            "clearance_x_tolerance_m": 0.040,
            "minimum_clearance_above_riser_m": 0.010,
            "support_anchor_limit_m": 0.050,
            "support_anchor_xy": False,
            "support_force_bodyweight_fraction": 0.25,
            "swing_unload_bodyweight_fraction": 0.25,
            "require_live_touchdown_commitment": False,
            "touchdown_containment_minimum": 0.75,
            "upper_support_minimum": 0.60,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "upper_support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    support_foot_anchor_guard = RewTerm(
        func=stair_rewards.support_foot_anchor_guard,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "swing_left": True,
            "support_force_bodyweight_fraction": 0.25,
            "swing_unload_bodyweight_fraction": 0.30,
            "anchor_free_m": 0.015,
            "anchor_zero_m": 0.060,
            "speed_free_mps": 0.020,
            "speed_zero_mps": 0.20,
            "active_phase_min": 1,
            "active_phase_max": 4,
            "anchor_phase_gate_only": False,
            "latch_anchor_on_active_phase": False,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "upper_support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    swing_waypoint_guard = RewTerm(
        func=stair_rewards.swing_waypoint_guard,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "swing_left": True,
            "enforce_phase0_support": False,
            "phase0_swing_load_free_bodyweight_fraction": 0.30,
            "phase0_swing_load_zero_bodyweight_fraction": 0.50,
            "phase0_swing_height_free_m": 0.012,
            "phase0_swing_height_zero_m": 0.035,
            "phase0_support_speed_free_mps": 0.15,
            "phase0_support_speed_zero_mps": 0.50,
            "lateral_free_m": 0.030,
            "lateral_zero_m": 0.100,
            "support_anchor_free_m": 0.020,
            "support_anchor_zero_m": 0.080,
            "maximum_height_above_riser_m": 0.070,
            "overheight_zero_m": 0.180,
            "backward_free_m": 0.020,
            "backward_zero_m": 0.100,
            "foot_roll_free_rad": 0.15,
            "foot_roll_zero_rad": 0.45,
            "foot_yaw_free_rad": 0.15,
            "foot_yaw_zero_rad": 0.45,
            "root_roll_free_rad": 0.12,
            "root_roll_zero_rad": 0.35,
            "root_pitch_free_rad": 0.25,
            "root_pitch_zero_rad": 0.55,
            "pelvis_support_free_m": 0.11,
            "pelvis_support_zero_m": 0.24,
            "support_force_bodyweight_fraction": 0.25,
            "lower_floor_recontact_force_bodyweight_fraction": 0.12,
            "maximum_swing_speed_mps": 1.20,
            "swing_speed_zero_mps": 2.20,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "upper_support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    posture_envelope_guard = RewTerm(
        func=stair_rewards.posture_envelope_guard,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "arms_free_rms_rad": 0.25,
            "arms_zero_rms_rad": 0.70,
            "arms_only": False,

            # V08-A3 opt-in continuous nominal-relative arm shaping.
            # Defaults preserve all historical posture-envelope behavior.
            "joint_specific_excess_squared": False,
            "joint_specific_target_excess_squared": False,
            "phase_specific_target_excess_squared": False,
            "phase_specific_family_target_excess_squared": False,
            "phase_shoulder_pitch_approach_tolerance_rad": 0.20,
            "phase_shoulder_pitch_ascent_tolerance_rad": 0.35,
            "phase_shoulder_pitch_safe_platform_tolerance_rad": 0.20,
            "phase_shoulder_roll_approach_tolerance_rad": 0.30,
            "phase_shoulder_roll_ascent_tolerance_rad": 0.40,
            "phase_shoulder_roll_safe_platform_tolerance_rad": 0.25,
            "phase_shoulder_yaw_approach_tolerance_rad": 0.10,
            "phase_shoulder_yaw_ascent_tolerance_rad": 0.20,
            "phase_shoulder_yaw_safe_platform_tolerance_rad": 0.10,
            "phase_elbow_approach_tolerance_rad": 0.35,
            "phase_elbow_ascent_tolerance_rad": 0.45,
            "phase_elbow_safe_platform_tolerance_rad": 0.25,
            "phase_wrist_approach_tolerance_rad": 0.15,
            "phase_wrist_ascent_tolerance_rad": 0.15,
            "phase_wrist_safe_platform_tolerance_rad": 0.15,
            "phase_command_name": "base_velocity",
            "phase_cruise_speed_mps": FIXED_FORWARD_COMMAND_MPS,
            "shoulder_pitch_tolerance_rad": 0.45,
            "shoulder_pitch_relative_weight": 0.25,
            "shoulder_roll_tolerance_rad": 0.40,
            "shoulder_roll_relative_weight": 0.50,
            "shoulder_yaw_tolerance_rad": 0.25,
            "shoulder_yaw_relative_weight": 1.50,
            "elbow_tolerance_rad": 0.45,
            "elbow_relative_weight": 0.25,
            "wrist_tolerance_rad": 0.25,
            "wrist_relative_weight": 2.00,

            # A6 opt-in phase-gated processed-target posture shaping.
            "a6_phase_gated_target_excess_squared": False,
            "a6_platform_progress_distance_m": 2.0,
            "a6_cruise_speed_mps": 0.35,
            "a6_weak_scale": 0.25,
            "a6_platform_gain": 0.75,
            "a6_finish_gain": 1.0,
            "a6_maximum_phase_scale": 2.0,
            "a6_command_name": "base_velocity",
            "waist_free_rms_rad": 0.15,
            "waist_zero_rms_rad": 0.45,
            "lateral_legs_free_rms_rad": 0.22,
            "lateral_legs_zero_rms_rad": 0.55,
        },
    )
    swing_target_progress = RewTerm(
        func=stair_rewards.swing_target_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "swing_left": True,
            "support_force_bodyweight_fraction": 0.45,
            "swing_unload_bodyweight_fraction": 0.20,
            "pelvis_support_tolerance_m": 0.08,
            "pelvis_support_zero_m": 0.18,
            "lift_clearance_m": 0.030,
            "lateral_lane_tolerance_m": 0.030,
            "lateral_lane_zero_m": 0.080,
            "foot_roll_tolerance_rad": 0.12,
            "foot_roll_zero_rad": 0.35,
            "foot_yaw_tolerance_rad": 0.12,
            "foot_yaw_zero_rad": 0.35,
            "root_roll_tolerance_rad": 0.10,
            "root_roll_zero_rad": 0.30,
            "root_pitch_tolerance_rad": 0.20,
            "root_pitch_zero_rad": 0.45,
            "root_yaw_tolerance_rad": 0.10,
            "root_yaw_zero_rad": 0.30,
            "maximum_lower_support_foot_speed_mps": 0.20,
            "transition_dwell_s": 0.06,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "upper_support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    swing_motion_guard = RewTerm(
        func=stair_rewards.swing_motion_guard,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "swing_left": True,
            "lateral_free_m": 0.030,
            "lateral_zero_m": 0.100,
            "maximum_height_above_riser_m": 0.070,
            "overheight_zero_m": 0.180,
            "backward_free_m": 0.020,
            "backward_zero_m": 0.100,
            "foot_roll_free_rad": 0.15,
            "foot_roll_zero_rad": 0.45,
            "foot_yaw_free_rad": 0.15,
            "foot_yaw_zero_rad": 0.45,
            "pelvis_support_free_m": 0.10,
            "pelvis_support_zero_m": 0.22,
            "support_force_bodyweight_fraction": 0.25,
            "activation_height_m": 0.010,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "upper_support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    foot_yaw_excess_guard = RewTerm(
        func=stair_rewards.foot_yaw_excess_guard,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*ankle_roll.*",
            ),
            "free_yaw_rad": 0.2617993877991494,
            "zero_at_yaw_rad": 0.4363323129985824,
        },
    )

    body_heading_guard = RewTerm(
        func=stair_rewards.body_heading_guard,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "free_yaw_rad": 0.10,
            "zero_at_yaw_rad": 0.35,
        },
    )

    swing_lift_progress = RewTerm(
        func=stair_rewards.swing_lift_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "lower_support_height_tolerance_m": 0.020,
            "lower_support_force_bodyweight_fraction": 0.30,
            "maximum_lower_support_foot_speed_mps": 0.20,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    swing_clearance_progress = RewTerm(
        func=stair_rewards.swing_clearance_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "activation_distance_m": 0.30,
            "insertion_target_m": 0.10,
            "clearance_margin_m": 0.015,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    upper_foothold_progress = RewTerm(
        func=stair_rewards.upper_foothold_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    body_transfer_progress = RewTerm(
        func=stair_rewards.body_transfer_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "transfer_start_before_riser_m": 0.12,
            "transfer_target_past_riser_m": 0.10,
            "required_height_fraction": 0.80,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    trailing_foot_recovery = RewTerm(
        func=stair_rewards.trailing_foot_recovery,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "required_height_fraction": 0.70,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    riser_strike_event = RewTerm(
        func=stair_rewards.riser_strike_event,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "face_band_m": 0.025,
            "clearance_margin_m": 0.010,
            "horizontal_force_bodyweight_fraction": 0.05,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
            "minimum_upright_gravity_z": 0.80,
        },
    )
    single_step_success = RewTerm(
        func=stair_rewards.single_step_success,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "minimum_support_score": 0.60,
            "base_past_riser_margin_m": 0.05,
            "required_height_fraction": 0.75,
            "support_minimum_upright_gravity_z": 0.80,
            "minimum_upright_gravity_z": 0.90,
            "maximum_root_linear_speed_mps": 0.35,
            "maximum_root_angular_speed_radps": 0.60,
            "minimum_dwell_s": 0.12,
            "tread_margin_m": 0.015,
            "foot_height_tolerance_m": 0.025,
            "support_force_bodyweight_fraction": 0.15,
            "maximum_supported_foot_speed_mps": 0.15,
        },
    )


    # V04 geometry-conditioned stair acquisition.
    # These terms remain inactive unless selected by an experiment YAML.
    terrain_relative_clearance = RewTerm(
        func=geometry_acquisition_rewards.terrain_relative_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "activation_distance_m": 0.30,
            "clearance_margin_m": 0.015,
            "tread_margin_m": 0.015,
            "insertion_target_m": 0.040,
            "unload_full_force_n": 15.0,
            "unload_zero_force_n": 80.0,
            "opposite_support_start_n": 30.0,
            "opposite_support_full_n": 100.0,
            "insertion_clearance_fraction": 0.70,
            "touchdown_force_n": 30.0,
            "touchdown_speed_mps": 0.20,
            "touchdown_height_tolerance_m": 0.025,
            "diagnostic_contact_threshold_n": 1.0,
            "suppress_riser_face_clearance_credit": False,
            "escrow_clearance_until_riser_crossing": False,
            "riser_face_band_m": 0.025,
            "riser_below_top_margin_m": 0.005,
            "riser_minimum_horizontal_force_n": 30.0,
            "riser_minimum_previous_forward_speed_mps": 0.10,
        },
    )

    forward_insertion = RewTerm(
        func=geometry_acquisition_rewards.forward_insertion,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "activation_distance_m": 0.30,
            "clearance_margin_m": 0.015,
            "tread_margin_m": 0.015,
            "insertion_target_m": 0.040,
            "unload_full_force_n": 15.0,
            "unload_zero_force_n": 80.0,
            "opposite_support_start_n": 30.0,
            "opposite_support_full_n": 100.0,
            "insertion_clearance_fraction": 0.70,
            "touchdown_force_n": 30.0,
            "touchdown_speed_mps": 0.20,
            "touchdown_height_tolerance_m": 0.025,
            "diagnostic_contact_threshold_n": 1.0,
            "suppress_riser_face_clearance_credit": False,
            "escrow_clearance_until_riser_crossing": False,
            "riser_face_band_m": 0.025,
            "riser_below_top_margin_m": 0.005,
            "riser_minimum_horizontal_force_n": 30.0,
            "riser_minimum_previous_forward_speed_mps": 0.10,
        },
    )

    first_touchdown_placement = RewTerm(
        func=geometry_acquisition_rewards.first_touchdown_placement,
        weight=0.0,
        params={},
    )

    anticipatory_clearance_progress = RewTerm(
        func=geometry_acquisition_rewards.anticipatory_clearance_progress,
        weight=0.0,
        params={
            "start_distance_m": 0.22,
            "full_clearance_distance_m": 0.05,
            "clearance_margin_m": 0.015,
            "support_advantage_start_n": 0.0,
            "support_advantage_full_n": 60.0,
            "max_credit_vertical_speed_mps": 0.60,
        },
    )

    nonpreferred_stair_acquisition = RewTerm(
        func=geometry_acquisition_rewards.nonpreferred_stair_acquisition,
        weight=0.0,
        params={},
    )

    preferred_old_tread_replant = RewTerm(
        func=geometry_acquisition_rewards.preferred_old_tread_replant,
        weight=0.0,
        params={
            "unload_force_n": 15.0,
            "touchdown_force_n": 30.0,
            "lift_clearance_m": 0.030,
            "top_height_tolerance_m": 0.025,
        },
    )

    preferred_transition_milestones = RewTerm(
        func=geometry_acquisition_rewards.preferred_transition_milestones,
        weight=0.0,
        params={
            "unload_force_n": 15.0,
            "opposite_support_force_n": 30.0,
            "touchdown_force_n": 30.0,
            "lift_clearance_m": 0.030,
            "riser_clearance_margin_m": 0.015,
            "top_height_tolerance_m": 0.025,
            "lift_value": 0.15,
            "clearance_value": 0.35,
            "success_value": 1.50,
            "replant_value": -0.10,
            "wrong_foot_value": -0.10,
            "unresolved_end_value": -0.10,
        },
    )

    acquisition_step_over_transition = RewTerm(
        func=geometry_acquisition_rewards.acquisition_step_over_transition,
        weight=0.0,
        params={
            "unload_force_n": 15.0,
            "touchdown_force_n": 30.0,
            "lift_clearance_m": 0.030,
            "top_height_tolerance_m": 0.025,
            "success_value": 0.50,
            "replant_value": -0.25,
            "wrong_foot_value": -0.25,
        },
    )

    step_over_transition_contract = RewTerm(
        func=geometry_acquisition_rewards.step_over_transition_contract,
        weight=0.0,
        params={
            "first_commit_distance_m": 0.30,
            "unload_force_n": 15.0,
            "opposite_support_force_n": 30.0,
            "touchdown_force_n": 30.0,
            "lift_clearance_m": 0.030,
            "riser_clearance_margin_m": 0.015,
            "clearance_window_m": 0.050,
            "top_height_tolerance_m": 0.025,
            "max_forward_excursion_m": 0.40,
            "strike_minimum_force_n": 30.0,
            "strike_full_severity_force_n": 600.0,
            "strike_minimum_event_fraction": 0.10,
            "success_value": 1.0,
            "failure_value": -1.0,
            "lift_milestone_value": 0.20,
            "clearance_milestone_value": 0.30,
            "vertical_clearance_progress_value": 0.0,
            "vertical_clearance_discount_gamma": 0.99,
            "insertion_milestone_value": 0.20,
            "strike_failure_base": -1.0,
            "strike_failure_severity_scale": -0.5,
        },
    )

    riser_face_strike = RewTerm(
        func=geometry_acquisition_rewards.riser_face_strike,
        weight=0.0,
        params={
            "face_band_m": 0.025,
            "below_top_margin_m": 0.005,
            "minimum_horizontal_force_n": 30.0,
            "minimum_previous_forward_speed_mps": 0.10,
            "severity_enabled": False,
            "full_severity_force_n": 250.0,
            "minimum_event_fraction": 0.25,
        },
    )

    knee_forward_velocity_guard = RewTerm(
        func=geometry_acquisition_rewards.knee_forward_velocity_guard,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces"),
            "activation_distance_m": 0.30,
            "clearance_margin_m": 0.015,
            "tread_margin_m": 0.015,
            "insertion_target_m": 0.040,
            "unload_full_force_n": 15.0,
            "unload_zero_force_n": 80.0,
            "opposite_support_start_n": 30.0,
            "opposite_support_full_n": 100.0,
            "insertion_clearance_fraction": 0.70,
            "touchdown_force_n": 30.0,
            "touchdown_speed_mps": 0.20,
            "touchdown_height_tolerance_m": 0.025,
            "diagnostic_contact_threshold_n": 1.0,
            "post_riser_distance_m": 0.12,
            "high_riser_start_m": 0.14,
            "high_riser_full_m": 0.18,
            "free_forward_velocity_mps": 0.20,
            "full_penalty_forward_velocity_mps": 0.70,
        },
    )

    stable_platform_finish = RewTerm(
        func=geometry_locomotion_rewards.stable_platform_finish,
        weight=0.0,
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
