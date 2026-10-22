"""Opt-in platform finish command. Import only after AppLauncher."""
import math
import torch
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from .command_runtime import platform_finish_speed
from .contracts import G1_SOLE_SPHERE_CENTRES_M, G1_SOLE_SPHERE_RADIUS_M
from .geometry_episode import EpisodeContract
from .terrain_state import stair_cfg, riser_height_per_env


class PlatformFinishCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.finish = EpisodeContract(**cfg.platform_finish_contract)
        self.options = cfg.platform_finish_options
        self.foot_ids = [
            self.robot.body_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        self.sensor = env.scene.sensors["contact_forces"]
        self.sensor_ids = [
            self.sensor.body_names.index(name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        with torch.inference_mode(False):
            self.settling = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device,
            )
            self.sole_centres = torch.tensor(
                G1_SOLE_SPHERE_CENTRES_M, device=env.device,
            )

    def _update_command(self):
        super()._update_command()
        env = self._env
        data = self.robot.data
        terrain = stair_cfg(env)
        platform_start = terrain.start_clearance + terrain.step_count * terrain.tread_depth
        completion_x = platform_start + self.finish.platform_continuation_m
        stop_x = completion_x + self.options["stop_buffer_m"]
        local_x = data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]

        feet = data.body_pos_w[:, self.foot_ids] - env.scene.env_origins[:, None, :]
        quat = data.body_quat_w[:, self.foot_ids]
        offsets = self.sole_centres[None, None].expand(env.num_envs, 2, -1, -1)
        qvec = quat[:, :, None, 1:].expand_as(offsets)
        cross = 2 * torch.cross(qvec, offsets, dim=-1)
        points = feet[:, :, None, :] + offsets + quat[:, :, None, :1] * cross
        points = points + torch.cross(qvec, cross, dim=-1)

        radius = G1_SOLE_SPHERE_RADIUS_M
        margin = self.finish.finish_sole_margin_m
        course_end = terrain.size[0] - (terrain.approach_length - terrain.start_clearance)
        half_width = terrain.size[1] / 2
        platform_z = terrain.step_count * riser_height_per_env(env)

        foot_safe = (
            (points[..., 0].amin(-1) - radius >= platform_start + margin)
            & (points[..., 0].amax(-1) + radius
               <= course_end - self.finish.boundary_margin_m)
            & (points[..., 1].amin(-1) - radius >= -half_width + margin)
            & (points[..., 1].amax(-1) + radius <= half_width - margin)
            & ((points[..., 2] - radius - platform_z[:, None, None]).abs().amax(-1)
               <= self.finish.sole_height_tolerance_m)
            & (self.sensor.data.net_forces_w[:, self.sensor_ids, 2]
               >= self.finish.support_force_n)
        )
        both_safe = foot_safe.all(-1)
        reset = env.episode_length_buf == 0

        cruise = float(self.options["cruise_speed_mps"])
        speed, self.settling = platform_finish_speed(
            local_x,
            stop_x,
            cruise,
            both_safe,
            self.settling,
            reset,
            slowdown_distance=float(self.options["slowdown_distance_m"]),
            creep_speed=float(self.options["creep_speed_mps"]),
            resume_x=completion_x,
        )

        self.vel_command_b[:, 0] = speed
        self.vel_command_b[:, 1] = 0.0


def apply_platform_finish_command(env_cfg, resolved, contract):
    if not contract["stable_finish"]:
        return
    command = resolved.get("command", {})
    if command.get("stair_velocity_schedule", {}).get("enabled", False):
        raise ValueError("stable finish cannot coexist with the old velocity schedule")
    base = env_cfg.commands.base_velocity
    if base.class_type not in (UniformVelocityCommand, PlatformFinishCommand):
        raise ValueError("Unsupported base command class for stable finish")

    lower, upper = base.ranges.lin_vel_x
    if lower != upper:
        raise ValueError("stable finish currently requires one fixed cruise speed")
    options = {
        "cruise_speed_mps": float(lower),
        "slowdown_distance_m": 0.35,
        "creep_speed_mps": 0.08,
        "stop_buffer_m": 0.10,
        "transition_duration_s": 0.40,
        "minimum_step_advance_m": 0.15,
    }
    overrides = command.get("platform_finish", {}) or {}
    unknown = set(overrides) - (set(options) - {"cruise_speed_mps"})
    if unknown:
        raise ValueError(f"Unknown platform_finish parameters: {sorted(unknown)}")
    options.update(overrides)
    for name, value in options.items():
        if not math.isfinite(value) or (
            value < 0 if name == "stop_buffer_m" else value <= 0
        ):
            raise ValueError(
                "platform_finish parameters must be finite and positive; "
                "stop_buffer_m may be zero"
            )
    if not options["creep_speed_mps"] <= options["cruise_speed_mps"] <= 0.50:
        raise ValueError("require creep <= cruise <= 0.50")

    terrain = env_cfg.scene.terrain.terrain_generator.sub_terrains["three_step_ascent"]
    platform_start = terrain.start_clearance + terrain.step_count * terrain.tread_depth
    course_end = terrain.size[0] - (terrain.approach_length - terrain.start_clearance)
    stop_x = platform_start + contract["platform_continuation_m"] + options["stop_buffer_m"]
    if stop_x >= course_end - contract["boundary_margin_m"]:
        raise ValueError("platform stop target falls outside usable course")

    term = getattr(env_cfg.rewards, "track_stair_axis_velocity", None)
    if term is None:
        raise ValueError("stable finish requires track_stair_axis_velocity")
    term.params = dict(term.params, command_name="base_velocity")
    base.class_type = PlatformFinishCommand
    base.platform_finish_contract = dict(contract)
    base.platform_finish_options = options
    base.rel_standing_envs = 0.0
