"""Isaac Lab command term for flat + stair mixing (see footstep_mixing.py)."""
from __future__ import annotations

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from .footstep_mixing import apply_stair_forward_commands
from .terrain_state import riser_height_per_env


class TerrainSplitVelocityCommand(UniformVelocityCommand):
    """Flat envs: cfg.ranges (joystick). Stair envs: forward-only speed."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        apply_stair_forward_commands(
            self.vel_command_b, self.is_standing_env, env_ids,
            riser_height_per_env(self._env) > 0, self.cfg.stair_lin_vel_x,
        )


@configclass
class TerrainSplitVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TerrainSplitVelocityCommand
    stair_lin_vel_x: tuple[float, float] = (0.2, 0.5)


def build_terrain_split_command_cfg(base, split):
    """Replace the base velocity command; ``split`` is validate_terrain_split output."""
    flat = split["flat"]
    return TerrainSplitVelocityCommandCfg(
        asset_name=base.asset_name,
        resampling_time_range=base.resampling_time_range,
        heading_command=False,
        heading_control_stiffness=base.heading_control_stiffness,
        rel_standing_envs=flat["rel_standing_envs"],
        rel_heading_envs=base.rel_heading_envs,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=flat["lin_vel_x"],
            lin_vel_y=flat["lin_vel_y"],
            ang_vel_z=flat["ang_vel_z"],
            heading=base.ranges.heading,
        ),
        stair_lin_vel_x=split["stairs"]["lin_vel_x"],
    )
