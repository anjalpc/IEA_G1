"""Register the stair task without importing Isaac Lab before AppLauncher."""

import gymnasium as gym

from .contracts import TASK_ID


def register() -> None:
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaac_lab.g1_stairs.env_cfg:StairEnvCfg",
            "play_env_cfg_entry_point": "isaac_lab.g1_stairs.env_cfg:StairPlayEnvCfg",
            "rsl_rl_cfg_entry_point": (
                "isaac_lab.g1_stairs.agent_cfg:StairPPORunnerCfg"
            ),
        },
    )


register()
