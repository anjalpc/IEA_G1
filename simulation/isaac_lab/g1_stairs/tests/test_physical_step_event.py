import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import types

import torch

from simulation.isaac_lab.g1_stairs.geometry_episode import Result

managers = types.ModuleType("isaaclab.managers")


class ManagerTermBase:
    def __init__(self, cfg, env):
        pass


class SceneEntityCfg:
    def __init__(self, name, *args, **kwargs):
        self.name = name


managers.ManagerTermBase = ManagerTermBase
managers.SceneEntityCfg = SceneEntityCfg
with patch.dict("sys.modules", {"isaaclab.managers": managers}):
    from simulation.isaac_lab.g1_stairs.specialist_rewards import (
        physical_step_event,
        top_platform_success_event,
    )


class Scene(dict):
    pass


class PhysicalStepEventTests(unittest.TestCase):
    def setUp(self):
        self.term = physical_step_event.__new__(physical_step_event)
        self.term.active_foot = torch.tensor([-1])
        self.term.last_completed_foot = torch.tensor([-1])
        self.term.start_x = torch.zeros(1)
        self.term.start_z = torch.zeros(1)
        self.term.apex_z = torch.zeros(1)
        self.term.best_touchdown_x = torch.full((1,), -torch.inf)

        data = NS(
            body_pos_w=torch.tensor(
                [[[1.10, 0.1, 0.05], [0.90, -0.1, 0.05]]]
            ),
            body_lin_vel_w=torch.zeros(1, 2, 3),
        )
        self.forces = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]]]
        )
        scene = Scene(robot=NS(data=data))
        scene.sensors = {"contact": NS(data=NS(net_forces_w=self.forces))}
        scene.env_origins = torch.zeros(1, 3)
        self.env = NS(num_envs=1, device="cpu", step_dt=0.02, scene=scene)
        self.asset = NS(name="robot", body_ids=[0, 1])
        self.sensor = NS(name="contact", body_ids=[0, 1])
        self.terrain = NS(
            start_clearance=1.0, step_count=3, tread_depth=0.25
        )

    def call(self):
        with patch(
            "simulation.isaac_lab.g1_stairs.specialist_rewards.stair_cfg",
            return_value=self.terrain,
        ):
            return self.term(
                self.env,
                minimum_forward_travel_m=0.15,
                minimum_vertical_excursion_m=0.015,
                maximum_touchdown_speed_mps=0.60,
                support_force_n=80.0,
                unload_force_n=20.0,
                approach_scale=0.10,
                ascent_scale=1.0,
                platform_scale=0.10,
                minimum_new_progress_m=0.05,
                asset_cfg=self.asset,
                contact_sensor_cfg=self.sensor,
            )

    def test_valid_ascent_step_pays_once_and_reset_clears_frontier(self):
        torch.testing.assert_close(self.call(), torch.tensor([0.0]))
        self.env.scene["robot"].data.body_pos_w[0, 0, [0, 2]] = torch.tensor(
            [1.30, 0.07]
        )
        torch.testing.assert_close(self.call(), torch.tensor([0.0]))
        self.forces[0, 0, 2] = 100.0
        torch.testing.assert_close(self.call(), torch.tensor([50.0]))
        torch.testing.assert_close(self.call(), torch.tensor([0.0]))

        self.term.reset(torch.tensor([0]))
        self.assertEqual(self.term.active_foot.item(), -1)
        self.assertEqual(self.term.last_completed_foot.item(), -1)
        self.assertFalse(torch.isfinite(self.term.best_touchdown_x).item())


class TopPlatformSuccessEventTests(unittest.TestCase):
    def test_success_pays_once_and_reset_rearms(self):
        term = top_platform_success_event.__new__(top_platform_success_event)
        term.paid = torch.zeros(1, dtype=torch.bool)
        env = NS(
            num_envs=1,
            device="cpu",
            step_dt=0.02,
            _g1_geometry_episode=NS(
                contract=NS(stable_finish=True),
                tracker=NS(reason=torch.tensor([Result.SUCCESS])),
            ),
        )
        torch.testing.assert_close(term(env), torch.tensor([50.0]))
        torch.testing.assert_close(term(env), torch.tensor([0.0]))
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term(env), torch.tensor([50.0]))


if __name__ == "__main__":
    unittest.main()
