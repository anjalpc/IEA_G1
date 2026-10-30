"""Flat + stair mixing: config validation, stair commands, safe region, no-plan rows."""
import unittest

import torch

from simulation.isaac_lab.g1_stairs.footstep_mixing import (
    apply_stair_forward_commands, course_exit_timeout,
    outside_safe_region, stair_escape, target_stalled, validate_terrain_split,
)
from simulation.isaac_lab.g1_stairs.footstep_tensor import planned_target_batch
from simulation.isaac_lab.g1_stairs.footstep_tread import qualified_tread_mask

SPLIT = {
    "flat": {"lin_vel_x": [-0.5, 1.0], "lin_vel_y": [-0.3, 0.3],
             "ang_vel_z": [-0.2, 0.2], "rel_standing_envs": 0.02},
    "stairs": {"lin_vel_x": [0.2, 0.5]},
}


class SplitConfigTests(unittest.TestCase):
    def test_valid(self):
        out = validate_terrain_split(SPLIT)
        self.assertEqual(out["flat"]["lin_vel_x"], (-0.5, 1.0))
        self.assertEqual(out["stairs"]["lin_vel_x"], (0.2, 0.5))
        self.assertEqual(out["flat"]["rel_standing_envs"], 0.02)

    def test_flat_outside_unitree_limits_rejected(self):
        bad = {**SPLIT, "flat": {**SPLIT["flat"], "lin_vel_x": [-0.5, 1.5]}}
        with self.assertRaises(ValueError):
            validate_terrain_split(bad)

    def test_stair_speed_bounds(self):
        for rng in ([0.0, 0.5], [0.3, 0.6], [0.4, 0.3]):
            with self.assertRaises(ValueError):
                validate_terrain_split({**SPLIT, "stairs": {"lin_vel_x": rng}})


class StairCommandTests(unittest.TestCase):
    def test_only_stair_envs_in_ids_are_overwritten(self):
        cmd = torch.tensor([[0.9, 0.2, 0.1]] * 4)
        standing = torch.tensor([True, True, True, False])
        on_stairs = torch.tensor([True, False, True, True])
        ids = apply_stair_forward_commands(cmd, standing, torch.tensor([0, 1, 3]),
                                           on_stairs, (0.2, 0.5))
        self.assertEqual(ids.tolist(), [0, 3])
        for i in (0, 3):
            self.assertTrue(0.2 <= cmd[i, 0].item() <= 0.5)
            self.assertEqual(cmd[i, 1:].tolist(), [0.0, 0.0])
            self.assertFalse(bool(standing[i]))
        # Flat env 1 and stair env 2 (not resampled) untouched.
        self.assertTrue(torch.allclose(cmd[1], torch.tensor([0.9, 0.2, 0.1])))
        self.assertTrue(torch.allclose(cmd[2], torch.tensor([0.9, 0.2, 0.1])))
        self.assertTrue(bool(standing[1]) and bool(standing[2]))

    def test_accepts_list_ids(self):
        cmd = torch.zeros(2, 3)
        ids = apply_stair_forward_commands(cmd, torch.zeros(2, dtype=torch.bool),
                                           [1], torch.tensor([True, True]), (0.3, 0.3))
        self.assertEqual(ids.tolist(), [1])
        self.assertAlmostEqual(cmd[1, 0].item(), 0.3, places=6)


class SafeRegionTests(unittest.TestCase):
    def test_bounds(self):
        xy = torch.tensor([[0.0, 0.0], [4.6, 0.0], [-1.6, 0.0], [1.0, 1.6], [4.4, -1.4]])
        self.assertEqual(outside_safe_region(xy).tolist(), [False, True, True, True, False])


class StairEscapeTests(unittest.TestCase):
    def test_stair_envs_fail_when_leaving_corridor_or_turning_away(self):
        xy = torch.tensor([[0.5, 0.0], [0.5, 0.7], [-0.6, 0.0], [0.5, 0.0], [4.6, 0.0]])
        yaw = torch.tensor([0.0, 0.0, 0.0, 1.2, 0.0])
        stairs = torch.ones(5, dtype=torch.bool)
        self.assertEqual(stair_escape(xy, yaw, stairs).tolist(),
                         [False, True, True, True, False])
        # Past the platform end on stairs is a time-out, not an escape.
        self.assertEqual(course_exit_timeout(xy, yaw, stairs).tolist(),
                         [False, False, False, False, True])

    def test_flat_envs_never_escape_only_time_out(self):
        xy = torch.tensor([[0.5, 0.7], [0.5, 1.6], [-1.6, 0.0]])
        yaw = torch.tensor([2.0, 0.0, 3.0])
        flat = torch.zeros(3, dtype=torch.bool)
        self.assertEqual(stair_escape(xy, yaw, flat).tolist(), [False, False, False])
        self.assertEqual(course_exit_timeout(xy, yaw, flat).tolist(), [False, True, True])

    def test_stair_escape_and_timeout_are_exclusive(self):
        xy = torch.tensor([[-1.6, 1.6]])
        yaw = torch.zeros(1)
        stairs = torch.ones(1, dtype=torch.bool)
        self.assertTrue(bool(stair_escape(xy, yaw, stairs)[0]))
        self.assertFalse(bool(course_exit_timeout(xy, yaw, stairs)[0]))


class TargetStallTests(unittest.TestCase):
    def test_only_active_stair_plans_stall_after_limit(self):
        since = torch.tensor([4.1, 4.1, 4.1, 3.9])
        active = torch.tensor([True, False, True, True])
        stairs = torch.tensor([True, True, False, True])
        # env1: plan complete (standing); env2: flat; env3: under the limit.
        self.assertEqual(target_stalled(since, active, stairs).tolist(),
                         [True, False, False, False])


class NoPlanOnFlatRowsTests(unittest.TestCase):
    def test_zero_riser_rows_have_no_targets(self):
        t, f, v = planned_target_batch(
            first_riser_x=torch.ones(2), tread_depth=torch.full((2,), .25),
            riser_height=torch.tensor([0.0, 0.04]), step_count=3,
            first_foot=torch.zeros(2, dtype=torch.long),
            target_index=torch.zeros(2, dtype=torch.long), centre_y=torch.zeros(2),
            foot_separation=torch.full((2,), .235), approach_steps=2)
        self.assertEqual(v.tolist(), [[False, False], [True, True]])
        self.assertEqual(t[0].abs().sum().item(), 0.0)

    def test_negative_riser_still_rejected(self):
        with self.assertRaises(ValueError):
            planned_target_batch(
                first_riser_x=torch.ones(1), tread_depth=torch.full((1,), .25),
                riser_height=torch.tensor([-0.01]), step_count=3,
                first_foot=torch.zeros(1, dtype=torch.long),
                target_index=torch.zeros(1, dtype=torch.long), centre_y=torch.zeros(1),
                foot_separation=torch.full((1,), .235))

    def test_zero_riser_rows_never_qualify(self):
        targets = torch.tensor([[[0.55, .1175, 0., 0.], [0.80, -.1175, 0., 0.]]] * 2)
        bounds = torch.tensor([[[0.47, 0.64, .06, .17, 0., 0.], [0., .1, -.17, -.06, 0., 0.]]] * 2)
        m = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=torch.full((2, 2), 110.),
            targets=targets, feet=torch.tensor([[0, 1]] * 2),
            valid=torch.ones((2, 2), dtype=torch.bool), first_riser_x=torch.ones(2),
            tread_depth=torch.full((2,), .25), riser_height=torch.tensor([0.0, 0.04]),
            step_count=3, half_width=.6, approach_steps=2)
        self.assertEqual(m.tolist(), [[False, False], [True, False]])


if __name__ == "__main__":
    unittest.main()


class StairSteeringTests(unittest.TestCase):
    def test_steers_back_within_unitree_ranges(self):
        from simulation.isaac_lab.g1_stairs.footstep_mixing import stair_steering
        y = torch.tensor([0.0, 0.1, -0.5])
        yaw = torch.tensor([0.0, 0.35, -0.05])
        out = stair_steering(y, yaw)
        self.assertEqual(out[0].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(out[1, 0].item(), -0.1, places=6)
        self.assertAlmostEqual(out[1, 1].item(), -0.2, places=6)   # clipped
        self.assertAlmostEqual(out[2, 0].item(), 0.2, places=6)    # clipped
        self.assertAlmostEqual(out[2, 1].item(), 0.05, places=6)

