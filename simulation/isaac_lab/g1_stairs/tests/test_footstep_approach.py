"""Approach targets before the first riser, plan length, and post-plan stop."""
import unittest
from types import SimpleNamespace as NS

import torch

from simulation.isaac_lab.g1_stairs.footstep_tensor import (
    APPROACH_LAST_GAP_M, NOMINAL_APPROACH_STEPS, NOMINAL_FOOT_SEPARATION_M,
    approach_target_x, planned_target_batch,
)
from simulation.isaac_lab.g1_stairs.footstep_tread import qualified_tread_mask
from simulation.isaac_lab.g1_stairs.footstep_runtime import FootstepRuntime
from simulation.isaac_lab.g1_stairs.footstep_live_progress import stop_command_after_plan


def course(n=1):
    return dict(first_riser_x=torch.ones(n), tread_depth=torch.full((n,), .25),
                riser_height=torch.full((n,), .2), centre_y=torch.zeros(n),
                foot_separation=torch.full((n,), NOMINAL_FOOT_SEPARATION_M))


def plan(index, approach=1, first_foot=0):
    return planned_target_batch(**course(), step_count=3,
                                first_foot=torch.tensor([first_foot]),
                                target_index=torch.tensor([index]),
                                approach_steps=approach)


class ApproachPlanTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(NOMINAL_APPROACH_STEPS, 1)
        self.assertAlmostEqual(APPROACH_LAST_GAP_M, 0.20)
        self.assertAlmostEqual(NOMINAL_FOOT_SEPARATION_M, 0.235)

    def test_full_sequence_ground_then_treads_then_platform(self):
        xs, zs, feet = [], [], []
        for i in range(6):
            t, f, v = plan(i)
            self.assertTrue(bool(v[0, 0]))
            xs.append(round(t[0, 0, 0].item(), 4))
            zs.append(round(t[0, 0, 2].item(), 4))
            feet.append(int(f[0, 0]))
        # Last ground placement 0.20 m before the riser (toe ~0.10 m clear).
        self.assertEqual(xs, [0.8, 1.125, 1.375, 1.625, 1.875, 2.125])
        self.assertEqual(zs, [0.0, 0.2, 0.4, 0.6, 0.6, 0.6])
        self.assertEqual(feet, [0, 1, 0, 1, 0, 1])           # strict alternation
        _, _, v = plan(6)
        self.assertFalse(bool(v[0, 0]))

    def test_earlier_approach_targets_one_tread_behind(self):
        t, _, _ = plan(0, approach=2)
        self.assertAlmostEqual(t[0, 0, 0].item(), 0.55, places=6)
        self.assertAlmostEqual(t[0, 1, 0].item(), 0.80, places=6)
        self.assertAlmostEqual(
            approach_target_x(torch.tensor(1.0), torch.tensor(.25), -1).item(), 0.8, places=6)

    def test_lateral_targets_use_prior_stance_width(self):
        t, _, _ = plan(0)
        self.assertAlmostEqual(t[0, 0, 1].item(), NOMINAL_FOOT_SEPARATION_M / 2, places=6)
        self.assertAlmostEqual(t[0, 1, 1].item(), -NOMINAL_FOOT_SEPARATION_M / 2, places=6)

    def test_zero_approach_keeps_original_stair_plan(self):
        t, f, _ = plan(0, approach=0)
        self.assertAlmostEqual(t[0, 0, 0].item(), 1.125)
        self.assertAlmostEqual(t[0, 0, 2].item(), 0.2, places=6)

    def test_approach_is_stairs_only(self):
        with self.assertRaises(ValueError):
            planned_target_batch(**course(), step_count=3, first_foot=torch.tensor([0]),
                                 target_index=torch.tensor([0]), flat_ground=True,
                                 approach_steps=2)


class ApproachQualificationTests(unittest.TestCase):
    def test_ground_cell_before_riser_qualifies_assigned_foot(self):
        # Approach target (x=0.8) is the left foot: ground cell [0.675, 0.925].
        targets, feet, _ = plan(0)
        self.assertEqual(int(feet[0, 0]), 0)
        bounds = torch.tensor([[[0.71, 0.90, .06, .17, 0., 0.],
                                [0., .1, -.17, -.06, 0., 0.]]])
        m = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=torch.tensor([[110., 110.]]),
            targets=targets, feet=feet, valid=torch.ones((1, 2), dtype=torch.bool),
            first_riser_x=torch.ones(1), tread_depth=torch.full((1,), .25),
            riser_height=torch.full((1,), .2), step_count=3, half_width=.6,
            approach_steps=1)
        self.assertEqual(m.tolist(), [[True, False]])

    def test_ground_cell_touching_riser_rejected(self):
        targets, feet, _ = plan(0)
        bounds = torch.tensor([[[0.81, 0.999, .06, .17, 0., 0.],
                                [0., .1, -.17, -.06, 0., 0.]]])
        m = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=torch.tensor([[110., 110.]]),
            targets=targets, feet=feet, valid=torch.ones((1, 2), dtype=torch.bool),
            first_riser_x=torch.ones(1), tread_depth=torch.full((1,), .25),
            riser_height=torch.full((1,), .2), step_count=3, half_width=.6,
            approach_steps=1)
        self.assertEqual(m.tolist(), [[False, False]])

    def test_ground_target_rejected_when_approach_disabled(self):
        targets, feet, _ = plan(0)
        bounds = torch.tensor([[[0.71, 0.90, .06, .17, 0., 0.],
                                [0., .1, -.17, -.06, 0., 0.]]])
        m = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=torch.tensor([[110., 110.]]),
            targets=targets, feet=feet, valid=torch.ones((1, 2), dtype=torch.bool),
            first_riser_x=torch.ones(1), tread_depth=torch.full((1,), .25),
            riser_height=torch.full((1,), .2), step_count=3, half_width=.6,
            approach_steps=0)
        self.assertEqual(m.tolist(), [[False, False]])


class PlanLengthAndStopTests(unittest.TestCase):
    def runtime(self):
        return FootstepRuntime(num_envs=3, device='cpu', step_count=3,
                               first_foot=torch.zeros(3, dtype=torch.long),
                               approach_steps=1)

    def test_plan_length_stairs_and_flat(self):
        r = self.runtime()
        self.assertEqual(r.plan_length(False), 6)
        self.assertEqual(r.plan_length(True), 5)

    def test_command_zeroed_only_for_completed_non_resetting_envs(self):
        r = self.runtime()
        r.target_index[:] = torch.tensor([6, 5, 6])
        command = torch.full((3, 3), 0.35)
        env = NS(command_manager=NS(get_command=lambda name: command))
        done = stop_command_after_plan(env, r, False, torch.tensor([False, False, True]))
        self.assertEqual(done.tolist(), [True, False, False])
        self.assertEqual(command[0].tolist(), [0., 0., 0.])
        self.assertAlmostEqual(command[1, 0].item(), 0.35, places=6)
        self.assertAlmostEqual(command[2, 0].item(), 0.35, places=6)


if __name__ == "__main__":
    unittest.main()


class TrainingAcceptanceTests(unittest.TestCase):
    """full_sole=False: heel on the tread, toe below the next riser, spot toe
    clear of the first riser, last platform target open-ended."""

    def mask(self, index, sole_x, z):
        targets, feet, _ = plan(index)
        f = int(feet[0, 0])
        bounds = torch.zeros((1, 2, 6))
        bounds[0, :, 2], bounds[0, :, 3] = -.17, -.06
        bounds[0, f] = torch.tensor([sole_x[0], sole_x[1], .06 if f == 0 else -.17,
                                     .17 if f == 0 else -.06, z, z])
        m = qualified_tread_mask(
            sole_bounds=bounds, vertical_force=torch.tensor([[110., 110.]]),
            targets=targets, feet=feet, valid=torch.ones((1, 2), dtype=torch.bool),
            first_riser_x=torch.ones(1), tread_depth=torch.full((1,), .25),
            riser_height=torch.full((1,), .2), step_count=3, half_width=.6,
            approach_steps=1, full_sole=False)
        return bool(m[0, f])

    def test_tread_heel_overhang_limit(self):
        # tread 1 (index 1): front edge at 1.0, next riser at 1.25, z 0.2
        self.assertTrue(self.mask(1, (0.98, 1.17), .2))     # heel 2 cm over
        self.assertFalse(self.mask(1, (0.94, 1.13), .2))    # heel 6 cm over (centre on the tread)
        self.assertFalse(self.mask(1, (1.08, 1.27), .2))    # toe past the next riser

    def test_top_tread_has_no_toe_limit(self):
        self.assertTrue(self.mask(3, (1.52, 1.78), .6))     # tread 3, platform ahead at the same level

    def test_spot_toe_must_clear_the_riser(self):
        self.assertTrue(self.mask(0, (0.70, 0.88), 0.))
        self.assertFalse(self.mask(0, (0.80, 0.99), 0.))    # toe 1 cm from the riser

    def test_last_platform_target_accepts_overshoot(self):
        self.assertTrue(self.mask(5, (2.18, 2.37), .6))     # centre 2.27, past the cell
        self.assertFalse(self.mask(5, (1.90, 2.09), .6))    # centre 1.995, before the cell
        self.assertFalse(self.mask(4, (2.03, 2.21), .6))    # first platform target keeps its cell
