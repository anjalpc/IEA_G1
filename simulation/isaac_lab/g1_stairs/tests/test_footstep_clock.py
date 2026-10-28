import math
import unittest

import torch

from simulation.isaac_lab.g1_stairs import footstep_clock as fc


class ClockConfigTests(unittest.TestCase):
    def test_disabled_and_valid(self):
        self.assertIsNone(fc.validate_clock_cfg(None))
        self.assertIsNone(fc.validate_clock_cfg({"enabled": False}))
        cfg = fc.validate_clock_cfg(
            {"enabled": True, "period_s": [0.6, 1.2], "max_stride_m": 0.55})
        self.assertEqual(cfg["period_s"], (0.6, 1.2))

    def test_rejects_bad_values(self):
        for bad in ({"enabled": True, "period_s": [1.2, 0.6], "max_stride_m": 0.5},
                    {"enabled": True, "period_s": [0.6, 1.2], "max_stride_m": 2.0},
                    {"enabled": True, "period_s": [0.6, 1.2], "max_stride_m": 0.5,
                     "typo": 1}):
            with self.assertRaises(ValueError):
                fc.validate_clock_cfg(bad)


class ScheduleTests(unittest.TestCase):
    def test_alternating_swings_and_assignment(self):
        phase = torch.tensor([0.05, 0.3, 0.55, 0.8])
        swing, assigned = fc.clock_schedule(phase)
        # double support, left swing, double support, right swing
        self.assertEqual(swing.tolist(), [[False, False], [True, False],
                                          [False, False], [False, True]])
        # assigned = swinging foot, or next one in double support
        self.assertEqual(assigned.tolist(), [0, 0, 1, 1])

    def test_swing_fractions_cover_cycle(self):
        total = 2 * (fc.DOUBLE_SUPPORT_FRACTION + fc.SINGLE_SUPPORT_FRACTION)
        self.assertAlmostEqual(total, 1.0)

    def test_features_zero_when_standing(self):
        phase = torch.tensor([0.25, 0.25])
        out = fc.clock_features(phase, torch.tensor([True, False]))
        self.assertAlmostEqual(out[0, 0].item(), 1.0, places=6)
        self.assertEqual(out[1].tolist(), [0.0, 0.0])

    def test_effective_period_caps_stride(self):
        base = torch.tensor([1.2, 1.2, 0.6])
        speed = torch.tensor([0.3, 0.6, 0.6])
        out = fc.effective_period(base, speed, 0.55)
        self.assertAlmostEqual(out[0].item(), 1.2, places=6)       # 0.36 m stride
        self.assertAlmostEqual(out[1].item(), 0.55 / 0.6, places=6)  # capped
        self.assertAlmostEqual(out[2].item(), 0.6, places=6)

    def test_moving(self):
        cmd = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.2]])
        self.assertEqual(fc.moving(cmd).tolist(), [False, True, True])


class RollingTargetTests(unittest.TestCase):
    def test_straight_ahead(self):
        stance = torch.tensor([[1.0, -0.1175]])          # right foot stance
        targets, feet = fc.rolling_targets(
            stance, torch.tensor([0]), torch.tensor([0.0]),
            torch.tensor([0.2]), torch.tensor([0.0]), 0.235)
        self.assertEqual(feet.tolist(), [[0, 1]])
        self.assertAlmostEqual(targets[0, 0, 0].item(), 1.2, places=6)
        self.assertAlmostEqual(targets[0, 0, 1].item(), 0.1175, places=6)
        self.assertAlmostEqual(targets[0, 1, 0].item(), 1.4, places=6)
        self.assertAlmostEqual(targets[0, 1, 1].item(), -0.1175, places=6)
        self.assertEqual(targets[0, 0, 2].item(), 0.0)

    def test_heading_rotates_steps(self):
        stance = torch.tensor([[0.0, 0.0]])               # left foot stance
        targets, _ = fc.rolling_targets(
            stance, torch.tensor([1]), torch.tensor([math.pi / 2]),
            torch.tensor([0.2]), torch.tensor([0.0]), 0.235)
        # Facing +y: forward is +y, the right foot lies at +x.
        self.assertAlmostEqual(targets[0, 0, 0].item(), 0.235, places=5)
        self.assertAlmostEqual(targets[0, 0, 1].item(), 0.2, places=5)
        self.assertAlmostEqual(targets[0, 0, 3].item(), math.pi / 2, places=6)


class StateTests(unittest.TestCase):
    def test_reset_and_advance(self):
        torch.manual_seed(0)
        state = fc.RollingClockState(64, "cpu")
        state.reset(None, (0.6, 1.2))
        self.assertTrue(bool(((state.base_period_s >= 0.6) & (state.base_period_s <= 1.2)).all()))
        self.assertTrue(set(state.phase.tolist()) <= {0.0, 0.5})
        self.assertEqual(set(state.phase.tolist()), {0.0, 0.5})  # both leads occur
        state.period_s[:] = 1.0
        walking = torch.zeros(64, dtype=torch.bool)
        walking[0] = True
        before = state.phase.clone()
        state.advance(0.25, walking)
        self.assertAlmostEqual(state.phase[0].item(), (before[0].item() + 0.25) % 1.0, places=6)
        self.assertEqual(state.phase[1].item(), before[1].item())   # standing holds
        state.reset(torch.tensor([0]), (0.8, 0.8))
        self.assertAlmostEqual(state.base_period_s[0].item(), 0.8, places=6)
        self.assertEqual(state.assigned[0].item(), -1)


if __name__ == "__main__":
    unittest.main()


class ApplyRollingClockTests(unittest.TestCase):
    """apply_rolling_clock on a stand-in env (Isaac pieces patched out)."""

    def setUp(self):
        from types import SimpleNamespace
        from unittest import mock
        import simulation.isaac_lab.g1_stairs.footstep_live_progress as lp
        self.lp = lp
        self.command = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.env = SimpleNamespace(
            num_envs=3, device="cpu",
            cfg=SimpleNamespace(footstep_clock={"enabled": True, "period_s": (1.0, 1.0),
                                                "max_stride_m": 0.55}),
            command_manager=SimpleNamespace(get_command=lambda name: self.command),
        )
        # rows 0 and 2 flat (clock), row 1 a stair tile
        patches = [
            mock.patch.object(lp, "riser_height_per_env",
                              lambda env: torch.tensor([0.0, 0.04, 0.0])),
            mock.patch.object(lp, "course_root_pose",
                              lambda env: (torch.zeros(3, 3), torch.zeros(3))),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.state = lp.ensure_rolling_clock(self.env)
        self.state.phase[:] = 0.0        # left assigned first
        self.centre = torch.tensor([[[0.0, 0.1175, 0.0], [0.0, -0.1175, 0.0]]] * 3)

    def run_once(self):
        targets = torch.full((3, 2, 4), 9.0)
        feet = torch.zeros((3, 2), dtype=torch.long)
        valid = torch.ones((3, 2), dtype=torch.bool)
        rows, walking, swing = self.lp.apply_rolling_clock(
            self.env, self.state, centre=self.centre, targets=targets,
            feet=feet, valid=valid, resets=torch.zeros(3, dtype=torch.bool))
        return rows, walking, targets, feet, valid

    def test_first_window_targets_swing_foot_from_stance(self):
        rows, walking, targets, feet, valid = self.run_once()
        self.assertEqual(rows.tolist(), [True, False, True])
        self.assertEqual(walking.tolist(), [True, False, False])
        # left foot: from the right stance foot, step = 0.4 * 1.0 / 2
        self.assertAlmostEqual(targets[0, 0, 0].item(), 0.2, places=6)
        self.assertAlmostEqual(targets[0, 0, 1].item(), 0.1175, places=6)
        self.assertEqual(feet[0].tolist(), [0, 1])
        self.assertEqual(valid[0].tolist(), [True, True])
        # stair row untouched; standing flat row has no valid target
        self.assertEqual(targets[1, 0, 0].item(), 9.0)
        self.assertEqual(valid[2].tolist(), [False, False])
        self.assertEqual(self.state.window[0].item(), 1)

    def test_window_holds_then_switches_foot(self):
        self.run_once()
        self.state.acquired[0] = True
        _, _, targets, _, valid = self.run_once()      # same window
        self.assertEqual(self.state.window[0].item(), 1)
        self.assertFalse(valid[0, 0].item())            # one hit per window
        self.state.phase[0] = 0.6                       # right foot's turn
        _, _, targets, feet, valid = self.run_once()
        self.assertEqual(self.state.window[0].item(), 2)
        self.assertEqual(feet[0].tolist(), [1, 0])
        self.assertTrue(valid[0, 0].item())
        self.assertAlmostEqual(targets[0, 0, 1].item(), -0.1175, places=6)

    def test_resume_after_standing_replans(self):
        self.run_once()
        self.command[0, 0] = 0.0
        self.run_once()                                  # standing
        self.assertEqual(self.state.window[0].item(), 1)
        self.command[0, 0] = 0.4
        self.centre[0, 1, 0] = 0.5                       # stance foot moved
        _, _, targets, _, _ = self.run_once()
        self.assertEqual(self.state.window[0].item(), 2)
        self.assertAlmostEqual(targets[0, 0, 0].item(), 0.7, places=6)
