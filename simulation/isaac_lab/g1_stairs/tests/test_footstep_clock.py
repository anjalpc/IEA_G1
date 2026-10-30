import math
import unittest

import torch

from simulation.isaac_lab.g1_stairs import footstep_clock as fc


class ClockConfigTests(unittest.TestCase):
    GOOD = {"enabled": True, "period_slow_s": 1.0, "period_fast_s": 0.7,
            "period_stairs_s": 1.3, "period_jitter": 0.1, "max_stride_m": 0.55}

    def test_disabled_and_valid(self):
        self.assertIsNone(fc.validate_clock_cfg(None))
        self.assertIsNone(fc.validate_clock_cfg({"enabled": False}))
        self.assertEqual(fc.validate_clock_cfg(dict(self.GOOD)), self.GOOD)

    def test_rejects_bad_values(self):
        for change in ({"period_fast_s": 1.2},            # faster than slow
                       {"period_jitter": 0.5},
                       {"period_stairs_s": 3.0},
                       {"max_stride_m": 2.0},
                       {"typo": 1},
                       {"period_s": [0.6, 1.2]}):          # old key
            with self.assertRaises(ValueError):
                fc.validate_clock_cfg({**self.GOOD, **change})
        missing = dict(self.GOOD)
        del missing["period_jitter"]
        with self.assertRaises(ValueError):
            fc.validate_clock_cfg(missing)


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

    def test_speed_period(self):
        cfg = {"enabled": True, "period_slow_s": 1.0, "period_fast_s": 0.7, "period_jitter": 0.1, "max_stride_m": 0.55}
        speed = torch.tensor([0.0, 0.1, 0.35, 0.6, 1.0, 0.35])
        scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.1])
        out = fc.speed_period(speed, scale, cfg)
        self.assertAlmostEqual(out[0].item(), 1.0, places=6)       # slow end
        self.assertAlmostEqual(out[1].item(), 1.0, places=6)
        self.assertAlmostEqual(out[2].item(), 0.85, places=6)      # halfway
        self.assertAlmostEqual(out[3].item(), 0.7, places=6)       # fast end
        self.assertAlmostEqual(out[4].item(), 0.55, places=6)      # stride cap 0.55 m
        self.assertAlmostEqual(out[5].item(), 0.935, places=6)     # jitter scale

    def test_own_window(self):
        phase = torch.tensor([0.10, 0.20, 0.60, 0.70, 0.10, 0.60])
        foot = torch.tensor([0, 0, 0, 0, 1, 1])
        self.assertEqual(fc.own_window(phase, foot).tolist(),
                         [False, True, True, False, True, False])

    def test_stair_handover(self):
        out = fc.stair_handover(torch.tensor([1.0, 1.0, 1.0, 1.0]),
                                torch.tensor([0.75, 0.60, 0.75, 1.2]),
                                torch.tensor([True, True, False, True]),
                                torch.tensor([True, True, True, True]))
        self.assertEqual(out.tolist(), [True, False, False, True])

    def test_follow_plan_foot(self):
        # rows: right swing opens but the plan waits for left (shift); right
        # swing opens for the planned right foot (keep); inside the right
        # swing with the plan already on left (keep); double support (keep,
        # a late touchdown can still count); not a stair row (keep).
        phase = torch.tensor([0.66, 0.66, 0.80, 0.55, 0.66])
        was_swinging = torch.tensor([False, False, True, False, False])
        plan_foot = torch.tensor([0, 1, 0, 0, 0])
        rows = torch.tensor([True, True, True, True, False])
        out = fc.follow_plan_foot(phase, was_swinging, plan_foot, rows)
        self.assertAlmostEqual(out[0].item(), 0.16, places=5)   # left swings again
        self.assertEqual(out[1:].tolist(), phase[1:].tolist())
        swing, _ = fc.clock_schedule(out)
        self.assertEqual(swing[0].tolist(), [True, False])

    def test_moving(self):
        cmd = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.2]])
        self.assertEqual(fc.moving(cmd).tolist(), [False, True, True])


class HeadingTargetTests(unittest.TestCase):
    def test_straight_ahead_one_step_from_stance(self):
        cmd = torch.tensor([[0.4, 0.0, 0.0]])
        targets, feet = fc.heading_targets(
            torch.tensor([[1.0, -0.1175]]), torch.tensor([0]), torch.tensor([0.0]),
            cmd, torch.tensor([1.0]), 0.235)
        self.assertEqual(feet.tolist(), [[0, 1]])
        self.assertAlmostEqual(targets[0, 0, 0].item(), 1.2, places=6)
        self.assertAlmostEqual(targets[0, 0, 1].item(), 0.1175, places=6)
        self.assertAlmostEqual(targets[0, 1, 0].item(), 1.4, places=6)
        self.assertAlmostEqual(targets[0, 1, 1].item(), -0.1175, places=6)
        self.assertEqual(targets[0, 0, 2].item(), 0.0)

    def test_robot_heading_rotates_steps(self):
        # Robot heading 90 deg: forward is +y, the right foot lies at +x.
        targets, _ = fc.heading_targets(
            torch.tensor([[0.0, 0.0]]), torch.tensor([1]), torch.tensor([math.pi / 2]),
            torch.tensor([[0.2, 0.0, 0.0]]), torch.tensor([2.0]), 0.235)
        self.assertAlmostEqual(targets[0, 0, 0].item(), 0.235, places=5)
        self.assertAlmostEqual(targets[0, 0, 1].item(), 0.2, places=5)
        self.assertAlmostEqual(targets[0, 0, 3].item(), math.pi / 2, places=6)

    def test_turn_command_turns_heading_at_touchdown(self):
        targets, _ = fc.heading_targets(
            torch.tensor([[0.0, 0.0]]), torch.tensor([0]), torch.tensor([0.0]),
            torch.tensor([[0.0, 0.0, 0.5]]), torch.tensor([1.0]), 0.235)
        self.assertAlmostEqual(targets[0, 0, 3].item(), 0.25, places=6)
        self.assertAlmostEqual(targets[0, 1, 3].item(), 0.5, places=6)


class StateTests(unittest.TestCase):
    def test_reset_and_advance(self):
        torch.manual_seed(0)
        state = fc.RollingClockState(64, "cpu")
        state.reset(None, 0.1)
        self.assertTrue(bool(((state.period_scale >= 0.9) & (state.period_scale <= 1.1)).all()))
        self.assertTrue(set(state.phase.tolist()) <= {0.0, 0.5})
        self.assertEqual(set(state.phase.tolist()), {0.0, 0.5})  # both leads occur
        state.period_s[:] = 1.0
        walking = torch.zeros(64, dtype=torch.bool)
        walking[0] = True
        before = state.phase.clone()
        state.advance(0.25, walking)
        self.assertAlmostEqual(state.phase[0].item(), (before[0].item() + 0.25) % 1.0, places=6)
        self.assertEqual(state.phase[1].item(), before[1].item())   # standing holds
        state.reset(torch.tensor([0]), 0.0)
        self.assertEqual(state.period_scale[0].item(), 1.0)
        self.assertEqual(state.assigned[0].item(), -1)

    def test_advance_moves_phase_of_walking_rows(self):
        state = fc.RollingClockState(2, "cpu")
        state.period_s[:] = 1.0
        state.advance(0.25, torch.tensor([True, False]))
        self.assertAlmostEqual(state.phase[0].item(), 0.25, places=6)
        self.assertEqual(state.phase[1].item(), 0.0)


class YawDriftTests(unittest.TestCase):
    def run_cycle(self, errors, walking=(True, True)):
        state = fc.RollingClockState(len(walking), "cpu")
        state.period_s[:] = 1.0
        rows = torch.tensor(walking)
        for err in errors:   # 4 steps of 0.25 s = one full cycle (exact in float)
            state.advance(0.25, rows, yaw_rate_error=torch.tensor(err))
        return state

    def test_steady_drift_is_held_after_the_cycle(self):
        state = self.run_cycle([[0.02, 0.0]] * 4)
        self.assertAlmostEqual(state.cycle_yaw_error[0].item(), 0.02, places=5)
        self.assertEqual(state.cycle_yaw_error[1].item(), 0.0)
        self.assertEqual(state.yaw_error_time[0].item(), 0.0)    # new cycle started

    def test_in_step_twist_cancels(self):
        twist = [[0.3, 0.0], [-0.3, 0.0], [0.3, 0.0], [-0.3, 0.0]]
        state = self.run_cycle(twist)
        self.assertLess(abs(state.cycle_yaw_error[0].item()), 1e-3)

    def test_no_value_before_a_full_cycle_and_standing_clears(self):
        state = self.run_cycle([[0.05, 0.05]] * 3)
        self.assertEqual(state.cycle_yaw_error.tolist(), [0.0, 0.0])
        state.advance(0.25, torch.tensor([True, True]),
                      yaw_rate_error=torch.tensor([0.05, 0.05]))
        self.assertAlmostEqual(state.cycle_yaw_error[0].item(), 0.05, places=5)
        state.advance(0.25, torch.tensor([True, False]),
                      yaw_rate_error=torch.tensor([0.05, 0.05]))
        self.assertAlmostEqual(state.cycle_yaw_error[0].item(), 0.05, places=5)
        self.assertEqual(state.cycle_yaw_error[1].item(), 0.0)   # standing row cleared
        state.reset(torch.tensor([0]), 0.0)
        self.assertEqual(state.cycle_yaw_error[0].item(), 0.0)

    def test_cost(self):
        cost = fc.yaw_drift_cost(torch.tensor([0.0, fc.YAW_DRIFT_STD_RADPS, -1.0]))
        self.assertEqual(cost[0].item(), 0.0)
        self.assertAlmostEqual(cost[1].item(), 1 - math.exp(-1), places=6)
        self.assertAlmostEqual(cost[2].item(), 1.0, places=6)


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
            # slow == fast and no jitter: a fixed 1.0 s cycle at any speed
            cfg=SimpleNamespace(footstep_clock={"enabled": True, "period_slow_s": 1.0,
                                                "period_fast_s": 1.0, "period_stairs_s": 1.3,
                                                "period_jitter": 0.0, "max_stride_m": 0.55}),
            command_manager=SimpleNamespace(get_command=lambda name: self.command),
        )
        # rows 0 and 2 flat (clock), row 1 a stair tile
        patches = [
            mock.patch.object(lp, "riser_height_per_env",
                              lambda env: torch.tensor([0.0, 0.04, 0.0])),
            mock.patch.object(lp, "course_root_pose",
                              lambda env: (torch.zeros(3, 3), self.yaw)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.state = lp.ensure_rolling_clock(self.env)
        self.state.phase[:] = 0.0        # left assigned first
        self.yaw = torch.zeros(3)
        self.centre = torch.tensor([[[0.0, 0.1175, 0.0], [0.0, -0.1175, 0.0]]] * 3)

    def run_once(self, plan_foot=0, on_plan=None, first_plan_x=5.0):
        if on_plan is not None:
            self.state.on_plan[1] = on_plan
        targets = torch.full((3, 2, 4), 9.0)
        feet = torch.zeros((3, 2), dtype=torch.long)
        feet[:, 0] = plan_foot
        valid = torch.ones((3, 2), dtype=torch.bool)
        rows, walking, swing, self.handover = self.lp.apply_rolling_clock(
            self.env, self.state, centre=self.centre, targets=targets,
            feet=feet, valid=valid, resets=torch.zeros(3, dtype=torch.bool),
            first_plan_x=torch.full((3,), first_plan_x))
        return rows, walking, targets, feet, valid

    def test_first_window_targets_swing_foot_from_stance(self):
        rows, walking, targets, feet, valid = self.run_once(on_plan=True)
        self.assertEqual(rows.tolist(), [True, False, True])
        # stair row on its plan walks while the plan has a target; standing flat row does not
        self.assertEqual(walking.tolist(), [True, True, False])
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
        self.yaw[0] = 0.3                                # turned while standing
        self.run_once()                                  # standing
        self.assertEqual(self.state.window[0].item(), 1)
        self.command[0, 0] = 0.4
        self.centre[0, 1, 0] = 0.5                       # stance foot moved
        _, _, targets, _, _ = self.run_once()
        self.assertEqual(self.state.window[0].item(), 2)
        self.assertAlmostEqual(targets[0, 0, 3].item(), 0.3, places=6)

    def test_targets_follow_the_robot_heading(self):
        self.run_once()
        self.yaw[0] = 0.2                                # robot now faces 0.2 rad
        self.state.phase[0] = 0.6                        # next window
        _, _, targets, _, _ = self.run_once()
        # body-relative: no stored heading pulls the step back
        self.assertAlmostEqual(targets[0, 0, 3].item(), 0.2, places=6)

    def test_period_follows_the_command(self):
        self.env.cfg.footstep_clock.update(period_slow_s=1.0, period_fast_s=0.7)
        self.command[0, 0] = 0.6
        self.run_once()
        self.assertAlmostEqual(self.state.period_s[0].item(), 0.7, places=6)

    def test_stair_approach_walks_like_a_flat_tile(self):
        rows, _, targets, _, _ = self.run_once()                  # far from the stairs
        self.assertEqual(rows.tolist(), [True, True, True])
        self.assertAlmostEqual(targets[1, 0, 0].item(), 0.2, places=6)   # rolling target
        self.assertAlmostEqual(self.state.period_s[1].item(), 1.0, places=6)
        self.assertFalse(bool(self.handover[1]))

    def test_handover_near_the_stairs_starts_the_plan_with_the_next_swing_foot(self):
        _, _, _, _, valid = self.run_once(first_plan_x=0.25)      # stance at 0.0
        self.assertTrue(bool(self.handover[1]))
        self.assertFalse(bool(self.handover[0]))                  # flat tiles never hand over
        self.assertTrue(bool(self.state.on_plan[1]))
        self.assertEqual(self.state.assigned[1].item(), 0)        # left swings next
        self.assertEqual(valid[1].tolist(), [False, False])       # plan targets next step
        rows, _, targets, _, _ = self.run_once()
        self.assertFalse(bool(rows[1]))
        self.assertEqual(targets[1, 0, 0].item(), 9.0)            # stair plan kept
        self.assertAlmostEqual(self.state.period_s[1].item(), 1.3, places=6)

    def test_stair_target_only_valid_in_its_own_window(self):
        self.state.phase[1] = 0.05                                # double support before left
        _, _, _, _, valid = self.run_once(plan_foot=0, on_plan=True)
        self.assertFalse(bool(valid[1, 0]))
        self.state.phase[1] = 0.30                                # left swinging
        _, _, _, _, valid = self.run_once(plan_foot=0, on_plan=True)
        self.assertTrue(bool(valid[1, 0]))
        self.state.phase[1] = 0.55                                # double support after left
        _, _, _, _, valid = self.run_once(plan_foot=0, on_plan=True)
        self.assertTrue(bool(valid[1, 0]))

    def test_miss_on_stairs_gives_the_same_foot_the_next_window(self):
        self.state.phase[1] = 0.45                                # left swinging
        self.run_once(plan_foot=0, on_plan=True)
        self.state.phase[1] = 0.55                                # double support:
        self.run_once(plan_foot=0, on_plan=True)                  # no shift yet
        self.assertAlmostEqual(self.state.phase[1].item(), 0.55, places=6)
        self.state.phase[1] = 0.66                                # right swing opens,
        self.run_once(plan_foot=0, on_plan=True)                  # plan waits for left
        swing, _ = self.lp.clock_schedule(self.state.phase)
        self.assertEqual(swing[1].tolist(), [True, False])

    def test_late_touchdown_in_double_support_needs_no_shift(self):
        self.state.phase[1] = 0.45                                # left swinging
        self.run_once(plan_foot=0, on_plan=True)
        self.state.phase[1] = 0.55                                # left lands late: hit,
        self.run_once(plan_foot=1, on_plan=True)                  # plan now on right
        self.state.phase[1] = 0.66                                # right swing opens
        self.run_once(plan_foot=1, on_plan=True)
        self.assertAlmostEqual(self.state.phase[1].item(), 0.66, places=6)

if __name__ == "__main__":
    unittest.main()
