"""Patch 57: clean-climb events, the clean-climb curriculum and the stair cycle."""
import types
import unittest

import torch

from simulation.isaac_lab.g1_stairs import clean_climb as cc
from simulation.isaac_lab.g1_stairs import footstep_clock as fc
from simulation.isaac_lab.g1_stairs import terrain_curriculum as tc

FIRST, TREAD = torch.tensor([1.0]), torch.tensor([0.25])
ROW = torch.tensor([True])


def bounds(heel_l, toe_l, heel_r=0.0, toe_r=0.2):
    """[1,2,6] sole bounds with only heel/toe x set."""
    out = torch.zeros(1, 2, 6)
    out[0, 0, :2] = torch.tensor([heel_l, toe_l])
    out[0, 1, :2] = torch.tensor([heel_r, toe_r])
    return out


class ToeStrikeTests(unittest.TestCase):
    def test_push_near_a_riser_line_is_a_strike(self):
        push = torch.tensor([[60.0, 0.0]])
        # toe 4 cm before riser 2 (1.25 m): strike; 7 cm before: none
        self.assertTrue(cc.toe_strikes(bounds(1.0, 1.21), push, FIRST, TREAD, 3, ROW)[0])
        self.assertFalse(cc.toe_strikes(bounds(1.0, 1.18), push, FIRST, TREAD, 3, ROW)[0])
        # under the force threshold, or not a stair row: none
        self.assertFalse(cc.toe_strikes(bounds(1.0, 1.21), push * 0.5, FIRST, TREAD, 3, ROW)[0])
        self.assertFalse(cc.toe_strikes(bounds(1.0, 1.21), push, FIRST, TREAD, 3, ~ROW)[0])
        # the right foot counts as well
        self.assertTrue(cc.toe_strikes(bounds(0.0, 0.2, 1.3, 1.52), torch.tensor([[0.0, 60.0]]),
                                       FIRST, TREAD, 3, ROW)[0])


class NoseLandingTests(unittest.TestCase):
    def run_frames(self, sole, pushes, touchdown_frame=0):
        window = torch.zeros(1, 2, dtype=torch.long)
        hits = []
        for k, push in enumerate(pushes):
            td = torch.tensor([[k == touchdown_frame, False]])
            hit, window = cc.nose_landings(td, sole, torch.tensor([[push, 0.0]]), window,
                                           FIRST, TREAD, 3, ROW)
            hits.append(bool(hit[0]))
        return hits

    def test_heel_over_the_edge_and_a_push_within_four_frames(self):
        # centre on tread 2 (1.25-1.50), heel 1 cm behind its edge
        sole = bounds(1.24, 1.44)
        self.assertEqual(self.run_frames(sole, [0, 0, 0, 60, 0]), [False, False, False, True, False])
        # the fifth frame is outside the window
        self.assertEqual(self.run_frames(sole, [0, 0, 0, 0, 60]), [False] * 5)
        # touchdown frame included
        self.assertEqual(self.run_frames(sole, [60, 60]), [True, False])

    def test_full_sole_on_the_tread_or_off_the_stairs_is_not_a_nose_landing(self):
        self.assertEqual(self.run_frames(bounds(1.26, 1.46), [60] * 4), [False] * 4)
        self.assertEqual(self.run_frames(bounds(0.5, 0.7), [60] * 4), [False] * 4)
        self.assertEqual(self.run_frames(bounds(1.80, 2.00), [60] * 4), [False] * 4)

    def test_rows_off_clear_the_window(self):
        window = torch.tensor([[3, 2]])
        _, window = cc.nose_landings(torch.zeros(1, 2, dtype=torch.bool), bounds(1.24, 1.44),
                                     torch.zeros(1, 2), window, FIRST, TREAD, 3, ~ROW)
        self.assertEqual(window.tolist(), [[0, 0]])


class StairPeriodTests(unittest.TestCase):
    CFG = {"enabled": True, "period_slow_s": 1.0, "period_fast_s": 0.7,
           "period_stairs_s": 1.25, "period_stairs_per_m": 2.15,
           "period_stairs_rise_from_m": 0.06, "period_jitter": 0.1, "max_stride_m": 0.55}

    def test_verified_heights_keep_their_cycle_higher_risers_get_longer(self):
        cfg = fc.validate_clock_cfg(dict(self.CFG))
        riser = torch.tensor([0.0, 0.02, 0.06, 0.08, 0.20])
        period = fc.stair_period(riser, torch.ones(5), cfg)
        expected = [1.25, 1.25, 1.25, 1.25 + 0.043, 1.25 + 0.301]
        for got, want in zip(period.tolist(), expected):
            self.assertAlmostEqual(got, want, places=5)
        self.assertAlmostEqual(fc.stair_period(riser[-1:], torch.tensor([0.9]), cfg).item(),
                               0.9 * 1.551, places=5)

    def test_optional_keys_default_to_one_cycle(self):
        cfg = dict(self.CFG)
        del cfg["period_stairs_per_m"], cfg["period_stairs_rise_from_m"]
        cfg = fc.validate_clock_cfg(cfg)
        self.assertEqual((cfg["period_stairs_per_m"], cfg["period_stairs_rise_from_m"]), (0.0, 0.0))
        self.assertTrue(torch.allclose(fc.stair_period(torch.tensor([0.0, 0.2]), torch.ones(2), cfg),
                                       torch.tensor([1.25, 1.25])))

    def test_rejects_a_cycle_over_two_seconds_or_bad_values(self):
        fc.validate_clock_cfg({**self.CFG, "period_stairs_per_m": 5.0})    # 1.95 s at 20 cm
        for change in ({"period_stairs_per_m": 5.0, "period_stairs_rise_from_m": 0.0},  # 2.25 s
                       {"period_stairs_per_m": -1.0},
                       {"period_stairs_rise_from_m": 0.3}):
            with self.assertRaises(ValueError):
                fc.validate_clock_cfg({**self.CFG, **change})


HEIGHTS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)


def make_env(num_envs=8, levels=None):
    stairs = types.SimpleNamespace(riser_heights=HEIGHTS)
    terrain = types.SimpleNamespace(
        terrain_levels=torch.zeros(num_envs, dtype=torch.long) if levels is None
        else torch.tensor(levels),
        max_terrain_level=len(HEIGHTS),
        terrain_types=torch.zeros(num_envs, dtype=torch.long),
        # origin x = 10 * level, y = column, so the tests can read the level back
        terrain_origins=torch.stack([torch.stack([torch.tensor([10.0 * r, 0.0, 0.0])])
                                     for r in range(len(HEIGHTS))]),
        env_origins=torch.zeros(num_envs, 3),
        cfg=types.SimpleNamespace(terrain_generator=types.SimpleNamespace(
            sub_terrains={"three_step_ascent": stairs})),
    )
    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]] * num_envs)))

    class Scene(types.SimpleNamespace):
        def __getitem__(self, name):
            return robot

    clock = fc.RollingClockState(num_envs, "cpu")
    return types.SimpleNamespace(
        num_envs=num_envs,
        scene=Scene(terrain=terrain),
        episode_length_buf=torch.full((num_envs,), 100, dtype=torch.long),
        termination_manager=types.SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
        _g1_rolling_clock=clock,
    ), terrain, robot, clock


PARAMS = dict(initial_level=3, mastered_level=3, mastered_clean_rate=0.5, max_levels_ahead=2)


class CleanClimbCurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def start(self, num_envs=8):
        env, terrain, robot, clock = make_env(num_envs)
        log = tc.clean_climb_levels(env, torch.arange(num_envs), **PARAMS)
        return env, terrain, robot, clock, log

    def test_first_reset_assigns_roles_without_judging(self):
        env, terrain, _, _, log = self.start()
        levels = terrain.terrain_levels.tolist()
        # roles by env id: flat, review, frontier, frontier
        self.assertEqual(levels[0::4], [0, 0])
        self.assertTrue(all(1 <= v <= 3 for v in levels[1::4]))
        self.assertEqual(levels[2::4] + levels[3::4], [3, 3, 3, 3])
        self.assertEqual(terrain.env_origins[:, 0].tolist(), [10.0 * v for v in levels])
        self.assertEqual(env._g1_stair_curriculum_attempts.sum().item(), 0)
        self.assertAlmostEqual(log["mastered_cm"], 6.0, places=4)
        self.assertAlmostEqual(log["frontier_cm_mean"], 6.0, places=4)
        self.assertNotIn("share_clean", log)

    def test_clean_climb_up_fault_stays_no_climb_down(self):
        env, terrain, robot, clock, _ = self.start()
        frontier = torch.tensor([2, 3, 6, 7])
        clock.climbed[:] = True
        clock.violations[3, cc.TOE_STRIKE] = True          # env 3: climbed with a toe strike
        clock.climbed[6] = False                            # env 6: no climb
        env.termination_manager.terminated[7] = True        # env 7: climbed, then fell
        tc.clean_climb_levels(env, frontier, **PARAMS)
        self.assertEqual(terrain.terrain_levels[frontier].tolist(), [4, 3, 2, 3])
        self.assertEqual(env._g1_stair_curriculum_attempts[3].item(), 4)
        self.assertEqual(env._g1_stair_curriculum_successes[3].item(), 1)

    def test_not_upright_is_not_clean(self):
        env, terrain, robot, clock, _ = self.start()
        clock.climbed[:] = True
        robot.data.projected_gravity_b[2] = torch.tensor([0.0, 0.8, -0.6])
        tc.clean_climb_levels(env, torch.tensor([2, 3]), **PARAMS)
        self.assertEqual(terrain.terrain_levels[[2, 3]].tolist(), [3, 4])

    def test_frontier_capped_ahead_of_mastered_and_floor_one(self):
        env, terrain, _, clock, _ = self.start()
        clock.climbed[:] = True
        for _ in range(4):                                  # clean every time
            tc.clean_climb_levels(env, torch.tensor([2]), **PARAMS)
        self.assertEqual(terrain.terrain_levels[2].item(), 5)   # mastered 3 + 2
        clock.climbed[:] = False
        for _ in range(8):
            tc.clean_climb_levels(env, torch.tensor([2]), **PARAMS)
        self.assertEqual(terrain.terrain_levels[2].item(), 1)

    def test_flat_never_moves_and_review_stays_within_mastered(self):
        env, terrain, _, clock, _ = self.start()
        clock.climbed[:] = True
        for _ in range(20):
            tc.clean_climb_levels(env, torch.tensor([0, 1, 4, 5]), **PARAMS)
            self.assertEqual(terrain.terrain_levels[[0, 4]].tolist(), [0, 0])
            self.assertTrue(all(1 <= v <= 3 for v in terrain.terrain_levels[[1, 5]].tolist()))

    def test_mastered_rises_with_clean_rate_and_never_falls(self):
        attempts = torch.tensor([0.0, 0, 0, 0, 150, 150, 50, 0, 0, 0, 0])
        clean = torch.tensor([0.0, 0, 0, 0, 90, 60, 50, 0, 0, 0, 0])
        current = torch.tensor(3)
        # 8 cm 0.6 >= 0.5 mastered, 10 cm 0.4 not, 12 cm too few episodes
        self.assertEqual(tc.compute_mastered(attempts, clean, current, 3, 0.5).item(), 4)
        # a known low rate at a trusted height does not lower it
        attempts[2], clean[2] = 200, 20
        self.assertEqual(tc.compute_mastered(attempts, clean, torch.tensor(4), 3, 0.5).item(), 4)
        # top reached when everything is mastered
        full = torch.full((11,), 200.0)
        self.assertEqual(tc.compute_mastered(full, full, current, 3, 0.5).item(), 10)

    def test_no_wrap_around_at_the_top(self):
        env, terrain, _, clock = make_env(4)
        params = dict(PARAMS, initial_level=10, mastered_level=10)
        tc.clean_climb_levels(env, torch.arange(4), **params)
        clock.climbed[:] = True
        for _ in range(3):
            tc.clean_climb_levels(env, torch.tensor([2, 3]), **params)
        self.assertEqual(terrain.terrain_levels[[2, 3]].tolist(), [10, 10])

    def test_window_keeps_the_rate(self):
        env, terrain, _, clock, _ = self.start(num_envs=400)
        frontier = torch.arange(400)[torch.arange(400) % 4 >= 2]
        clock.climbed[:] = True
        clock.violations[frontier[::2], cc.EXTRA_STEP] = True   # half clean
        for _ in range(6):
            terrain.terrain_levels[frontier] = 3
            log = tc.clean_climb_levels(env, frontier, **PARAMS)
        state = env._g1_clean_climb
        self.assertLessEqual(state.attempts[3].item(), tc.CLEAN_CLIMB_WINDOW + 1e-3)
        self.assertAlmostEqual(log["clean_h06cm"], 0.5, places=4)
        self.assertAlmostEqual(log["share_extra_step"], 0.5, places=4)

    def test_needs_the_clock_once_episodes_are_judged(self):
        env, terrain, _, clock, _ = self.start()
        env._g1_rolling_clock = None
        with self.assertRaises(RuntimeError):
            tc.clean_climb_levels(env, torch.tensor([2]), **PARAMS)

    def test_config_validation(self):
        good = {"enabled": True, "type": "clean_climb", **PARAMS}
        self.assertEqual(tc.validate_clean_climb_cfg(good, HEIGHTS), PARAMS)
        for change in ({"initial_level": 0}, {"initial_level": 6}, {"mastered_level": 11},
                       {"mastered_clean_rate": 0.0}, {"max_levels_ahead": 0}, {"typo": 1}):
            with self.assertRaises(ValueError):
                tc.validate_clean_climb_cfg({**good, **change}, HEIGHTS)
        with self.assertRaises(ValueError):
            tc.validate_clean_climb_cfg(good, HEIGHTS[1:])          # no flat level
        missing = dict(good)
        del missing["max_levels_ahead"]
        with self.assertRaises(ValueError):
            tc.validate_clean_climb_cfg(missing, HEIGHTS)


if __name__ == "__main__":
    unittest.main()
