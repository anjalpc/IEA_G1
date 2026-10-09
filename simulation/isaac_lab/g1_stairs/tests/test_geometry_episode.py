import types
import unittest
from unittest.mock import patch

import torch

from simulation.isaac_lab.g1_stairs.geometry_episode import EpisodeContract, EpisodeTracker, Result
from simulation.isaac_lab.g1_stairs.geometry_episode_runtime import (
    GeometryEpisodeRuntime, finish_geometry_episodes, apply_geometry_episode_contract,
)


def tracker(n=1, **kwargs):
    return EpisodeTracker(n, "cpu", EpisodeContract(**kwargs), first_riser=1., tread=.25,
                          step_count=3, course_start=-1., course_end=4.75, half_width=.6, dt=.02)


def sample(n=1, x=3.9, height=.04):
    return dict(
        position=torch.tensor([[x, 0., .8]] * n),
        roll_pitch=torch.zeros(n, 2), riser_height=torch.full((n,), height),
        foot_position=torch.tensor([[[x, -.1, 3*height], [x-.1, .1, 3*height]]] * n),
        foot_force_z=torch.full((n, 2), 100.),
        forbidden_contact=torch.zeros(n, dtype=torch.bool),
        invalid_state=torch.zeros(n, dtype=torch.bool),
        joint_limit=torch.zeros(n, dtype=torch.bool),
        velocity_limit=torch.zeros(n, dtype=torch.bool),
        torque_limit=torch.zeros(n, dtype=torch.bool),
        timed_out=torch.zeros(n, dtype=torch.bool),
    )


class GeometryEpisodeTests(unittest.TestCase):
    def test_supported_completion_requires_dwell(self):
        t = tracker()
        s = sample()
        for _ in range(49):
            self.assertEqual(t.update(**s).item(), Result.RUNNING)
        self.assertEqual(t.update(**s).item(), Result.SUCCESS)

    def test_airborne_and_riser_side_contact_do_not_count(self):
        for case in ("air", "wrong_height", "side_force"):
            t = tracker()
            s = sample()
            if case == "air":
                s["foot_force_z"].zero_()
            elif case == "wrong_height":
                s["foot_position"][:, :, 2] = 0.
            else:
                s["foot_force_z"][:] = -100.
            for _ in range(50):
                t.update(**s)
            self.assertFalse(t.platform_seen.item(), case)
            self.assertNotEqual(t.reason.item(), Result.SUCCESS, case)

    def test_stable_finish_keeps_upper_platform_contact_separate_from_settling(self):
        t = tracker(stable_finish=True)
        s = sample(x=2.0)
        s.update(
            sole_bounds=torch.tensor([[[1.90, 2.10, -0.15, -0.05, .12, .12],
                                       [1.80, 2.00,  0.05,  0.15, .12, .12]]]),
            body_linear_velocity=torch.tensor([[0.40, 0.0, 0.0]]),
            body_angular_velocity=torch.zeros(1, 3),
            foot_linear_velocity=torch.zeros(1, 2, 3),
        )
        self.assertEqual(t.update(**s).item(), Result.RUNNING)
        self.assertTrue(t.platform_contact_seen.item())
        self.assertFalse(t.platform_seen.item())
        self.assertEqual(t.phase.item(), 2)

    def test_interrupted_support_restarts_dwell(self):
        t, s = tracker(), sample()
        for _ in range(30):
            t.update(**s)
        s["foot_force_z"].zero_()
        t.update(**s)
        self.assertEqual(t.dwell.item(), 0.)
        s["foot_force_z"][:] = 100.
        for _ in range(30):
            t.update(**s)
        self.assertEqual(t.reason.item(), Result.RUNNING)

    def test_finish_dwell_peak_progress_is_not_recycled(self):
        t, s = tracker(), sample()
        credited = 0.0

        for _ in range(30):
            t.update(**s)
            credited += float(t.dwell_peak_increment.item())

        self.assertAlmostEqual(credited, 0.60, places=5)
        self.assertAlmostEqual(t.best_dwell.item(), 0.60, places=5)

        s["foot_force_z"].zero_()
        t.update(**s)
        self.assertEqual(t.dwell.item(), 0.0)
        self.assertEqual(t.dwell_peak_increment.item(), 0.0)

        s["foot_force_z"][:] = 100.0
        recycled = 0.0
        for _ in range(30):
            t.update(**s)
            recycled += float(t.dwell_peak_increment.item())

        self.assertAlmostEqual(recycled, 0.0, places=5)
        self.assertAlmostEqual(t.best_dwell.item(), 0.60, places=5)

        for _ in range(10):
            t.update(**s)
            recycled += float(t.dwell_peak_increment.item())

        self.assertAlmostEqual(recycled, 0.20, places=5)
        self.assertAlmostEqual(t.best_dwell.item(), 0.80, places=5)

    def test_failure_overrides_success_on_same_step(self):
        t, s = tracker(supported_dwell_s=.02), sample()
        s["forbidden_contact"][:] = True
        self.assertEqual(t.update(**s).item(), Result.FORBIDDEN_CONTACT)

    def test_success_is_latched_and_reset_clears_it(self):
        t, s = tracker(supported_dwell_s=.02), sample()
        t.update(**s)
        s["position"][:, 2] = 0.
        self.assertEqual(t.update(**s).item(), Result.SUCCESS)
        t.reset(torch.tensor([0]))
        self.assertEqual(t.update(**s).item(), Result.FALL)

    def test_both_course_boundaries(self):
        for axis, value in ((0, 4.7), (0, -.9), (1, .5)):
            t, s = tracker(), sample()
            s["position"][:, axis] = value
            self.assertEqual(t.update(**s).item(), Result.COURSE_BOUNDARY)

    def test_height_relative_fall(self):
        t, s = tracker(), sample(height=.16)
        self.assertEqual(t.update(**s).item(), Result.FALL)

    def test_invalid_state_and_limits_are_terminal(self):
        for key, expected in (("invalid_state", Result.INVALID_STATE),
                              ("joint_limit", Result.JOINT_LIMIT),
                              ("velocity_limit", Result.VELOCITY_LIMIT),
                              ("torque_limit", Result.TORQUE_LIMIT)):
            t, s = tracker(), sample()
            s[key][:] = True
            self.assertEqual(t.update(**s).item(), expected)

    def test_timeout_and_success_priority(self):
        t, s = tracker(), sample(x=.2)
        s["timed_out"][:] = True
        self.assertEqual(t.update(**s).item(), Result.TIMEOUT)
        t, s = tracker(supported_dwell_s=.02), sample()
        s["timed_out"][:] = True
        self.assertEqual(t.update(**s).item(), Result.SUCCESS)

    def test_stall_and_spawn_relative_wrong_direction(self):
        t, s = tracker(), sample(x=.7)
        for _ in range(102):
            t.update(**s)
        self.assertEqual(t.reason.item(), Result.STALL)
        t, s = tracker(), sample(x=.7)
        t.update(**s)
        s["position"][:, 0] = .4
        self.assertEqual(t.update(**s).item(), Result.WRONG_DIRECTION)

    def test_asynchronous_reset_does_not_clear_other_environments(self):
        t, s = tracker(2, supported_dwell_s=.02), sample(2)
        s["position"][1, 0] = .2
        t.update(**s)
        t.reset(torch.tensor([0]))
        self.assertTrue(t.started[1].item())
        self.assertFalse(t.started[0].item())

    def test_bad_contract_is_rejected(self):
        for kwargs in ({"supported_dwell_s": 0.}, {"support_force_n": float("nan")},
                       {"maximum_roll_pitch_rad": 2.}):
            with self.assertRaises(ValueError):
                EpisodeContract(**kwargs)
        with self.assertRaises(ValueError):
            tracker(platform_continuation_m=4.)

    def test_old_experiments_do_not_opt_in(self):
        env_cfg = object()
        self.assertFalse(apply_geometry_episode_contract(env_cfg, {}))


class EpisodeAccountingTests(unittest.TestCase):
    def fake_env(self, reason, levels):
        n = len(levels)
        t = tracker(n)
        t.reason[:] = torch.tensor(reason)
        t.platform_contact_seen[:] = torch.tensor([r == Result.SUCCESS for r in reason])
        t.platform_seen[:] = torch.tensor([r == Result.SUCCESS for r in reason])
        terrain = types.SimpleNamespace(terrain_levels=torch.tensor(levels), calls=0)
        def move(ids, move_up, move_down):
            terrain.calls += 1
            terrain.terrain_levels[ids] += move_up.long() - move_down.long()
        terrain.update_env_origins = move
        runtime = object.__new__(GeometryEpisodeRuntime)
        runtime.tracker = t
        runtime.heights = (0., .04, .06)
        runtime.counts = torch.zeros(3, len(Result), dtype=torch.long)
        runtime.phase_failures = torch.zeros(3, 3, dtype=torch.long)
        runtime.platform_contact_counts = torch.zeros(3, dtype=torch.long)
        runtime.platform_counts = torch.zeros(3, dtype=torch.long)
        runtime.last_completed = None
        env = types.SimpleNamespace(
            _g1_geometry_episode=runtime, device="cpu", common_step_counter=99,
            episode_length_buf=torch.full((n,), 100),
            scene=types.SimpleNamespace(terrain=terrain),
        )
        return env

    def test_counts_use_old_level_and_only_verified_success_promotes(self):
        env = self.fake_env([Result.SUCCESS, Result.FALL, Result.SUCCESS], [1, 1, 2])
        finish_geometry_episodes(env, [0, 1, 2], promote=True)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [2, 0, 2])
        self.assertEqual(env._g1_stair_curriculum_attempts.tolist(), [0, 2, 1])
        self.assertEqual(env._g1_stair_curriculum_successes.tolist(), [0, 1, 1])
        self.assertEqual(env._g1_geometry_episode.last_completed["level"].tolist(), [1, 1, 2])
        self.assertEqual(
            env._g1_geometry_episode.last_completed["upper_platform_contact"].tolist(),
            [True, False, True],
        )
        self.assertEqual(env._g1_geometry_episode.tracker.reason.tolist(), [0, 0, 0])

    def test_supported_platform_promotes_without_final_dwell(self):
        env = self.fake_env(
            [Result.TIMEOUT, Result.FALL],
            [1, 1],
        )
        env._g1_geometry_episode.tracker.platform_seen[:] = torch.tensor(
            [True, False]
        )

        finish_geometry_episodes(env, [0, 1], promote=True)

        self.assertEqual(
            env.scene.terrain.terrain_levels.tolist(),
            [2, 0],
        )
        self.assertEqual(
            env._g1_stair_curriculum_successes.tolist(),
            [0, 1, 0],
        )

    def test_nonflat_population_floor_limits_level1_demotion(self):
        env = self.fake_env(
            [Result.FALL] * 8,
            [1, 1, 1, 1, 0, 0, 0, 0],
        )
        finish_geometry_episodes(
            env,
            list(range(8)),
            promote=True,
            minimum_nonflat_env_fraction=0.25,
        )
        self.assertEqual(
            sum(level > 0 for level in env.scene.terrain.terrain_levels.tolist()),
            2,
        )

    def test_zero_nonflat_floor_preserves_existing_demotion(self):
        env = self.fake_env([Result.FALL, Result.FALL], [1, 1])
        finish_geometry_episodes(
            env,
            [0, 1],
            promote=True,
            minimum_nonflat_env_fraction=0.0,
        )
        self.assertEqual(
            env.scene.terrain.terrain_levels.tolist(),
            [0, 0],
        )

    def test_fixed_distribution_counts_without_moving_terrain(self):
        env = self.fake_env([Result.SUCCESS, Result.FALL], [0, 1])
        finish_geometry_episodes(env, [0, 1], promote=False)
        self.assertEqual(env.scene.terrain.calls, 0)
        metrics = env._g1_geometry_episode.metrics()
        self.assertEqual(metrics["flat_traversal_success_rate"], 1.)
        self.assertEqual(metrics["verified_stair_success_rate"], 0.)
        self.assertEqual(metrics["outcome_h04cm_upper_platform_contact_episodes"], 0)
        self.assertIsNone(metrics["outcome_h06cm_success_rate"])

    def test_initial_reset_does_not_count(self):
        env = self.fake_env([Result.RUNNING], [1])
        env.episode_length_buf.zero_()
        finish_geometry_episodes(env, [0], promote=True)
        self.assertEqual(env.scene.terrain.terrain_levels.item(), 1)
        self.assertEqual(env._g1_geometry_episode.counts.sum().item(), 0)

    def test_missing_terminal_result_fails_closed(self):
        env = self.fake_env([Result.RUNNING], [1])
        with self.assertRaisesRegex(RuntimeError, "without a terminal"):
            finish_geometry_episodes(env, [0])


if __name__ == "__main__":
    unittest.main()
