import unittest

from simulation.isaac_lab.g1_stairs.showcase_camera import (
    KEYFRAMES, camera_pose, field_geometry, start_delay_steps,
)

GROUP = (2.0, 0.0, 4.8, 0.0)
HERO = (2.5, -4.0)


class ShowcaseCameraTests(unittest.TestCase):
    def test_keyframes_are_hit_exactly(self):
        for t, frame, eye, look, _ in KEYFRAMES:
            e, l = camera_pose(t, GROUP, HERO)
            if frame == "hero":
                self.assertAlmostEqual(e[0], HERO[0] + eye[0])
                self.assertAlmostEqual(e[1], HERO[1] + eye[2])
                self.assertAlmostEqual(l[0], HERO[0] + look[0])
            else:
                self.assertAlmostEqual(e[0], GROUP[0] + eye[0])
                self.assertAlmostEqual(e[1], eye[1] * 4.8 + eye[2])
            self.assertAlmostEqual(e[2], eye[3])

    def test_path_is_continuous_and_clamped(self):
        prev = camera_pose(0.0, GROUP, HERO)[0]
        for k in range(1, 701):                         # 14 s at 50 fps
            eye = camera_pose(k / 700, GROUP, HERO)[0]
            step = sum((a - b) ** 2 for a, b in zip(eye, prev)) ** 0.5
            self.assertLess(step, 0.1)                  # < 5 m/s at 50 fps
            prev = eye
        self.assertEqual(camera_pose(-1, GROUP, HERO), camera_pose(0, GROUP, HERO))
        self.assertEqual(camera_pose(2, GROUP, HERO), camera_pose(1, GROUP, HERO))

    def test_camera_stays_above_ground(self):
        for k in range(101):
            self.assertGreater(camera_pose(k / 100, GROUP, HERO)[0][2], 0.5)

    def test_spread_moves_only_wide_shots_out(self):
        for t, _, _, _, wide in KEYFRAMES:
            near = camera_pose(t, GROUP, HERO)[0]
            far = camera_pose(t, GROUP, HERO, spread=2.0)[0]
            moved = max(abs(a - b) for a, b in zip(near, far)) > 1e-9
            self.assertEqual(moved, wide)

    def test_field_geometry(self):
        centre, half, spread = field_geometry([(0.0, 0.0), (1.0, 2.4), (0.5, 4.8)])
        self.assertAlmostEqual(centre, 2.4)
        self.assertAlmostEqual(half, 3.0)
        self.assertAlmostEqual(spread, 1.1)
        self.assertAlmostEqual(field_geometry([(0, y * 2.4) for y in range(16)])[2], (18.0 + 0.6) / 4.8)

    def test_start_delays_random_first_at_zero(self):
        d = start_delay_steps(16, 4.0, 0.02, seed=3)
        self.assertEqual(min(d), 0)
        self.assertLessEqual(max(d), 200)
        self.assertGreater(max(d), 100)                 # spread over seconds, not 0.25 s steps
        self.assertEqual(d, start_delay_steps(16, 4.0, 0.02, seed=3))
        gaps = sorted(d)
        self.assertGreater(len({b - a for a, b in zip(gaps, gaps[1:])}), 3)   # irregular
        self.assertEqual(start_delay_steps(4, 0.0, 0.02, seed=1), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
