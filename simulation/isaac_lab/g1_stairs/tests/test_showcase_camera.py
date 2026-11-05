import unittest

from simulation.isaac_lab.g1_stairs.showcase_camera import KEYFRAMES, camera_pose, lane_geometry, stagger_hold_steps


class ShowcaseCameraTests(unittest.TestCase):
    def test_keyframes_are_hit_exactly(self):
        for t, eye, look in KEYFRAMES:
            e, l = camera_pose(t, mean_x=2.0, centre_y=0.0, half_width=4.8)
            self.assertAlmostEqual(e[0], 2.0 + eye[0])
            self.assertAlmostEqual(e[1], eye[1] * 4.8 + eye[2])
            self.assertAlmostEqual(e[2], eye[3])
            self.assertAlmostEqual(l[0], 2.0 + look[0])

    def test_path_is_continuous_and_clamped(self):
        prev = camera_pose(0.0, 0.0, 0.0, 4.8)[0]
        for k in range(1, 501):
            eye = camera_pose(k / 500, 0.0, 0.0, 4.8)[0]
            step = sum((a - b) ** 2 for a, b in zip(eye, prev)) ** 0.5
            self.assertLess(step, 0.1)       # no jumps between frames
            prev = eye
        self.assertEqual(camera_pose(-1, 0, 0, 1), camera_pose(0, 0, 0, 1))
        self.assertEqual(camera_pose(2, 0, 0, 1), camera_pose(1, 0, 0, 1))

    def test_camera_stays_above_ground_and_off_the_lanes(self):
        for k in range(101):
            eye, look = camera_pose(k / 100, 0.0, 0.0, 4.8)
            self.assertGreater(eye[2], 0.5)
            # never inside the lane block unless high above it
            if -4.8 <= eye[1] <= 4.8:
                self.assertGreater(eye[2], 1.5)

    def test_lane_geometry(self):
        centre, half = lane_geometry([0.0, 1.2, 2.4, 3.6])
        self.assertAlmostEqual(centre, 1.8)
        self.assertAlmostEqual(half, 2.4)
        self.assertEqual(lane_geometry([5.0]), (5.0, 0.6))

    def test_stagger_is_a_shuffled_wave(self):
        hold = stagger_hold_steps(8, 0.25, 0.02, seed=1)
        self.assertEqual(sorted(hold), [round(k * 12.5) for k in range(8)])
        self.assertEqual(hold, stagger_hold_steps(8, 0.25, 0.02, seed=1))
        self.assertNotEqual(hold, sorted(hold))           # not lane order
        self.assertEqual(stagger_hold_steps(4, 0.0, 0.02, seed=1), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
