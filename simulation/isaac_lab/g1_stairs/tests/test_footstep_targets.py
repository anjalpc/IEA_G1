"""No Isaac Sim/GPU required for geometry and metrics-contract validation."""
import math
import unittest

from simulation.isaac_lab.g1_stairs.footstep_targets import (
    StairCourse, FootstepProgress, planned_targets, relative_target, metrics,
)


class FootstepPlannerTests(unittest.TestCase):
    def setUp(self):
        self.course = StairCourse(1.0, .25, .20, 3)

    def test_stairs_follow_course_geometry_and_alternate(self):
        targets = planned_targets(self.course, 0)
        self.assertEqual([t.foot for t in targets], [0, 1, 0, 1, 0])
        self.assertEqual([t.terrain_level for t in targets], [1, 2, 3, 3, 3])
        self.assertEqual([round(t.z, 2) for t in targets], [.2, .4, .6, .6, .6])
        self.assertAlmostEqual(targets[0].x, 1.125)
        self.assertAlmostEqual(targets[2].x, 1.625)
        self.assertGreater(targets[-1].x, targets[2].x)

    def test_plan_adapts_no_hardcoded_course_dimensions(self):
        course = StairCourse(.8, .3, .1, 4, tread_margin=.01)
        targets = planned_targets(course, 1)
        self.assertEqual(len(targets), 6)
        self.assertAlmostEqual(targets[0].x, .95)
        self.assertAlmostEqual(targets[3].z, .4)
        self.assertEqual(targets[0].foot, 1)

    def test_robot_relative_coordinates(self):
        t = planned_targets(self.course, 0)[0]
        x, y, z, yaw = relative_target(t, 0., 0., math.pi/2)
        self.assertAlmostEqual(x, t.y)
        self.assertAlmostEqual(y, -t.x)
        self.assertAlmostEqual(z, t.z)
        self.assertAlmostEqual(yaw, -math.pi/2)

    def test_progress_requires_correct_foot_level_support_and_target(self):
        p = FootstepProgress(planned_targets(self.course, 0))
        self.assertFalse(p.touchdown(foot=1, level=1, within_target=True, supported=True))
        self.assertEqual(p.wrong_foot, 1)
        self.assertFalse(p.touchdown(foot=0, level=1, within_target=True, supported=False))
        self.assertFalse(p.touchdown(foot=0, level=0, within_target=True, supported=True))
        self.assertTrue(p.touchdown(foot=0, level=1, within_target=True, supported=True))
        self.assertFalse(p.touchdown(foot=0, level=1, within_target=True, supported=True))
        self.assertTrue(p.touchdown(foot=1, level=2, within_target=True, supported=True))
        self.assertEqual(p.next_index, 2)
        self.assertGreater(p.same_level_replants, 0)

    def test_reset_and_metric_denominators(self):
        p = FootstepProgress(planned_targets(self.course, 0))
        self.assertIsNone(metrics(p)['footstep_target_completion_rate'])
        p.touchdown(foot=0, level=1, within_target=True, supported=True)
        self.assertEqual(metrics(p)['footstep_target_completion_rate'], 1.)
        p.reset()
        self.assertEqual(p.next_index, 0)
        self.assertEqual(p.attempted, 0)
        self.assertEqual(metrics(p)['metric_contract'], 'footstep_target_v1')


if __name__ == '__main__':
    unittest.main()
