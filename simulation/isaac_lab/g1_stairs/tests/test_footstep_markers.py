import math
import unittest

import torch

from simulation.isaac_lab.g1_stairs import footstep_markers as fm


class MarkerHelperTests(unittest.TestCase):
    def test_force_bands(self):
        bw = fm.BODY_WEIGHT_N
        bands = fm.force_band(torch.tensor([0.5 * bw, 1.19 * bw, 1.5 * bw, 2.5 * bw]))
        self.assertEqual(bands.tolist(), [0, 0, 1, 2])

    def test_target_marker_index(self):
        self.assertEqual(fm.target_marker_index(torch.tensor([0, 1]), 0).tolist(), [0, 1])
        self.assertEqual(fm.target_marker_index(torch.tensor([0, 1]), 1).tolist(), [2, 3])

    def test_yaw_quat(self):
        q = fm.yaw_quat(torch.tensor([0.0, math.pi]))
        self.assertTrue(torch.allclose(q[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(q[1], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
