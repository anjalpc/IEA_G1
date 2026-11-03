import unittest

import torch
from torch import nn

from simulation.isaac_lab.g1_stairs.training_resume import cap_action_noise_std


class _Scalar(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = nn.Parameter(torch.tensor(std))


class _Log(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.log_std = nn.Parameter(torch.log(torch.tensor(std)))


class CapNoiseTests(unittest.TestCase):
    # joints: 0 left, 1 right, 2 centre
    partner = (1, 0, 2)

    def test_cap_then_symmetrise(self):
        p = _Scalar([0.2, 0.3, 4.0])
        old, new = cap_action_noise_std(p, 0.5, self.partner)
        self.assertEqual(old, [0.2, 0.3, 4.0])
        self.assertTrue(torch.allclose(p.std, torch.tensor([0.25, 0.25, 0.5])))
        self.assertTrue(torch.allclose(torch.tensor(new), p.std.detach()))

    def test_log_std_and_both_sides_capped(self):
        p = _Log([5.0, 3.0, 0.1])
        cap_action_noise_std(p, 0.5, self.partner)
        self.assertTrue(torch.allclose(p.log_std.exp(), torch.tensor([0.5, 0.5, 0.1])))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            cap_action_noise_std(_Scalar([1.0, 1.0, 1.0]), 0.0, self.partner)
        with self.assertRaises(ValueError):
            cap_action_noise_std(_Scalar([1.0, 1.0]), 0.5, self.partner)


if __name__ == "__main__":
    unittest.main()
