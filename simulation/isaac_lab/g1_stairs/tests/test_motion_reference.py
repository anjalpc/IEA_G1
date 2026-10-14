"""Run with Isaac Python; no Kit startup required."""
import unittest
import torch
from simulation.isaac_lab.g1_stairs.motion_reference import (
    sole_state, swing_reference, bounded_square,
)


class TestMotionReference(unittest.TestCase):
    def test_endpoints_and_no_wrap(self):
        start = torch.tensor([[0., .12, 0.]]).repeat(4, 1)
        end = torch.tensor([[.5, .12, .2]]).repeat(4, 1)
        out = swing_reference(start, end, torch.tensor([0., 1., 2., 10.]), .03)
        torch.testing.assert_close(out[0], start[0])
        torch.testing.assert_close(out[1:], end[1:])

    def test_entire_forward_sweep_is_above_tread(self):
        u = torch.linspace(.2, .8, 101)
        start = torch.tensor([[0., .12, 0.]]).repeat(len(u), 1)
        end = torch.tensor([[.5, .12, .2]]).repeat(len(u), 1)
        out = swing_reference(start, end, u, .03)
        self.assertTrue(bool((out[:, 2] >= .23-1e-6).all()))
        self.assertTrue(bool((out[1:, 0] >= out[:-1, 0]).all()))
        torch.testing.assert_close(out[:, 1], start[:, 1])

    def test_smooth_join_velocity(self):
        u = torch.tensor([.2-1e-4, .2+1e-4, .8-1e-4, .8+1e-4])
        start = torch.zeros(4, 3)
        end = torch.tensor([[.5, 0., .2]]).repeat(4, 1)
        out = swing_reference(start, end, u, .03)
        self.assertLess(float(torch.linalg.vector_norm(out[1]-out[0])/.0002), .01)
        self.assertLess(float(torch.linalg.vector_norm(out[3]-out[2])/.0002), .01)

    def test_envelope_detects_center_on_edge(self):
        p = torch.tensor([[[1., 0., .035]]])
        q = torch.tensor([[[1., 0., 0., 0.]]])
        zero = torch.zeros_like(p)
        centre, bounds, velocity = sole_state(p, q, zero, zero)
        self.assertLess(float(bounds[0, 0, 0]), 1.)
        self.assertGreater(float(bounds[0, 0, 1]), 1.)
        torch.testing.assert_close(velocity, torch.zeros_like(velocity))
        self.assertAlmostEqual(float(centre[0, 0, 2]), 0., places=6)

    def test_angular_motion_contributes_to_slip(self):
        p = torch.zeros(1, 2, 3)
        q = torch.tensor([1., 0., 0., 0.]).expand(1, 2, 4)
        w = torch.zeros_like(p)
        w[..., 2] = 2.
        _, _, velocity = sole_state(p, q, torch.zeros_like(p), w)
        self.assertGreater(float(velocity[..., :2].abs().max()), .1)

    def test_cost_positive_monotonic_bounded(self):
        out = bounded_square(torch.tensor([0., .01, .1, 1., 100.]), .05)
        self.assertEqual(float(out[0]), 0.)
        self.assertTrue(bool((out[1:] > out[:-1]).all()))
        self.assertTrue(bool((out <= 1).all()))


if __name__ == '__main__':
    unittest.main()
