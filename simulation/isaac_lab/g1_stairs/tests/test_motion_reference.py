"""Run with Isaac Python; no Kit startup required."""
import unittest
import torch
from simulation.isaac_lab.g1_stairs.motion_reference import (
    sole_state, swing_reference, bounded_square, normalized_huber,
    target_support_mask, transition_quality_escrow_update,
)


class TestMotionReference(unittest.TestCase):
    def test_endpoints_and_no_wrap(self):
        start = torch.tensor([[0., .12, 0.]]).repeat(4, 1)
        end = torch.tensor([[.5, .12, .2]]).repeat(4, 1)
        out = swing_reference(start, end, torch.tensor([0., 1., 2., 10.]), .03)
        torch.testing.assert_close(out[0], start[0])
        torch.testing.assert_close(out[1:], end[1:])

    def test_swing_overlaps_forward_and_vertical_motion(self):
        u = torch.tensor([.10, .20, .50, .80, .90])
        start = torch.tensor([[0., .12, 0.]]).repeat(len(u), 1)
        end = torch.tensor([[.5, .12, .2]]).repeat(len(u), 1)
        out = swing_reference(start, end, u, .03)
        self.assertGreater(float(out[0, 0]), 0.0)
        self.assertGreater(float(out[0, 2]), 0.0)
        self.assertLess(float(out[0, 0]), float(out[1, 0]))
        self.assertLess(float(out[1, 0]), float(out[2, 0]))
        self.assertGreater(float(out[2, 2]), .20)
        self.assertTrue(bool((out[1:, 0] >= out[:-1, 0]).all()))
        torch.testing.assert_close(out[:, 1], start[:, 1])

    def test_reference_endpoint_velocity_and_acceleration_are_smooth(self):
        eps = 1e-3
        u = torch.tensor([0., eps, 2*eps, 1-2*eps, 1-eps, 1.])
        start = torch.zeros(len(u), 3)
        end = torch.tensor([[.25, 0., .20]]).repeat(len(u), 1)
        out = swing_reference(start, end, u, .03)
        v0 = (out[1]-out[0])/eps
        v1 = (out[-1]-out[-2])/eps
        a0 = (out[2]-2*out[1]+out[0])/(eps*eps)
        a1 = (out[-1]-2*out[-2]+out[-3])/(eps*eps)
        self.assertLess(float(torch.linalg.vector_norm(v0)), .05)
        self.assertLess(float(torch.linalg.vector_norm(v1)), .05)
        self.assertLess(float(torch.linalg.vector_norm(a0)), 2.0)
        self.assertLess(float(torch.linalg.vector_norm(a1)), 2.0)

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

    def test_huber_remains_informative_for_large_tracking_errors(self):
        out = normalized_huber(torch.tensor([0., .03, .06, .12, .24]), .06)
        self.assertEqual(float(out[0]), 0.)
        self.assertTrue(bool((out[1:] > out[:-1]).all()))
        self.assertGreater(float(out[-1]), 3.0)

    def test_target_support_requires_intended_tread_and_load(self):
        bounds = torch.tensor([[1.06, 1.20, -.10, .10, .04, .04],
                               [.82, .96, -.10, .10, .00, .00],
                               [1.06, 1.20, -.10, .10, .04, .04]])
        force = torch.tensor([100., 100., 10.])
        result = target_support_mask(bounds, force,
            tread_start=torch.tensor([1.,1.,1.]),
            tread_end=torch.tensor([1.25,1.25,1.25]),
            tread_top=torch.tensor([.04,.04,.04]),
            terrain_half_width=.60, min_force_n=30., height_tolerance_m=.025)
        self.assertEqual(result.tolist(), [True, False, False])

    def test_quality_escrow_releases_only_direct_and_discards_failures(self):
        z=torch.zeros(2)
        total,count,credit=transition_quality_escrow_update(
            z,z,torch.tensor([.8,.8]),torch.ones(2),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.long))
        self.assertEqual(count.tolist(), [1.,1.])
        total,count,credit=transition_quality_escrow_update(
            total,count,torch.tensor([.6,.6]),torch.tensor([.9,.9]),torch.ones(2,dtype=torch.bool),torch.tensor([1,2]))
        self.assertGreater(float(credit[0]),0.)
        self.assertEqual(float(credit[1]),0.)
        self.assertEqual(count.tolist(), [0.,0.])


if __name__ == '__main__':
    unittest.main()
