"""Run with Isaac Python; no Kit startup required."""
import unittest
import torch
from simulation.isaac_lab.g1_stairs.motion_reference import (
    sole_state, swing_reference, bounded_square, normalized_huber,
    foot_riser_distance, per_foot_next_riser_distance,
    target_support_mask, provisional_supported_acquisition_mask,
    transition_quality_escrow_update,
    strict_tread_landing_mask, riser_contact_masks,
)


class TestMotionReference(unittest.TestCase):
    def test_foot_riser_distance_is_per_foot_signed_and_masked(self):
        bounds = torch.tensor([
            [[.70, .90, 0., 0., 0., 0.], [.60, .80, 0., 0., 0., 0.]],
            [[1.00, 1.20, 0., 0., 0., 0.], [1.10, 1.30, 0., 0., 0., 0.]],
        ])
        result = foot_riser_distance(
            torch.tensor([1.0, 1.25]), bounds, .25, torch.tensor([1., 0.])
        )
        torch.testing.assert_close(result[0], torch.tensor([1/7, 2/7]))
        torch.testing.assert_close(result[1], torch.zeros(2))

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

    def test_strict_landing_rejects_final_tread_overrun_and_loose_height(self):
        bounds = torch.tensor([[1.535, 1.715, -.10, -.03, .12, .12],
                               [1.535, 1.715, -.10, -.03, .135, .135],
                               [1.60, 1.78, -.10, -.03, .12, .12]])
        result = strict_tread_landing_mask(
            bounds, torch.full((3,), 100.), torch.zeros(3), torch.zeros(3),
            torch.zeros(3), tread_start=torch.full((3,), 1.50),
            tread_end=torch.full((3,), 1.75), tread_top=torch.full((3,), .12),
            lane_target=torch.full((3,), -.065), terrain_half_width=.60,
            riser_height=torch.full((3,), .04),
        )
        self.assertEqual(result.tolist(), [True, False, False])

    def test_riser_strike_detects_below_edge_face_collision(self):
        # Horizontal force while the whole sole remains below the 4 cm top.
        # Bottom points stay on the lower floor, but the 5 mm collision
        # spheres extend upward and can still strike the vertical face.
        bounds = torch.tensor([[[.94, 1.06, -.10, -.03, 0., 0.],
                                [.50, .68, .03, .10, 0., 0.]]])
        force = torch.tensor([[[60., 0., 5.], [0., 0., 100.]]])
        strike, bridge = riser_contact_masks(
            bounds, force, first_riser=1., tread=.25, steps=3,
            riser_height=torch.tensor([.04]), active=torch.tensor([True]),
        )
        self.assertTrue(bool(strike[0]))
        self.assertFalse(bool(bridge[0]))

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



    def test_per_foot_next_riser_distance_handles_straddled_riser(self):
        bounds = torch.tensor([[
            [.85, 1.03, 0., 0., 0., 0.],
            [.78, .96, 0., 0., 0., 0.],
        ]])
        result = per_foot_next_riser_distance(
            bounds,
            1.0,
            .25,
            3,
            torch.tensor([1.0]),
        )
        torch.testing.assert_close(
            result,
            torch.tensor([[.22 / .68, .04 / .68]]),
        )

    def test_per_foot_next_riser_distance_masks_edge_final_and_flat(self):
        bounds = torch.tensor([
            [
                [.82, 1.00, 0., 0., 0., 0.],
                [1.33, 1.51, 0., 0., 0., 0.],
            ],
            [
                [.82, .99, 0., 0., 0., 0.],
                [.82, .99, 0., 0., 0., 0.],
            ],
        ])
        result = per_foot_next_riser_distance(
            bounds,
            1.0,
            .25,
            3,
            torch.tensor([1.0, 0.0]),
        )
        torch.testing.assert_close(
            result,
            torch.zeros_like(result),
        )


if __name__ == '__main__':
    unittest.main()


def test_provisional_supported_acquisition_requires_intended_supported_transition():
    bounds = torch.tensor([[[1.02, 1.20, -0.04, 0.04, 0.04, 0.04],
                            [0.70, 0.88, -0.04, 0.04, 0.00, 0.00]]])
    force = torch.tensor([[60.0, 80.0]])
    event = provisional_supported_acquisition_mask(
        bounds, force,
        intended_foot=torch.tensor([0]),
        support_transition=torch.tensor([[True, False]]),
        previous_acquired_foot=torch.tensor([-1]),
        strike=torch.tensor([False]), bridge=torch.tensor([False]),
        tread_start=torch.tensor([[1.0, 1.0]]),
        tread_end=torch.tensor([[1.25, 1.25]]),
        tread_top=torch.tensor([[0.04, 0.04]]),
        terrain_half_width=0.6, min_force_n=30.0, height_tolerance_m=0.02,
    )
    assert event.tolist() == [True]


def test_provisional_supported_acquisition_rejects_wrong_or_simultaneous_foot():
    bounds = torch.tensor([[[1.02, 1.20, -0.04, 0.04, 0.04, 0.04],
                            [1.02, 1.20, -0.04, 0.04, 0.04, 0.04]]])
    force = torch.tensor([[60.0, 60.0]])
    common = dict(
        bounds=bounds, vertical_force=force, intended_foot=torch.tensor([0]),
        previous_acquired_foot=torch.tensor([-1]), strike=torch.tensor([False]),
        bridge=torch.tensor([False]), tread_start=torch.tensor([[1.0, 1.0]]),
        tread_end=torch.tensor([[1.25, 1.25]]), tread_top=torch.tensor([[0.04, 0.04]]),
        terrain_half_width=0.6, min_force_n=30.0, height_tolerance_m=0.02,
    )
    wrong = provisional_supported_acquisition_mask(
        support_transition=torch.tensor([[False, True]]), **common
    )
    simultaneous = provisional_supported_acquisition_mask(
        support_transition=torch.tensor([[True, True]]), **common
    )
    assert wrong.tolist() == [False]
    assert simultaneous.tolist() == [False]


def test_provisional_supported_acquisition_rejects_bad_support_or_contact_failure():
    bounds = torch.tensor([[[0.99, 1.20, -0.04, 0.04, 0.04, 0.04],
                            [0.70, 0.88, -0.04, 0.04, 0.00, 0.00]]])
    force = torch.tensor([[60.0, 80.0]])
    common = dict(
        bounds=bounds, vertical_force=force, intended_foot=torch.tensor([0]),
        support_transition=torch.tensor([[True, False]]),
        previous_acquired_foot=torch.tensor([-1]),
        tread_start=torch.tensor([[1.0, 1.0]]), tread_end=torch.tensor([[1.25, 1.25]]),
        tread_top=torch.tensor([[0.04, 0.04]]), terrain_half_width=0.6,
        min_force_n=30.0, height_tolerance_m=0.02,
    )
    outside = provisional_supported_acquisition_mask(
        strike=torch.tensor([False]), bridge=torch.tensor([False]), **common
    )
    bounds[:, 0, 0] = 1.02
    strike = provisional_supported_acquisition_mask(
        strike=torch.tensor([True]), bridge=torch.tensor([False]), **common
    )
    bridge = provisional_supported_acquisition_mask(
        strike=torch.tensor([False]), bridge=torch.tensor([True]), **common
    )
    assert outside.tolist() == [False]
    assert strike.tolist() == [False]
    assert bridge.tolist() == [False]


def test_provisional_supported_acquisition_rejects_same_foot_progression():
    bounds = torch.tensor([[[1.27, 1.45, -0.04, 0.04, 0.08, 0.08],
                            [0.90, 1.08, -0.04, 0.04, 0.04, 0.04]]])
    force = torch.tensor([[60.0, 80.0]])
    event = provisional_supported_acquisition_mask(
        bounds, force, intended_foot=torch.tensor([0]),
        support_transition=torch.tensor([[True, False]]),
        previous_acquired_foot=torch.tensor([0]),
        strike=torch.tensor([False]), bridge=torch.tensor([False]),
        tread_start=torch.tensor([[1.25, 1.25]]), tread_end=torch.tensor([[1.50, 1.50]]),
        tread_top=torch.tensor([[0.08, 0.08]]), terrain_half_width=0.6,
        min_force_n=30.0, height_tolerance_m=0.02,
    )
    assert event.tolist() == [False]
