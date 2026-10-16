"""Pure CPU checks for the strict physical stair-gait contract."""

import unittest

import torch

from simulation.isaac_lab.g1_stairs.strict_stair_gait import StrictStairGait


class StrictGaitTests(unittest.TestCase):
    def setUp(self):
        self.g = StrictStairGait(
            1, "cpu", first_riser=1.0, tread=0.25,
            steps=3, half_width=0.6, dt=0.02,
        )
        # [xmin, xmax, ymin, ymax, zmin, zmax]
        self.b = torch.tensor([[[-0.10, 0.08, 0.05, 0.12, 0.0, 0.0],
                                [-0.10, 0.08, -0.12, -0.05, 0.0, 0.0]]])
        self.f = torch.tensor([[[0.0, 0.0, 150.0], [0.0, 0.0, 150.0]]])
        self.v = torch.zeros(1, 2)
        self.h = torch.tensor([0.04])
        self.update()

    def update(self):
        return self.g.update(
            self.b, self.f, self.v,
            torch.zeros(1, 2), torch.zeros(1, 2), self.h,
        )

    def swing(self, foot, target):
        start = 1.0 + 0.25 * target
        top = 0.04 * (target + 1)
        self.f[0, foot, 2] = 0.0
        self.b[0, foot, [0, 1, 4, 5]] = torch.tensor(
            [start - 0.08, start + 0.10, top + 0.02, top + 0.02]
        )
        self.update()
        # Perfectly centered 18 cm sole -> 3.5 cm margins.
        self.b[0, foot, [0, 1, 4, 5]] = torch.tensor(
            [start + 0.035, start + 0.215, top, top]
        )

    def land(self, foot):
        self.f[0, foot, 2] = 150.0
        for _ in range(5):
            self.update()

    def test_alternating_three_tread_climb(self):
        for target, foot in enumerate((0, 1, 0)):
            self.swing(foot, target)
            self.land(foot)
            self.assertEqual(self.g.count.item(), target + 1)
            self.assertFalse(self.g.failed.item())
        self.assertTrue(self.update()[1].item())

    def test_same_foot_next_tread_is_rejected(self):
        self.swing(0, 0); self.land(0)
        self.swing(0, 1); self.land(0)
        self.assertTrue(self.g.failed.item())
        self.assertEqual(self.g.reason.item(), StrictStairGait.BAD_LANDING)

    def test_old_tread_replant_is_rejected(self):
        self.swing(0, 0); self.land(0)
        self.swing(1, 1)
        self.b[0, 1, [0, 1, 4, 5]] = torch.tensor([1.035, 1.215, 0.04, 0.04])
        self.land(1)
        self.assertTrue(self.g.failed.item())

    def test_edge_biased_landing_is_rejected(self):
        self.swing(0, 0)
        # 1.6 cm front margin: physically fits but violates strict 2.5 cm margin.
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([1.054, 1.234, 0.04, 0.04])
        self.land(0)
        self.assertTrue(self.g.failed.item())

    def test_bridge_is_rejected(self):
        self.f[0, 0, 2] = 0.0
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([0.92, 1.10, 0.02, 0.06])
        self.update()
        self.f[0, 0, 2] = 150.0
        self.update()
        self.assertTrue(self.g.failed.item())
        self.assertEqual(self.g.reason.item(), StrictStairGait.EDGE_BRIDGE)

    def test_below_edge_riser_strike_is_rejected(self):
        # Sole is entirely below the first tread top but intersects the riser
        # plane with substantial horizontal contact force.
        self.f[0, 0] = torch.tensor([60.0, 0.0, 5.0])
        self.b[0, 0] = torch.tensor([0.94, 1.06, 0.05, 0.12, 0.005, 0.030])
        self.update()
        self.assertTrue(self.g.failed.item())
        self.assertEqual(self.g.reason.item(), StrictStairGait.RISER_STRIKE)

    def test_approach_landing_consumes_stale_clearance(self):
        # Clear the first riser during a hop, then land back on the approach.
        self.f[0, 0, 2] = 0.0
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([0.92, 1.10, 0.06, 0.06])
        self.update()
        self.assertTrue(bool(self.g.cleared[0, 0]))
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([-0.10, 0.08, 0.0, 0.0])
        self.f[0, 0, 2] = 150.0
        self.update()
        self.assertFalse(bool(self.g.cleared[0, 0]))

    def test_loss_of_opposite_support_during_swing_is_rejected(self):
        self.f[0, 0, 2] = 0.0
        self.f[0, 1, 2] = 0.0
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([0.92, 1.10, 0.06, 0.06])
        self.update()
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([1.035, 1.215, 0.04, 0.04])
        self.f[0, 0, 2] = 150.0
        self.f[0, 1, 2] = 150.0
        self.update()
        self.assertTrue(self.g.failed.item())
        self.assertEqual(self.g.reason.item(), StrictStairGait.BAD_LANDING)

    def test_bad_landing_trace_exposes_speed_failure(self):
        self.swing(0, 0)
        self.v[0, 0] = 0.50
        self.f[0, 0, 2] = 150.0
        self.update()

        self.assertTrue(self.g.failed.item())
        self.assertEqual(self.g.reason.item(), StrictStairGait.BAD_LANDING)

        tr = self.g.trace
        self.assertTrue(bool(tr["fresh"][0, 0]))
        self.assertTrue(bool(tr["relevant"][0, 0]))
        self.assertEqual(int(tr["number"][0]), 1)
        self.assertEqual(int(tr["selected_foot"][0]), 0)
        self.assertTrue(bool(tr["expected"][0]))
        self.assertFalse(bool(tr["good"][0]))
        self.assertTrue(bool(tr["cleared"][0]))
        self.assertTrue(bool(tr["support_valid"][0]))
        self.assertFalse(bool(tr["valid_landing"][0]))
        self.assertTrue(bool(tr["bad_landing"][0]))
        self.assertFalse(bool(tr["good_speed"][0]))

    def test_third_acquisition_must_land_on_third_tread(self):
        self.swing(0, 0); self.land(0)
        self.swing(1, 1); self.land(1)
        self.swing(0, 2)
        # Land far onto the upper platform instead of third tread.
        self.b[0, 0, [0, 1, 4, 5]] = torch.tensor([1.80, 1.98, 0.12, 0.12])
        self.land(0)
        self.assertTrue(self.g.failed.item())
        self.assertEqual(self.g.count.item(), 2)

    def test_flat_is_exempt(self):
        self.h.zero_()
        self.assertTrue(self.update()[1].item())

    def test_reset_clears_state(self):
        self.swing(0, 0); self.land(0)
        self.g.failed.fill_(True)
        self.g.reset(torch.tensor([0]))
        self.assertEqual(self.g.count.item(), 0)
        self.assertFalse(self.g.failed.item())
        self.assertEqual(self.g.last_foot.item(), -1)


if __name__ == "__main__":
    unittest.main()
