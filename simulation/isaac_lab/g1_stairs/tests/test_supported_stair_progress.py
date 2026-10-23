import unittest
import torch
from simulation.isaac_lab.g1_stairs.supported_stair_progress import SupportedAscent

class SupportedAscentTests(unittest.TestCase):
    def test_missing_sole_geometry_is_a_noop(self):
        s = SupportedAscent(1, "cpu", dwell_s=0.04)
        s.pending_time[:] = 0.02

        credited = s.update(
            None,
            torch.tensor([[100.0, 0.0]]),
            torch.tensor([0.0]),
            1.0,
            0.25,
            0.5,
            0.02,
            torch.tensor([True]),
        )

        self.assertFalse(credited.any())
        self.assertEqual(s.pending_time.sum().item(), 0.0)
        self.assertEqual(s.next_step.item(), 0)

    def test_unique_alternation_and_reset(self):
        s = SupportedAscent(1, "cpu", dwell_s=.04)
        b = torch.zeros((1, 2, 6))
        f = torch.tensor([[100., 0.]])
        h = torch.tensor([.04])
        def step(foot, stair):
            f[:] = 0
            f[0,foot] = 100
            b[:] = 0
            left = 1.0 + stair*.25
            b[0,foot] = torch.tensor([left+.02,left+.20,-.03,.03,(stair+1)*.04,(stair+1)*.04])
            for _ in range(2):
                s.update(b,f,h,1.,.25,.5,.02,torch.tensor([True]))
        step(0,0)
        self.assertEqual(s.next_step.item(),1)
        self.assertTrue(s.last_credit.item())
        s.update(b, f, h, 1., .25, .5, .02, torch.tensor([True]))
        self.assertFalse(s.last_credit.item())
        step(1,1)
        self.assertEqual(s.next_step.item(),2)
        step(0,2)
        self.assertTrue(s.alternating_complete.item())
        s.reset(torch.tensor([0]))
        self.assertFalse(s.alternating_complete.item())
        self.assertFalse(s.sequence_failed.item())

    def test_wrong_foot_on_next_tread_invalidates_sequence(self):
        s = SupportedAscent(1, "cpu", dwell_s=0.04)
        b = torch.zeros((1, 2, 6))
        f = torch.zeros((1, 2))
        h = torch.tensor([0.04])
        active = torch.tensor([True])

        def dwell(foot, stair):
            f.zero_()
            f[0, foot] = 100.0
            b.zero_()
            left = 1.0 + stair * 0.25
            b[0, foot] = torch.tensor(
                [left + .02, left + .20, -.03, .03,
                 (stair + 1) * .04, (stair + 1) * .04]
            )
            for _ in range(2):
                s.update(b, f, h, 1., .25, .5, .02, active)

        dwell(0, 0)
        dwell(0, 1)
        self.assertTrue(s.sequence_failed.item())
        self.assertTrue(s.last_invalid.item())
        self.assertEqual(s.next_step.item(), 1)
        dwell(1, 1)
        self.assertEqual(s.next_step.item(), 1)
        self.assertFalse(s.last_credit.item())

    def test_invalid_support_cannot_progress(self):
        s = SupportedAscent(1,"cpu",dwell_s=.04)
        b = torch.tensor([[[.90,1.08,-.03,.03,.04,.04], [0.,0.,0.,0.,0.,0.]]])
        f = torch.tensor([[200.,0.]])
        for _ in range(10):
            s.update(b,f,torch.tensor([.04]),1.,.25,.5,.02,torch.tensor([True]))
        self.assertEqual(s.next_step.item(),0)

    def test_partial_supported_tread_can_advance_but_not_strike(self):
        s = SupportedAscent(1, "cpu", dwell_s=0.04)
        b = torch.tensor([[[1.078, 1.262, -.03, .03, .040, .080],
                           [0., 0., 0., 0., 0., 0.]]])
        f = torch.tensor([[100., 0.]])
        h = torch.tensor([.04])
        active = torch.tensor([True])
        for _ in range(2):
            s.update(b, f, h, 1., .25, .5, .02, active,
                     invalid_contact=torch.tensor([True]))
        self.assertEqual(s.next_step.item(), 0)
        for _ in range(2):
            s.update(b, f, h, 1., .25, .5, .02, active,
                     invalid_contact=torch.tensor([False]))
        self.assertEqual(s.next_step.item(), 1)

    def test_riser_overlap_without_enough_tread_cannot_progress(self):
        s = SupportedAscent(1, "cpu", dwell_s=0.04)
        b = torch.tensor([[[.90, 1.08, -.03, .03, .04, .04],
                           [0., 0., 0., 0., 0., 0.]]])
        for _ in range(3):
            s.update(b, torch.tensor([[200., 0.]]), torch.tensor([.04]),
                     1., .25, .5, .02, torch.tensor([True]))
        self.assertEqual(s.next_step.item(), 0)

    def test_credit_requires_sustained_weight_transfer_when_configured(self):
        s = SupportedAscent(
            1, "cpu", dwell_s=0.04, minimum_load_share=0.45,
            transfer_margin_m=0.05,
            maximum_roll_pitch_rad=0.30,
        )
        bounds = torch.tensor([[[1.02, 1.20, -.03, .03, .04, .04],
                                [0., 0., 0., 0., 0., 0.]]])
        active = torch.tensor([True])
        height = torch.tensor([.04])
        controlled = dict(roll_pitch=torch.zeros((1, 2)))

        for _ in range(3):
            s.update(bounds, torch.tensor([[40., 160.]]), height,
                     1., .25, .5, .02, active, body_x=torch.tensor([1.10]),
                     **controlled)
        self.assertEqual(s.next_step.item(), 0)

        # Adequate load without body transfer remains provisional.
        for _ in range(2):
            s.update(bounds, torch.tensor([[100., 100.]]), height,
                     1., .25, .5, .02, active, body_x=torch.tensor([1.04]),
                     **controlled)
        self.assertEqual(s.next_step.item(), 0)
        self.assertEqual(s.candidate_foot.item(), 0)

        # Credit only after placement, load and body transfer persist together.
        for _ in range(2):
            s.update(bounds, torch.tensor([[100., 100.]]), height,
                     1., .25, .5, .02, active, body_x=torch.tensor([1.05]),
                     **controlled)
        self.assertEqual(s.next_step.item(), 1)

    def test_latched_foot_cannot_be_replaced_by_trailing_foot(self):
        s = SupportedAscent(
            1, "cpu", dwell_s=.04, touchdown_dwell_s=.04,
            minimum_load_share=.45, transfer_margin_m=.05,
        )
        bounds = torch.tensor([[[1.02, 1.20, -.03, .03, .04, .04],
                                [0., 0., 0., 0., 0., 0.]]])
        forces = torch.tensor([[100., 100.]])
        kwargs = dict(body_x=torch.tensor([1.02]),
                      roll_pitch=torch.zeros((1, 2)))
        for _ in range(2):
            s.update(bounds, forces, torch.tensor([.04]), 1., .25, .5,
                     .02, torch.tensor([True]), **kwargs)
        self.assertEqual(s.candidate_foot.item(), 0)

        bounds[0, 1] = bounds[0, 0]
        s.update(bounds, forces, torch.tensor([.04]), 1., .25, .5,
                 .02, torch.tensor([True]), **kwargs)
        self.assertTrue(s.sequence_failed.item())
        self.assertFalse(s.last_credit.item())

    def test_swing_progress_is_spatial_reversible_and_expected_foot_only(self):
        s = SupportedAscent(
            1, "cpu", minimum_load_share=0.45, transfer_margin_m=0.05
        )
        bounds = torch.tensor([[[-.20, -.02, -.03, .03, .00, .00],
                                [.80, .98, -.03, .03, .02, .02]]])
        forces = torch.tensor([[100., 0.]])
        active = torch.tensor([True])
        kwargs = dict(
            body_x=torch.tensor([.8]),
            roll_pitch=torch.zeros((1, 2)),
        )

        s.update(bounds, forces, torch.tensor([.04]), 1., .25, .5, .02,
                 active, **kwargs)
        first = s.swing_potential.item()
        self.assertGreater(first, 0.0)

        bounds[0, 1, 0:2] += .10
        bounds[0, 1, 4:6] += .03
        s.update(bounds, forces, torch.tensor([.04]), 1., .25, .5, .02,
                 active, **kwargs)
        self.assertGreater(s.last_swing_progress.item(), 0.0)

        bounds[0, 1, 0:2] -= .10
        bounds[0, 1, 4:6] -= .03
        s.update(bounds, forces, torch.tensor([.04]), 1., .25, .5, .02,
                 active, **kwargs)
        self.assertLess(s.last_swing_progress.item(), 0.0)

    def test_touchdown_continues_into_reversible_body_transfer(self):
        s = SupportedAscent(
            1, "cpu", dwell_s=.20, touchdown_dwell_s=.04,
            minimum_load_share=.45, transfer_margin_m=.05,
            maximum_roll_pitch_rad=.30,
        )
        bounds = torch.tensor([[[1.02, 1.20, -.03, .03, .04, .04],
                                [.70, .88, -.03, .03, .00, .00]]])
        active = torch.tensor([True])
        common = dict(
            heights=torch.tensor([.04]), first_riser=1., tread=.25,
            half_width=.5, dt=.02, active=active,
            roll_pitch=torch.zeros((1, 2)),
        )

        # Valid touchdown latches the foot without erasing earned swing progress.
        for _ in range(2):
            s.update(bounds, torch.tensor([[60., 140.]]),
                     body_x=torch.tensor([.96]), **common)
        at_touchdown = s.swing_potential.item()
        self.assertEqual(s.candidate_foot.item(), 0)
        self.assertGreater(at_touchdown, 0.0)

        # Loading the planted foot and moving the pelvis over it is rewarded.
        s.update(bounds, torch.tensor([[100., 100.]]),
                 body_x=torch.tensor([1.03]), **common)
        transferred = s.swing_potential.item()
        self.assertGreater(transferred, at_touchdown)
        self.assertGreater(s.last_swing_progress.item(), 0.0)

        # Recoiling loses the same dense progress; it cannot be farmed.
        s.update(bounds, torch.tensor([[60., 140.]]),
                 body_x=torch.tensor([.96]), **common)
        self.assertLess(s.last_swing_progress.item(), 0.0)
