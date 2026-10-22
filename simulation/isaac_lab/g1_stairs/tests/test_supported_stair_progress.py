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
        step(0,1)
        self.assertEqual(s.next_step.item(),1)
        step(1,1)
        self.assertEqual(s.next_step.item(),2)
        step(0,2)
        self.assertTrue(s.alternating_complete.item())
        s.reset(torch.tensor([0]))
        self.assertFalse(s.alternating_complete.item())

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
