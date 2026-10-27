import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_rewards import (
    footstep_target_score,clock_contact_scores)
from simulation.isaac_lab.g1_stairs.footstep_rewards import assert_paper_only_rewards

class PaperScoreTests(unittest.TestCase):
    def test_no_positive_without_acquisition(self):
        sole=torch.tensor([[[1.125,.08,.2],[0.,-.08,0.]]])
        targets=torch.tensor([[[1.125,.08,.2,0.],[1.375,-.08,.4,0.]]])
        feet=torch.tensor([[0,1]])
        valid=torch.tensor([[True,True]])
        x=footstep_target_score(sole,targets,feet,valid,torch.tensor([False]))
        self.assertEqual(x.item(),0.0)
        y=footstep_target_score(sole,targets,feet,valid,torch.tensor([True]))
        self.assertAlmostEqual(y.item(),1.0)
    def test_signed_gait_rewards_both_feet(self):
        force = torch.tensor([[100., 0.]])
        sole = torch.zeros((1, 2, 3))
        velocity = torch.tensor([[[0., 0., 0.], [.3, 0., 0.]]])
        targets = torch.zeros((1, 2, 4))
        feet = torch.tensor([[0, 1]])
        valid = torch.tensor([[True, True]])
        swing = torch.tensor([[0., 1.]])
        stance = torch.tensor([[1., 0.]])

        def scores():
            return clock_contact_scores(
                force, sole, velocity, targets, feet, valid, swing, stance
            )

        contact, speed = scores()
        self.assertAlmostEqual(contact.item(), .5)
        self.assertAlmostEqual(speed.item(), .5)

        # Right swing still receives credit while LEFT target is assigned.
        feet[0, 0] = 0
        self.assertAlmostEqual(scores()[1].item(), .5)

        # Contact on the scheduled swing foot is penalized.
        force[0, 1] = 100.
        self.assertAlmostEqual(scores()[0].item(), 0.)

        # Stance-foot motion reduces the foot-speed reward.
        velocity[0, 0, 0] = .3
        self.assertAlmostEqual(scores()[1].item(), 0.)

        # Mirror both feet: same physical scores.
        force = force.flip(1)
        velocity = velocity.flip(1)
        swing = swing.flip(1)
        stance = stance.flip(1)
        self.assertAlmostEqual(scores()[0].item(), 0.)
        self.assertAlmostEqual(scores()[1].item(), 0.)

        valid[0, 0] = False
        self.assertEqual(scores()[0].item(), 0.)
        self.assertEqual(scores()[1].item(), 0.)

    def test_double_support_penalizes_foot_motion(self):
        force = torch.tensor([[100., 100.]])
        sole = torch.zeros((1, 2, 3))
        velocity = torch.zeros((1, 2, 3))
        targets = torch.zeros((1, 2, 4))
        feet = torch.tensor([[0, 1]])
        valid = torch.tensor([[True, True]])
        swing = torch.zeros((1, 2))
        stance = torch.ones((1, 2))

        def scores():
            return clock_contact_scores(
                force, sole, velocity, targets, feet, valid, swing, stance
            )

        self.assertAlmostEqual(scores()[0].item(), 1.)
        self.assertAlmostEqual(scores()[1].item(), 0.)

        velocity[:, :, 0] = .3
        self.assertAlmostEqual(scores()[1].item(), -1.)

    def test_new_progress_no_recycling(self):
        from simulation.isaac_lab.g1_stairs.footstep_rewards import new_target_progress
        previous=torch.tensor([1.0])
        best=torch.tensor([1.0])
        valid=torch.tensor([True])
        total=0.0
        for current in (0.8, 0.95, 0.8, 0.75):
            score,best=new_target_progress(previous,torch.tensor([current]),best,valid)
            total+=score.item()
            previous=torch.tensor([current])
        self.assertAlmostEqual(total,2.5,places=5)
        self.assertAlmostEqual(best.item(),0.75,places=5)

    def test_new_progress_invalid_is_zero(self):
        from simulation.isaac_lab.g1_stairs.footstep_rewards import new_target_progress
        score,best=new_target_progress(torch.tensor([1.]),torch.tensor([.5]),
                                      torch.tensor([1.]),torch.tensor([False]))
        self.assertEqual(score.item(),0.)
        self.assertEqual(best.item(),1.)

    def test_reject_legacy_rewards(self):
        with self.assertRaises(RuntimeError):
            assert_paper_only_rewards(["track_lin_vel_xy"])
        self.assertTrue(assert_paper_only_rewards(["footstep_acquisition"]))
if __name__=="__main__":
    unittest.main()
