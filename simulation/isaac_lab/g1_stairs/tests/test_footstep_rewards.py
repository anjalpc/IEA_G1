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
    def test_contact_clock(self):
        force=torch.tensor([[100.,0.]])
        sole=torch.zeros((1,2,3))
        vel=torch.tensor([[[0.,0.,0.],[.4,0.,0.]]])
        targets=torch.tensor([[[1.,0.,0.,0.],[0.,0.,0.,0.]]])
        feet=torch.tensor([[1,0]])
        valid=torch.tensor([[True,True]])
        swing=torch.tensor([[0.,1.]])
        stance=torch.tensor([[1.,0.]])
        def score():
            return clock_contact_scores(force,sole,vel,targets,feet,valid,swing,stance)
        a,b=score()
        self.assertEqual((a.item(),b.item()),(1.,1.))
        vel[0,1]=torch.tensor([-.4,0.,0.])
        self.assertEqual(score()[1].item(),0.)
        vel[0,1]=torch.tensor([0.,.4,0.])
        self.assertEqual(score()[1].item(),0.)
        vel[0,1]=torch.tensor([.4,0.,0.])
        valid[0,0]=False
        self.assertEqual(score()[1].item(),0.)
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
