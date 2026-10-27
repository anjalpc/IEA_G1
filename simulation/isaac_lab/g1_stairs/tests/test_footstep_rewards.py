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
        speed=torch.tensor([[0.,.4]])
        swing=torch.tensor([[0.,1.]])
        stance=torch.tensor([[1.,0.]])
        a,b=clock_contact_scores(force,speed,swing,stance)
        self.assertEqual((a.item(),b.item()),(1.,1.))
    def test_reject_legacy_rewards(self):
        with self.assertRaises(RuntimeError):
            assert_paper_only_rewards(["track_lin_vel_xy"])
        self.assertTrue(assert_paper_only_rewards(["footstep_acquisition"]))
if __name__=="__main__":
    unittest.main()
