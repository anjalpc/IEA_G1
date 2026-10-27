import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_tread import qualified_tread_mask

class TreadTests(unittest.TestCase):
    def setup(self):
        bounds=torch.tensor([[[1.06,1.19,.04,.12,.2,.2],[0.,.1,-.12,-.04,0.,0.]]])
        force=torch.tensor([[110.,110.]])
        targets=torch.tensor([[[1.125,.08,.2,0.],[1.375,-.08,.4,0.]]])
        feet=torch.tensor([[0,1]])
        valid=torch.ones((1,2),dtype=torch.bool)
        return dict(sole_bounds=bounds,vertical_force=force,targets=targets,feet=feet,
                    valid=valid, first_riser_x=torch.tensor([1.]),
                    tread_depth=torch.tensor([.25]),riser_height=torch.tensor([.2]),
                    step_count=3,half_width=.6)
    def test_assigned_sole_on_tread(self):
        x=self.setup()
        self.assertEqual(qualified_tread_mask(**x).tolist(),[[True,False]])
    def test_riser_overlap_rejected(self):
        x=self.setup();x['sole_bounds'][0,0,0]=.99
        self.assertEqual(qualified_tread_mask(**x).tolist(),[[False,False]])
    def test_wrong_height_rejected(self):
        x=self.setup();x['sole_bounds'][0,0,4:]=.05
        self.assertEqual(qualified_tread_mask(**x).tolist(),[[False,False]])
    def test_unloaded_rejected(self):
        x=self.setup();x['vertical_force'][0,0]=0
        self.assertEqual(qualified_tread_mask(**x).tolist(),[[False,False]])
