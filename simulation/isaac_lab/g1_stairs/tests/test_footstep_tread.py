import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_tread import qualified_tread_mask, qualified_flat_mask

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

    def test_centre_on_tread_passes_only_lenient_check(self):
        # Tread 1 spans [1.0, 1.25]; sole [1.02, 1.20] would pass both,
        # sole [0.98, 1.16] (heel over the riser edge, centre 1.07) only lenient.
        x=self.setup();x['sole_bounds'][0,0,0:2]=torch.tensor([.98,1.16])
        self.assertEqual(qualified_tread_mask(**x).tolist(),[[False,False]])
        self.assertEqual(qualified_tread_mask(**x,full_sole=False).tolist(),[[True,False]])

    def test_lenient_check_still_needs_correct_tread_level_and_load(self):
        x=self.setup();x['sole_bounds'][0,0,0:2]=torch.tensor([1.27,1.45])
        self.assertEqual(qualified_tread_mask(**x,full_sole=False).tolist(),[[False,False]])
        x=self.setup();x['sole_bounds'][0,0,4:]=.05
        self.assertEqual(qualified_tread_mask(**x,full_sole=False).tolist(),[[False,False]])
        x=self.setup();x['vertical_force'][0,0]=0
        self.assertEqual(qualified_tread_mask(**x,full_sole=False).tolist(),[[False,False]])


class FlatLandingTests(unittest.TestCase):
    def inputs(self):
        return dict(
            sole_centre=torch.tensor([[
                [0.20, 0.08, 0.015],
                [0.02, -0.08, 0.015],
            ]]),
            sole_bounds=torch.tensor([[
                [0.11, 0.29, 0.045, 0.115, 0.0, 0.03],
                [-0.07, 0.11, -0.115, -0.045, 0.0, 0.03],
            ]]),
            vertical_force=torch.tensor([[90.0, 90.0]]),
            targets=torch.tensor([[
                [0.20, 0.08, 0.0, 0.0],
                [0.40, -0.08, 0.0, 0.0],
            ]]),
            feet=torch.tensor([[0, 1]]),
            valid=torch.tensor([[True, True]]),
            half_width=0.60,
        )

    def test_assigned_foot_qualifies(self):
        self.assertEqual(
            qualified_flat_mask(**self.inputs()).tolist(),
            [[True, False]],
        )

    def test_distant_foot_rejected(self):
        args = self.inputs()
        args["sole_centre"][0, 0, 0] = 0.40
        self.assertEqual(
            qualified_flat_mask(**args).tolist(),
            [[False, False]],
        )

    def test_airborne_foot_rejected(self):
        args = self.inputs()
        args["sole_bounds"][0, 0, 4] = 0.10
        self.assertEqual(
            qualified_flat_mask(**args).tolist(),
            [[False, False]],
        )

    def test_unloaded_foot_rejected(self):
        args = self.inputs()
        args["vertical_force"][0, 0] = 0.0
        self.assertEqual(
            qualified_flat_mask(**args).tolist(),
            [[False, False]],
        )

    def test_outside_course_rejected(self):
        args = self.inputs()
        args["sole_bounds"][0, 0, 3] = 0.70
        self.assertEqual(
            qualified_flat_mask(**args).tolist(),
            [[False, False]],
        )

    def test_invalid_target_rejected(self):
        args = self.inputs()
        args["valid"][0, 0] = False
        self.assertEqual(
            qualified_flat_mask(**args).tolist(),
            [[False, False]],
        )
