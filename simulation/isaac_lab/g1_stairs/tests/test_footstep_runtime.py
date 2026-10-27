import unittest
import torch
from simulation.isaac_lab.g1_stairs.footstep_runtime import FootstepRuntime, evaluate_footstep_sequence

class FootstepRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FootstepRuntime(num_envs=2, device='cpu', step_count=3,
                                       first_foot=torch.tensor([0, 1]))
        self.geo = dict(first_riser_x=torch.tensor([1., 1.]),
                        tread_depth=torch.tensor([.25, .25]),
                        riser_height=torch.tensor([.2, .2]),
                        centre_y=torch.zeros(2), foot_separation=torch.full((2,), .16))

    def inputs(self):
        t, f, v = self.runtime.targets(**self.geo)
        sole = torch.zeros(2, 2, 3)
        sole[torch.arange(2), f[:, 0]] = t[:, 0, :3]
        return t, f, v, sole

    def update(self, supported):
        t, f, v, sole = self.inputs()
        return self.runtime.update(targets=t, feet=f, valid=v,
                                   sole_xyz=sole, supported=supported,
                                   xy_tolerance=.04, z_tolerance=.03)

    def test_landing_advances_only_assigned_foot(self):
        self.assertEqual(self.update(torch.tensor([[False, True], [True, False]])).tolist(), [False, False])
        self.assertEqual(self.update(torch.tensor([[True, True], [True, True]])).tolist(), [True, True])
        self.assertEqual(self.runtime.target_index.tolist(), [1, 1])

    def test_supported_hold_does_not_retrigger(self):
        self.update(torch.tensor([[True, False], [False, True]]))
        self.assertEqual(self.update(torch.tensor([[True, False], [False, True]])).tolist(), [False, False])
        self.assertEqual(self.runtime.completed.tolist(), [1, 1])

    def test_distance_rejects_false_landing(self):
        t, f, v, sole = self.inputs()
        sole[:, :, 0] += .4
        result = self.runtime.update(targets=t, feet=f, valid=v,
                                     sole_xyz=sole, supported=torch.tensor([[True, False], [False, True]]),
                                     xy_tolerance=.04, z_tolerance=.03)
        self.assertEqual(result.tolist(), [False, False])
        self.assertEqual(self.runtime.target_index.tolist(), [0, 0])

    def test_partial_reset_isolated(self):
        self.update(torch.tensor([[True, False], [False, True]]))
        self.runtime.reset(torch.tensor([0]))
        self.assertEqual(self.runtime.target_index.tolist(), [0, 1])
        self.assertEqual(self.runtime.was_supported.tolist(), [[False, False], [False, True]])

    def test_flat_targets_reuse_runtime(self):
        geo = dict(
            first_riser_x=torch.zeros(2),
            tread_depth=torch.full((2,), 0.20),
            riser_height=torch.zeros(2),
            centre_y=torch.zeros(2),
            foot_separation=torch.full((2,), 0.16),
        )

        targets, feet, valid = self.runtime.targets(
            **geo, flat_ground=True
        )
        self.assertEqual(feet.tolist(), [[0, 1], [1, 0]])
        self.assertTrue(valid.all().item())
        self.assertTrue(torch.allclose(
            targets[:, 0, 0], torch.tensor([0.20, 0.20])
        ))
        self.assertTrue(torch.all(targets[:, :, 2] == 0))

        sole = torch.zeros(2, 2, 3)
        sole[torch.arange(2), feet[:, 0]] = targets[:, 0, :3]

        first = self.runtime.update(
            targets=targets, feet=feet, valid=valid,
            sole_xyz=sole,
            supported=torch.tensor([[True, False], [False, True]]),
            xy_tolerance=0.08, z_tolerance=0.025,
        )
        self.assertEqual(first.tolist(), [True, True])
        self.assertEqual(self.runtime.target_index.tolist(), [1, 1])

        next_targets, next_feet, next_valid = self.runtime.targets(
            **geo, flat_ground=True
        )
        self.assertTrue(next_valid.all().item())
        self.assertEqual(next_feet.tolist(), [[1, 0], [0, 1]])
        self.assertTrue(torch.allclose(
            next_targets[:, 0, 0], torch.tensor([0.40, 0.40])
        ))

        stair_targets, _, _ = self.runtime.targets(**self.geo)
        self.assertTrue(torch.all(stair_targets[:, 0, 2] > 0))

    def test_observation_shape(self):
        t, f, v, sole = self.inputs()
        out = self.runtime.observation(targets=t, feet=f, valid=v,
                                       root_xyz=torch.zeros(2, 3), root_yaw=torch.zeros(2), position_scale=1.)
        self.assertEqual(out.shape, (2, 14))

if __name__ == '__main__':
    unittest.main()


class SequenceContractTests(unittest.TestCase):
    def event(self, index, foot, **overrides):
        event = dict(
            target_index=index,
            foot=foot,
            placement_valid=True,
            support_valid=True,
            slip_valid=True,
            clearance_valid=True,
        )
        event.update(overrides)
        return event

    def evaluate(self, landings, **kwargs):
        return evaluate_footstep_sequence(
            required_feet=[0, 1, 0],
            landings=landings,
            **kwargs,
        )

    def test_complete_sequence_and_destination(self):
        result = self.evaluate(
            [self.event(0, 0), self.event(1, 1), self.event(2, 0)],
            destination_verified=True,
        )
        self.assertEqual(result["outcome"], "SUCCESS")
        self.assertEqual(result["verified_landings"], 3)

    def test_missing_destination_is_not_success(self):
        result = self.evaluate(
            [self.event(0, 0), self.event(1, 1), self.event(2, 0)]
        )
        self.assertEqual(result["outcome"], "NOT_VERIFIED")

    def test_wrong_foot_fails(self):
        result = self.evaluate([self.event(0, 1)])
        self.assertEqual(result["failure_reason"], "wrong_foot")

    def test_wrong_order_fails(self):
        result = self.evaluate([self.event(1, 0)])
        self.assertEqual(result["failure_reason"], "wrong_target_order")

    def test_invalid_support_fails(self):
        result = self.evaluate([
            self.event(0, 0, support_valid=False)
        ])
        self.assertEqual(result["failure_reason"], "support_valid")

    def test_missing_support_is_not_verified(self):
        result = self.evaluate([
            self.event(0, 0, support_valid=None)
        ])
        self.assertEqual(result["outcome"], "NOT_VERIFIED")

    def test_explicit_fall_fails(self):
        result = self.evaluate(
            [], terminal_failure="fall"
        )
        self.assertEqual(result["outcome"], "FAILURE")
        self.assertEqual(result["failure_reason"], "fall")

    def test_incomplete_sequence_is_not_success(self):
        result = self.evaluate(
            [self.event(0, 0)], destination_verified=True
        )
        self.assertEqual(result["outcome"], "NOT_VERIFIED")
