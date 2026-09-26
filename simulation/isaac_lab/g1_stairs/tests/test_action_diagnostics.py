import unittest

from simulation.isaac_lab.g1_stairs.action_diagnostics import (
    ACTION_GROUP_INDICES,
    action_group_mean_abs,
)
from simulation.isaac_lab.g1_stairs.contracts import BODY_JOINTS


class ActionDiagnosticsTest(unittest.TestCase):
    def test_groups_partition_canonical_action_order(self):
        flattened = [
            index
            for indices in ACTION_GROUP_INDICES.values()
            for index in indices
        ]
        self.assertEqual(sorted(flattened), list(range(len(BODY_JOINTS))))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_group_means_use_canonical_indices(self):
        import torch

        actions = torch.zeros((1, len(BODY_JOINTS)))
        actions[:, ACTION_GROUP_INDICES["legs"]] = 1.0
        actions[:, ACTION_GROUP_INDICES["arms"]] = 2.0
        actions[:, ACTION_GROUP_INDICES["waist"]] = 3.0
        values = action_group_mean_abs(actions)
        self.assertAlmostEqual(float(values["legs"]), 1.0)
        self.assertAlmostEqual(float(values["arms"]), 2.0)
        self.assertAlmostEqual(float(values["waist"]), 3.0)

    def test_nonfinite_actions_fail_closed(self):
        import torch

        actions = torch.zeros((1, len(BODY_JOINTS)))
        actions[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            action_group_mean_abs(actions)


if __name__ == "__main__":
    unittest.main()
