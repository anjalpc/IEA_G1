import unittest

import torch

from simulation.isaac_lab.g1_stairs.reward_math import (
    first_stair_acquisition_event,
)


class FirstStairAcquisitionRewardTests(unittest.TestCase):

    def test_only_first_stair_acquisition_is_selected(self):
        result = first_stair_acquisition_event(
            torch.tensor([True, True, False]),
            torch.tensor([0, 1, 0]),
        )

        torch.testing.assert_close(
            result,
            torch.tensor([True, False, False]),
        )

    def test_event_is_one_shot_when_runtime_event_clears(self):
        first = first_stair_acquisition_event(
            torch.tensor([True]),
            torch.tensor([0]),
        )

        second = first_stair_acquisition_event(
            torch.tensor([False]),
            torch.tensor([-1]),
        )

        torch.testing.assert_close(first, torch.tensor([True]))
        torch.testing.assert_close(second, torch.tensor([False]))

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            first_stair_acquisition_event(
                torch.tensor([True, False]),
                torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
