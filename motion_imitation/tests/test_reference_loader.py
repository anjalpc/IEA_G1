from pathlib import Path
import sys
import unittest

import numpy as np


MOTION_IMITATION_ROOT = Path(__file__).resolve().parents[1]
if str(MOTION_IMITATION_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTION_IMITATION_ROOT))

from g1_stair_imitation.paths import REFERENCE_FILE
from g1_stair_imitation.reference import load_g1_stair_imitation_reference


class ReferenceLoaderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_g1_stair_imitation_reference(REFERENCE_FILE)

    def test_reference_loader_contract(self):
        ref = self.ref
        self.assertEqual(ref.frame_count, 127)
        self.assertEqual(ref.q_ref.shape, (127, 29))
        self.assertEqual(ref.dq_ref.shape, (127, 29))
        self.assertEqual(ref.left_sole_points.shape, (127, 4, 3))
        self.assertEqual(ref.right_sole_points.shape, (127, 4, 3))
        self.assertFalse(bool(ref.taskspace_valid[0]))
        self.assertTrue(bool(np.all(ref.taskspace_valid[1:])))
        self.assertEqual(int(np.sum(ref.pose_mask > 0.0)), 6)

    def test_critical_contact_sequence(self):
        ref = self.ref

        expected = {
            152: (True, 0, False, -1),
            179: (False, -1, True, 1),
            207: (True, 2, False, -1),
            239: (False, -1, True, 3),
            269: (True, 3, True, 3),
            277: (True, 3, True, 3),
        }

        for source_frame, wanted in expected.items():
            with self.subTest(source_frame=source_frame):
                index = ref.reference_index_for_source_frame(source_frame)
                got = (
                    bool(ref.left_contact[index]),
                    int(ref.left_step[index]),
                    bool(ref.right_contact[index]),
                    int(ref.right_step[index]),
                )
                self.assertEqual(got, wanted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
