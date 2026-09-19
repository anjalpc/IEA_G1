import unittest

from simulation.isaac_lab.g1_stairs.runtime_contract import (
    EXPECTED_ISAAC_LAB_COMMIT,
    EXPECTED_ISAAC_LAB_TAG,
    RuntimeVersionMismatch,
    normalize_release_tag,
    validate_runtime_versions,
)


class RuntimeContractTest(unittest.TestCase):
    def test_exact_frozen_versions_are_accepted(self):
        validate_runtime_versions(
            isaac_lab_source_commit=EXPECTED_ISAAC_LAB_COMMIT,
            isaac_lab_source_tag=EXPECTED_ISAAC_LAB_TAG,
            rsl_rl="3.0.1",
        )

    def test_missing_source_provenance_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeVersionMismatch,
            "found unavailable",
        ):
            validate_runtime_versions(
                isaac_lab_source_commit=None,
                isaac_lab_source_tag=None,
                rsl_rl="3.0.1",
            )

    def test_wrong_isaac_lab_commit_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeVersionMismatch,
            "Isaac Lab commit expected",
        ):
            validate_runtime_versions(
                isaac_lab_source_commit="0" * 40,
                isaac_lab_source_tag=EXPECTED_ISAAC_LAB_TAG,
                rsl_rl="3.0.1",
            )

    def test_wrong_isaac_lab_tag_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeVersionMismatch,
            "Isaac Lab tag expected v2.3.0, found v2.2.1",
        ):
            validate_runtime_versions(
                isaac_lab_source_commit=EXPECTED_ISAAC_LAB_COMMIT,
                isaac_lab_source_tag="v2.2.1",
                rsl_rl="3.0.1",
            )

    def test_mismatched_rsl_rl_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeVersionMismatch,
            "RSL-RL expected 3.0.1, found 2.3.1",
        ):
            validate_runtime_versions(
                isaac_lab_source_commit=EXPECTED_ISAAC_LAB_COMMIT,
                isaac_lab_source_tag=EXPECTED_ISAAC_LAB_TAG,
                rsl_rl="2.3.1",
            )

    def test_release_tag_normalization_is_explicit(self):
        self.assertEqual(normalize_release_tag("v2.3.0\n"), "2.3.0")
        self.assertEqual(normalize_release_tag("2.3.0"), "2.3.0")
        self.assertIsNone(normalize_release_tag(None))


if __name__ == "__main__":
    unittest.main()
