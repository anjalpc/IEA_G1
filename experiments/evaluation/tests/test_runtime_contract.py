import unittest

from experiments.runtime_contract import (
    RuntimeVersionMismatch,
    normalize_release_tag,
    validate_runtime_versions,
)


class RuntimeContractTest(unittest.TestCase):
    def test_exact_frozen_versions_are_accepted(self):
        validate_runtime_versions(isaac_lab="2.3.0", rsl_rl="3.0.1")

    def test_mismatched_isaac_lab_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeVersionMismatch,
            "Isaac Lab expected 2.3.0, found 2.2.1",
        ):
            validate_runtime_versions(
                isaac_lab="2.2.1", rsl_rl="3.0.1"
            )

    def test_mismatched_rsl_rl_is_rejected(self):
        with self.assertRaisesRegex(
            RuntimeVersionMismatch,
            "RSL-RL expected 3.0.1, found 2.3.1",
        ):
            validate_runtime_versions(
                isaac_lab="2.3.0", rsl_rl="2.3.1"
            )

    def test_release_tag_normalization_is_explicit(self):
        self.assertEqual(normalize_release_tag("v2.3.0\n"), "2.3.0")
        self.assertEqual(normalize_release_tag("2.3.0"), "2.3.0")
        self.assertIsNone(normalize_release_tag(None))


if __name__ == "__main__":
    unittest.main()
