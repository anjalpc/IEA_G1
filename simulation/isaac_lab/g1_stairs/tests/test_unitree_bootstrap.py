"""Regression tests for Unitree RL Lab import-path bootstrapping."""

from pathlib import Path
import sys
import tempfile
import unittest

from simulation.isaac_lab.g1_stairs.unitree_bootstrap import (
    bootstrap_unitree_rl_lab,
    resolve_unitree_rl_lab_paths,
)


class UnitreeRlLabBootstrapTest(unittest.TestCase):
    def make_checkout(self, root: Path) -> tuple[Path, Path]:
        source_root = root / "source" / "unitree_rl_lab"
        (source_root / "unitree_rl_lab").mkdir(parents=True)
        scripts_root = root / "scripts" / "rsl_rl"
        scripts_root.mkdir(parents=True)
        return source_root, scripts_root

    def test_resolves_actual_unitree_repository_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, scripts_root = self.make_checkout(root)

            paths = resolve_unitree_rl_lab_paths(root)

            self.assertEqual(paths.repository_root, root.resolve())
            self.assertEqual(paths.python_source_root, source_root)
            self.assertEqual(paths.rsl_rl_scripts, scripts_root)

    def test_bootstrap_adds_package_parent_and_launcher_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, scripts_root = self.make_checkout(root)
            original_path = list(sys.path)
            try:
                bootstrap_unitree_rl_lab(root)
                self.assertIn(str(source_root), sys.path)
                self.assertIn(str(scripts_root), sys.path)
            finally:
                sys.path[:] = original_path

    def test_rejects_repository_root_without_python_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts" / "rsl_rl").mkdir(parents=True)

            with self.assertRaisesRegex(
                RuntimeError,
                "Unitree RL Lab Python package missing",
            ):
                resolve_unitree_rl_lab_paths(root)


if __name__ == "__main__":
    unittest.main()
