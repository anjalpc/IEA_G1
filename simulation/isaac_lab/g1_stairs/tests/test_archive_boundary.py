import ast
import re
import unittest
from pathlib import Path


ARCHIVED_MODULE_BASENAMES = {
    "approach_teacher",
    "approach_teacher_ppo",
    "halfway_reset",
    "physical_step_config",
    "physical_step_runtime",
    "structured_step044_config",
    "structured_step044_runtime",
    "rewards",
    "unified_rewards",
}


class ArchiveBoundaryTest(unittest.TestCase):
    def test_active_config_directory_excludes_numbered_legacy_profiles(self):
        root = Path(__file__).resolve().parents[4]
        config_dir = root / "experiments/g1_stairs/configs"
        names = sorted(path.name for path in config_dir.glob("*.yaml"))

        # The active config directory is intentionally extensible.  Future
        # specialist/curriculum profiles may be added without changing this
        # test; only historical numbered generations belong in the archive.
        self.assertIn("nominal_stairs_base.yaml", names)
        self.assertIn("specialist_base.yaml", names)
        self.assertIn("specialist_smoke.yaml", names)
        self.assertFalse(
            [name for name in names if re.match(r"^\d{3}[a-z]?_", name)],
            msg=f"numbered pre-specialist config leaked into active directory: {names}",
        )

    def test_active_python_does_not_import_archived_runtime_modules(self):
        package = Path(__file__).resolve().parents[1]
        offenders = []

        for path in package.rglob("*.py"):
            if "tests" in path.parts:
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imported.append(module)
                    imported.extend(
                        f"{module}.{alias.name}" if module else alias.name
                        for alias in node.names
                    )

            for module in imported:
                basename = module.rsplit(".", 1)[-1]
                if basename in ARCHIVED_MODULE_BASENAMES:
                    offenders.append(
                        f"{path.relative_to(package)} imports {module}"
                    )
                if "controller_port" in module.split("."):
                    offenders.append(
                        f"{path.relative_to(package)} imports {module}"
                    )

        self.assertEqual(offenders, [])

    def test_archive_manifest_exists(self):
        root = Path(__file__).resolve().parents[4]
        archive = root / "experiments/g1_stairs/archive/pre_specialist_20260925"
        self.assertTrue((archive / "README.md").is_file())
        self.assertTrue((archive / "manifest.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
