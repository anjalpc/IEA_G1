"""Protect ownership boundaries of the reusable MuJoCo runtime."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MUJOCO_ROOT = REPOSITORY_ROOT / "simulation" / "mujoco"


def test_model_runtime_has_a_dedicated_reusable_module():
    sys.path.insert(0, str(MUJOCO_ROOT))
    try:
        from runtime.model_runtime import ElasticHarness, initialize
    finally:
        sys.path.remove(str(MUJOCO_ROOT))

    assert ElasticHarness.__module__ == "runtime.model_runtime"
    assert initialize.__module__ == "runtime.model_runtime"


def test_model_runner_preserves_compatibility_exports():
    source = (MUJOCO_ROOT / "run_g1_model.py").read_text(
        encoding="utf-8"
    )

    assert "from runtime.model_runtime import" in source
    assert "ElasticHarness" in source
    assert "initialize" in source


def test_handshake_runner_uses_reusable_runtime_directly():
    source = (MUJOCO_ROOT / "run_handshake_demo.py").read_text(
        encoding="utf-8"
    )

    assert "from runtime.model_runtime import" in source
    assert "from run_g1_model import ElasticHarness, initialize" not in source
