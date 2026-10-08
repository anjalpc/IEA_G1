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

    assert "from runtime.integrated_backend import" in source
    assert "from run_g1_model import" not in source

def test_integrated_backend_has_a_dedicated_runtime_module():
    sys.path.insert(0, str(MUJOCO_ROOT))
    try:
        from runtime.integrated_backend import (
            IntegratedG1Backend,
            SimulationControls,
        )
    finally:
        sys.path.remove(str(MUJOCO_ROOT))

    assert IntegratedG1Backend.__module__ == "runtime.integrated_backend"
    assert SimulationControls.__module__ == "runtime.integrated_backend"


def test_ros_adapter_loads_the_reusable_backend_directly():
    adapter = (
        REPOSITORY_ROOT
        / "ros2_ws"
        / "src"
        / "g1_simulation_bridge"
        / "g1_simulation_bridge"
        / "adapter_node.py"
    ).read_text(encoding="utf-8")

    assert "from runtime.integrated_backend import" in adapter
    assert "from run_handshake_demo import" not in adapter


def test_handshake_runner_consumes_the_reusable_backend():
    source = (MUJOCO_ROOT / "run_handshake_demo.py").read_text(
        encoding="utf-8"
    )

    assert "from runtime.integrated_backend import" in source
    assert "class SimulationControls:" not in source
    assert "class IntegratedG1Backend:" not in source
