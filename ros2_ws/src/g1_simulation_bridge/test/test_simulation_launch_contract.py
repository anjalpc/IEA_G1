"""Contract tests for the application-neutral simulation launch."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NEUTRAL_LAUNCH = PACKAGE_ROOT / "launch/simulation.launch.py"
HANDSHAKE_LAUNCH = PACKAGE_ROOT / "launch/simulation_handshake.launch.py"


def _parsed_launch():
    source = NEUTRAL_LAUNCH.read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_neutral_simulation_launch_exists():
    assert NEUTRAL_LAUNCH.is_file()


def test_neutral_launch_starts_only_the_simulation_bridge():
    source, tree = _parsed_launch()
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }
    node_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Node"
    ]

    assert len(node_calls) == 1
    assert "g1_simulation_bridge" in constants
    assert "simulation_bridge" in constants
    assert "simulation.yaml" in constants
    assert "G1_SIMULATION_ARGUMENTS" in constants
    assert "g1_handshake" not in source
    assert "handshake_node" not in source
    assert "event_handshake" not in source


def test_existing_handshake_launch_remains_available():
    assert HANDSHAKE_LAUNCH.is_file()
