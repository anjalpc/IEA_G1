"""Exact runtime-version contract for reproducible Isaac Lab experiments."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


EXPECTED_ISAAC_LAB_VERSION = "2.3.0"
EXPECTED_RSL_RL_VERSION = "3.0.1"
DEFAULT_ISAAC_LAB_ROOT = Path("/workspace/isaaclab")


class RuntimeVersionMismatch(RuntimeError):
    """Raised when an installed training dependency violates the freeze."""


def normalize_release_tag(value: str | None) -> str | None:
    """Normalize an exact Git release tag such as ``v2.3.0``."""
    if not value:
        return None
    value = value.strip()
    return value[1:] if value.startswith("v") else value


def isaac_lab_checkout_version(root: Path | None = None) -> str | None:
    """Return the exact Isaac Lab Git release checked out for this runtime."""
    if root is None:
        root = Path(
            os.environ.get("ISAAC_LAB_ROOT", DEFAULT_ISAAC_LAB_ROOT)
        )
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "describe",
                "--tags",
                "--exact-match",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return normalize_release_tag(result.stdout)


def validate_runtime_versions(
    *,
    isaac_lab: str | None,
    rsl_rl: str | None,
    expected_isaac_lab: str = EXPECTED_ISAAC_LAB_VERSION,
    expected_rsl_rl: str = EXPECTED_RSL_RL_VERSION,
) -> None:
    """Require exact frozen versions before any canonical training launch."""
    mismatches = []
    if isaac_lab != expected_isaac_lab:
        mismatches.append(
            f"Isaac Lab expected {expected_isaac_lab}, found "
            f"{isaac_lab or 'unavailable'}"
        )
    if rsl_rl != expected_rsl_rl:
        mismatches.append(
            f"RSL-RL expected {expected_rsl_rl}, found "
            f"{rsl_rl or 'unavailable'}"
        )
    if mismatches:
        raise RuntimeVersionMismatch(
            "runtime version preflight failed: " + "; ".join(mismatches)
        )
