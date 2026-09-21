"""Resolve and install Python paths from a Unitree RL Lab checkout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class UnitreeRlLabPaths:
    """Import roots provided by the pinned Unitree RL Lab repository."""

    repository_root: Path
    python_source_root: Path
    rsl_rl_scripts: Path


def resolve_unitree_rl_lab_paths(
    repository_root: str | Path,
) -> UnitreeRlLabPaths:
    """Validate a Unitree checkout and return its Python import roots."""
    root = Path(repository_root).expanduser().resolve()
    python_source_root = root / "source" / "unitree_rl_lab"
    package_dir = python_source_root / "unitree_rl_lab"
    rsl_rl_scripts = root / "scripts" / "rsl_rl"

    if not package_dir.is_dir():
        raise RuntimeError(
            "Unitree RL Lab Python package missing: "
            f"{package_dir}"
        )
    if not rsl_rl_scripts.is_dir():
        raise RuntimeError(
            f"Unitree RSL-RL scripts missing: {rsl_rl_scripts}"
        )

    return UnitreeRlLabPaths(
        repository_root=root,
        python_source_root=python_source_root,
        rsl_rl_scripts=rsl_rl_scripts,
    )


def bootstrap_unitree_rl_lab(
    repository_root: str | Path,
) -> UnitreeRlLabPaths:
    """Make both the Unitree package and its launcher helpers importable."""
    paths = resolve_unitree_rl_lab_paths(repository_root)
    for path in (paths.python_source_root, paths.rsl_rl_scripts):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return paths
