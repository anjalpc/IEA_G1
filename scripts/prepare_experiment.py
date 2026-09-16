#!/usr/bin/env python3
"""Resolve an experiment config and create its reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(*args: str) -> str | None:
    """Run a read-only command and return stripped output when successful."""
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge mappings; lists and scalar values are replaced."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if data.get("schema_version") != 1:
        raise ValueError(f"{path} must declare schema_version: 1")
    return data


def resolve_config(
    path: Path,
    active_paths: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], list[str]]:
    path = path.resolve()
    if path in active_paths:
        chain = " -> ".join(str(item) for item in (*active_paths, path))
        raise ValueError(f"configuration inheritance cycle: {chain}")

    data = load_yaml(path)
    inheritance = data.get("inheritance", {})
    base_value = inheritance.get("base_config") if isinstance(inheritance, dict) else None

    if not base_value:
        resolved = data
        sources = [str(path.relative_to(REPO_ROOT))]
    else:
        base_path = Path(base_value)
        if not base_path.is_absolute():
            base_path = REPO_ROOT / base_path
        base_config, sources = resolve_config(base_path, (*active_paths, path))
        resolved = deep_merge(base_config, data)
        sources.append(str(path.relative_to(REPO_ROOT)))

    resolved.pop("inheritance", None)
    resolved["_resolution"] = {
        "sources_in_order": sources,
        "leaf_config": str(path.relative_to(REPO_ROOT)),
    }
    return resolved, sources


def gpu_inventory() -> list[dict[str, str]]:
    query = run_command(
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    if not query:
        return []

    inventory = []
    for line in query.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        inventory.append(
            {
                "index": values[0],
                "name": values[1],
                "driver_version": values[2],
                "memory_mib": values[3],
            }
        )
    return inventory


def git_information() -> dict[str, Any]:
    status = run_command("git", "status", "--porcelain")
    return {
        "repository_root": str(REPO_ROOT),
        "branch": run_command("git", "branch", "--show-current"),
        "commit": run_command("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines() if status else [],
        "remote": run_command("git", "remote", "get-url", "personal"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise SystemExit(f"missing experiment config: {config_path}")

    resolved, sources = resolve_config(config_path)
    git = git_information()

    require_clean = resolved.get("code", {}).get(
        "require_clean_worktree",
        True,
    )
    if require_clean and git["dirty"] and not args.allow_dirty:
        raise SystemExit(
            "refusing to prepare a run from a dirty worktree; "
            "commit the experiment infrastructure first"
        )

    gates = resolved.get("launch_gates", {})
    if gates and not gates.get("allow_launch", False) and not args.allow_blocked:
        raise SystemExit(
            "experiment launch is blocked by launch_gates; "
            "complete the required evaluator and metrics work first"
        )

    experiment = resolved.get("experiment", {})
    experiment_id = experiment.get("id")
    if not experiment_id:
        raise SystemExit("resolved config is missing experiment.id")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{experiment_id}_{timestamp}"

    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "run": {
            "id": run_id,
            "experiment_id": experiment_id,
            "status": "prepared",
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_directory": str(run_dir),
        },
        "configuration": {
            "leaf_file": str(config_path.relative_to(REPO_ROOT)),
            "sources_in_order": sources,
            "source_sha256": file_sha256(config_path),
            "resolved_file": resolved_path.name,
            "resolved_sha256": file_sha256(resolved_path),
        },
        "git": git,
        "host": {
            "hostname": socket.gethostname(),
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
            "gpus": gpu_inventory(),
        },
        "packages": {
            "isaac_sim": package_version("isaacsim"),
            "isaac_lab": package_version("isaaclab"),
            "rsl_rl": package_version("rsl-rl-lib", "rsl_rl"),
            "torch": package_version("torch"),
            "pyyaml": package_version("PyYAML"),
        },
    }

    manifest_path = run_dir / "run_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    print(f"RUN_ID={run_id}")
    print(f"RUN_DIR={run_dir}")
    print(f"RESOLVED_CONFIG={resolved_path}")
    print(f"RUN_MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
