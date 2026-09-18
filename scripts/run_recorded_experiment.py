#!/usr/bin/env python3
"""Prepare, record, launch, and finalize a reproducible RL experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runtime_contract import validate_runtime_versions


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a YAML mapping to disk."""
    path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )


def prepare_run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve the experiment and generate its initial manifest."""
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_experiment.py"),
        str(args.config),
        "--output-root",
        str(args.output_root),
    ]

    if args.run_id:
        command.extend(("--run-id", args.run_id))

    if args.allow_dirty:
        command.append("--allow-dirty")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(
            details
            or f"experiment preparation failed with exit code {result.returncode}"
        )

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

    required = ("RUN_DIR", "RESOLVED_CONFIG", "RUN_MANIFEST")
    missing = [key for key in required if key not in values]

    if missing:
        raise RuntimeError(
            "prepare_experiment.py did not return: "
            + ", ".join(missing)
        )

    return (
        Path(values["RUN_DIR"]),
        Path(values["RESOLVED_CONFIG"]),
        Path(values["RUN_MANIFEST"]),
    )


def build_training_command(
    resolved: dict[str, Any],
    extra_args: list[str],
) -> list[str]:
    """Build the pinned upstream or custom stair-training command."""
    unitree_root_value = os.environ.get("UNITREE_RL_LAB_ROOT")
    if not unitree_root_value:
        raise RuntimeError(
            "UNITREE_RL_LAB_ROOT must point to the pinned "
            "Unitree RL Lab checkout"
        )

    unitree_root = Path(unitree_root_value).expanduser().resolve()
    task = resolved["task"]["id"]
    execution = resolved["execution"]

    custom_task = resolved["task"].get(
        "custom_stair_task_active",
        task == "Unitree-IEA-G1-29dof-Stairs-v0",
    )

    if custom_task:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_isaac_stairs.py"),
            "train",
        ]
    else:
        upstream_script = (
            unitree_root
            / "scripts"
            / "rsl_rl"
            / "train.py"
        )

        if not upstream_script.is_file():
            raise RuntimeError(
                f"missing upstream training script: {upstream_script}"
            )

        command = [
            sys.executable,
            str(upstream_script),
        ]

    command.extend(
        [
            "--headless",
            "--task",
            task,
            "--num_envs",
            str(execution["environments"]),
            "--max_iterations",
            str(execution["maximum_iterations"]),
            "--seed",
            str(execution["training_seed"]),
        ]
    )

    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command.extend(extra_args)
    return command


def update_manifest(
    manifest_path: Path,
    updates: dict[str, Any],
) -> None:
    """Update fields under the manifest's run section."""
    manifest = load_yaml(manifest_path)

    run_section = manifest.setdefault("run", {})
    if not isinstance(run_section, dict):
        raise ValueError(
            f"{manifest_path} contains an invalid run section"
        )

    run_section.update(updates)
    write_yaml(manifest_path, manifest)


def stream_training_process(
    command: list[str],
    console_path: Path,
    run_dir: Path,
) -> int:
    """Run training while streaming and preserving combined output."""
    process: subprocess.Popen[str] | None = None

    try:
        with console_path.open("w", encoding="utf-8") as log:
            environment = os.environ.copy()
            environment["IEA_G1_METRICS_DIR"] = str(run_dir)
            environment["IEA_G1_RUN_ID"] = run_dir.name
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )

            if process.stdout is None:
                raise RuntimeError(
                    "training process did not provide an output stream"
                )

            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()

            return process.wait()

    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        return 130


def parse_arguments() -> argparse.Namespace:
    """Parse launcher options while preserving upstream training options."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and launch a reproducible Unitree RL experiment"
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Experiment YAML configuration",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Directory under which the run directory is created",
    )

    parser.add_argument(
        "--run-id",
        help="Explicit unique run identifier",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the resolved config and manifest without training",
    )

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Permit a dirty worktree for local infrastructure testing; "
            "never use for real experiments"
        ),
    )

    args, extra_training_arguments = parser.parse_known_args()
    args.extra_training_arguments = extra_training_arguments
    return args


def main() -> int:
    args = parse_arguments()

    if not args.config.is_absolute():
        args.config = (REPO_ROOT / args.config).resolve()

    args.output_root = args.output_root.expanduser().resolve()

    run_dir, resolved_path, manifest_path = prepare_run(args)
    resolved = load_yaml(resolved_path)
    manifest = load_yaml(manifest_path)

    print(f"Prepared run directory: {run_dir}")
    print(f"Resolved configuration: {resolved_path}")
    print(f"Run manifest: {manifest_path}")

    if args.prepare_only:
        update_manifest(
            manifest_path,
            {
                "status": "prepared_only",
                "prepared_only_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        )

        print(
            "Preparation-only test complete; training was not started"
        )
        return 0

    try:
        packages = manifest.get("packages", {})
        provenance = manifest.get("source_provenance", {})
        software = resolved.get("software", {})
        validate_runtime_versions(
            isaac_lab_source_commit=provenance.get("isaac_lab_commit"),
            isaac_lab_source_tag=provenance.get("isaac_lab_tag"),
            rsl_rl=packages.get("rsl_rl"),
        )
        validate_runtime_versions(
            isaac_lab_source_commit=software.get("isaac_lab_commit"),
            isaac_lab_source_tag=software.get("isaac_lab_tag"),
            rsl_rl=software.get("rsl_rl"),
        )
        command = build_training_command(
            resolved,
            args.extra_training_arguments,
        )
    except (KeyError, TypeError, RuntimeError) as error:
        update_manifest(
            manifest_path,
            {
                "status": "failed_before_launch",
                "finished_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "error": str(error),
            },
        )
        raise SystemExit(str(error)) from error

    console_path = run_dir / "console.log"

    update_manifest(
        manifest_path,
        {
            "status": "running",
            "started_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "command": command,
            "console_log": console_path.name,
        },
    )

    print("Launching:", " ".join(command))
    print(f"Console output: {console_path}")

    exit_code = stream_training_process(
        command,
        console_path,
        run_dir,
    )

    update_manifest(
        manifest_path,
        {
            "status": (
                "completed"
                if exit_code == 0
                else "interrupted"
                if exit_code == 130
                else "failed"
            ),
            "finished_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "exit_code": exit_code,
        },
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
