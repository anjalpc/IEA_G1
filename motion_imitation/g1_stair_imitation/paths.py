"""Repository paths for the isolated imitation-learning stack."""

from __future__ import annotations

from pathlib import Path


MOTION_IMITATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MOTION_IMITATION_ROOT.parent

REFERENCE_FILE = (
    MOTION_IMITATION_ROOT
    / "outputs/g1_stair_reference_v1/g1_stair_reference_20cm_v001.json"
)

EXPERIMENTS_ROOT = MOTION_IMITATION_ROOT / "experiments"
CONFIGS_ROOT = EXPERIMENTS_ROOT / "configs"
RUNS_ROOT = EXPERIMENTS_ROOT / "runs"
LOGS_ROOT = EXPERIMENTS_ROOT / "logs"
CHECKPOINTS_ROOT = EXPERIMENTS_ROOT / "checkpoints"
OUTPUTS_ROOT = EXPERIMENTS_ROOT / "outputs"
ARCHIVE_ROOT = EXPERIMENTS_ROOT / "archive"
