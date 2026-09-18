#!/usr/bin/env bash
# Launch a recorded experiment with host-derived Isaac Lab provenance.

set -euo pipefail

ISAACLAB_SOURCE="${ISAACLAB_SOURCE:-/home/ubuntu/workspace/src/IsaacLab}"
ISAACLAB_CONTAINER="${ISAACLAB_CONTAINER:-isaac-lab-fixed}"

ISAACLAB_SOURCE_COMMIT="$(git -C "${ISAACLAB_SOURCE}" rev-parse HEAD)"
ISAACLAB_SOURCE_TAG="$(
  git -C "${ISAACLAB_SOURCE}" describe --tags --exact-match HEAD
)"

docker exec \
  --env "ISAACLAB_SOURCE_COMMIT=${ISAACLAB_SOURCE_COMMIT}" \
  --env "ISAACLAB_SOURCE_TAG=${ISAACLAB_SOURCE_TAG}" \
  --env PYTHONPATH=/workspace/IEA_G1 \
  --env UNITREE_RL_LAB_ROOT=/workspace/unitree_rl_lab \
  --workdir /workspace/IEA_G1 \
  "${ISAACLAB_CONTAINER}" \
  /workspace/isaaclab/isaaclab.sh \
  -p \
  isaac_lab/g1_stairs/scripts/training/run_recorded_experiment.py \
  "$@"
