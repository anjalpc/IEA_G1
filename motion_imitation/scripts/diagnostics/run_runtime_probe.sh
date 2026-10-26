#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${ISAAC_CONTAINER:-isaac-lab-fixed}"
PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"
CONTAINER_REPO="/tmp/iea_g1_imitation_runtime_probe"
REFERENCE_SOURCE="/workspace/IEA_G1/motion_imitation/outputs"
SCRIPT="motion_imitation/scripts/diagnostics/runtime_probe.py"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

if [ "$(git -C "$HOST_REPO" branch --show-current)" != "g1-stair-imitation" ]; then
    echo "Refusing imitation runtime probe from the wrong branch/worktree:" >&2
    echo "  $HOST_REPO" >&2
    echo "  branch=$(git -C "$HOST_REPO" branch --show-current)" >&2
    exit 1
fi

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
    echo "Isaac container is not running: $CONTAINER" >&2
    exit 1
fi

for required in \
    motion_imitation/scripts/diagnostics/runtime_probe.py \
    motion_imitation/g1_stair_imitation/isaac/runtime_diagnostics.py \
    motion_imitation/g1_stair_imitation/isaac/runtime_contract.py \
    motion_imitation/g1_stair_imitation/reference.py
do
    if ! git -C "$HOST_REPO" ls-files --error-unmatch "$required" >/dev/null 2>&1; then
        echo "Required imitation source is not staged/tracked: $required" >&2
        echo "Stage the curated imitation source before running runtime-probe." >&2
        exit 1
    fi
done

if ! docker exec "$CONTAINER" test -d "$REFERENCE_SOURCE"; then
    echo "Canonical imitation reference directory is unavailable in the container:" >&2
    echo "  $REFERENCE_SOURCE" >&2
    exit 1
fi

echo "[imitation] host worktree  : $HOST_REPO"
echo "[imitation] container copy : $CONTAINER_REPO"

docker exec "$CONTAINER" rm -rf "$CONTAINER_REPO"
docker exec "$CONTAINER" mkdir -p "$CONTAINER_REPO"
(
    cd "$HOST_REPO"
    git ls-files -z | tar --null -T - -cf -
) | docker exec -i "$CONTAINER" tar -xf - -C "$CONTAINER_REPO"

docker exec "$CONTAINER" \
    ln -s "$REFERENCE_SOURCE" "$CONTAINER_REPO/motion_imitation/outputs"

set +e
docker exec -i \
    -e PYTHONUNBUFFERED=1 \
    -e UNITREE_RL_LAB_ROOT=/workspace/unitree_rl_lab \
    -e PYTHONPATH="$CONTAINER_REPO:$CONTAINER_REPO/simulation:/workspace/unitree_rl_lab/source/unitree_rl_lab" \
    --workdir "$CONTAINER_REPO" \
    "$CONTAINER" \
    "$PYTHON" \
    -u \
    "$SCRIPT"
status=$?
set -e

docker exec "$CONTAINER" rm -rf "$CONTAINER_REPO" || true
exit "$status"
