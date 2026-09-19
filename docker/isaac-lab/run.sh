#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="isaac-lab-fixed"
IMAGE_NAME="isaac-lab-base-fixed:2.3.0"
VOLUME_SOURCE_CONTAINER="isaac-lab-base"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${SCRIPT_DIR}/.env.local"
USE_PROXY=false

for argument in "$@"; do
    case "${argument}" in
        --proxy)
            USE_PROXY=true
            ;;
        *)
            echo "Usage: $0 [--proxy]" >&2
            exit 2
            ;;
    esac
done

if [[ "${USE_PROXY}" == true ]]; then
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "ERROR: --proxy requested but ${ENV_FILE} does not exist." >&2
        exit 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
else
    unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy || true
fi
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${IEA_WORKSPACE:-$(dirname "${REPO_ROOT}")}"

UNITREE_RL_HOST="${UNITREE_RL_HOST:-${WORKSPACE}/src/unitree_rl_lab}"
UNITREE_ROS_HOST="${UNITREE_ROS_HOST:-${WORKSPACE}/src/unitree_ros}"
PROJECT_HOST="${PROJECT_HOST:-${REPO_ROOT}}"

echo "=== VERIFY REQUIRED PATHS ==="

test -d "${UNITREE_RL_HOST}"
test -d "${UNITREE_ROS_HOST}"
test -d "${PROJECT_HOST}"

echo "VALID: required host paths exist"

echo
echo "=== VERIFY BASE CONTAINER AND FIXED IMAGE ==="

docker inspect \
  "${VOLUME_SOURCE_CONTAINER}" \
  --format 'base={{.Name}} status={{.State.Status}}'

docker image inspect \
  "${IMAGE_NAME}" \
  --format 'fixed_image={{.Id}}'

echo
echo "=== STOP OLD FIXED CONTAINER ==="

docker stop \
  --timeout 30 \
  "${CONTAINER_NAME}" \
  2>/dev/null || true

echo
echo "=== REMOVE OLD FIXED CONTAINER ==="

docker rm \
  "${CONTAINER_NAME}" \
  2>/dev/null || true

echo
echo "=== START FIXED INTEGRATION CONTAINER ==="

docker run \
  --detach \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --network=host \
  --ipc=host \
  --entrypoint bash \
  --volumes-from "${VOLUME_SOURCE_CONTAINER}" \
  --volume "${UNITREE_RL_HOST}:/workspace/unitree_rl_lab:ro" \
  --volume "${UNITREE_ROS_HOST}:/workspace/unitree_ros:ro" \
  --volume "${PROJECT_HOST}:/workspace/IEA_G1:rw" \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env UNITREE_RL_LAB_ROOT=/workspace/unitree_rl_lab \
  --env PYTHONPATH=/workspace/IEA_G1:/workspace/unitree_rl_lab/source/unitree_rl_lab \
  --env HTTP_PROXY="${HTTP_PROXY:-}" \
  --env HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --env NO_PROXY="${NO_PROXY:-}" \
  --env http_proxy="${http_proxy:-${HTTP_PROXY:-}}" \
  --env https_proxy="${https_proxy:-${HTTPS_PROXY:-}}" \
  --env no_proxy="${no_proxy:-${NO_PROXY:-}}" \
  "${IMAGE_NAME}" \
  -lc 'sleep infinity'

docker exec \
  "${CONTAINER_NAME}" \
  git config --global --add safe.directory /workspace/IEA_G1

echo
echo "=== CONTAINER STATE ==="

docker inspect \
  "${CONTAINER_NAME}" \
  --format 'status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} image={{.Config.Image}} error={{.State.Error}}'

echo
echo "=== REQUIRED MOUNTS ==="

docker inspect \
  "${CONTAINER_NAME}" \
  --format '{{range .Mounts}}{{println .Source "->" .Destination "rw=" .RW}}{{end}}' |
grep -E '/workspace/(isaaclab|unitree_rl_lab|unitree_ros|IEA_G1)'

echo
echo "=== PROXY ==="

docker exec \
  "${CONTAINER_NAME}" \
  env |
grep -iE '^(http|https|no)_proxy=' |
sort

echo
echo "=== GPU ==="

docker exec \
  "${CONTAINER_NAME}" \
  nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader
