#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(dirname "${REPO_ROOT}")"
SRC_ROOT="${WORKSPACE}/src"

ISAACLAB_URL="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT="3c6e67bb5c7ada942a6d1884ab69338f57596f77"

UNITREE_RL_URL="https://github.com/unitreerobotics/unitree_rl_lab.git"
UNITREE_RL_COMMIT="4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"

UNITREE_ROS_URL="https://github.com/unitreerobotics/unitree_ros.git"
UNITREE_ROS_COMMIT="7d6075f7f58588b189b940130e3edab3c839b2df"

prepare_repo() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local path="${SRC_ROOT}/${name}"

    echo
    echo "=== ${name} ==="

    if [[ ! -e "${path}" ]]; then
        echo "Cloning ${url}"
        git clone "${url}" "${path}"
    fi

    if [[ ! -d "${path}/.git" ]]; then
        echo "ERROR: ${path} exists but is not a Git repository."
        exit 1
    fi

    if [[ -n "$(git -C "${path}" status --porcelain)" ]]; then
        echo "ERROR: ${path} has local changes."
        echo "Refusing to overwrite a dirty dependency checkout."
        git -C "${path}" status --short
        exit 1
    fi

    if ! git -C "${path}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
        echo "Fetching required revision..."
        git -C "${path}" fetch --tags origin
    fi

    git -C "${path}" checkout --detach "${commit}"

    actual="$(git -C "${path}" rev-parse HEAD)"

    if [[ "${actual}" != "${commit}" ]]; then
        echo "ERROR: expected ${commit}, got ${actual}"
        exit 1
    fi

    echo "PASS: ${name}"
    echo "      ${actual}"
}

mkdir -p "${SRC_ROOT}"

prepare_repo \
    "IsaacLab" \
    "${ISAACLAB_URL}" \
    "${ISAACLAB_COMMIT}"

prepare_repo \
    "unitree_rl_lab" \
    "${UNITREE_RL_URL}" \
    "${UNITREE_RL_COMMIT}"

prepare_repo \
    "unitree_ros" \
    "${UNITREE_ROS_URL}" \
    "${UNITREE_ROS_COMMIT}"

echo
echo "========================================"
echo " WORKSPACE READY"
echo "========================================"
echo
echo "Project:"
echo "  ${REPO_ROOT}"
echo
echo "Pinned dependencies:"
echo "  ${SRC_ROOT}/IsaacLab"
echo "  ${SRC_ROOT}/unitree_rl_lab"
echo "  ${SRC_ROOT}/unitree_ros"
echo
echo "Next:"
echo "  cd ${REPO_ROOT}"
echo "  docker/isaac-lab/build.sh"
echo "  docker/isaac-lab/run.sh"
echo
echo "Then run the documented stair smoke test."
