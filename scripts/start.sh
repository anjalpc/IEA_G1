#!/usr/bin/env bash

# One operator entry point for the complete physical handshake feature.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
SERVER_ROOT="${PROJECT_ROOT}/vendor/brainco_hand_server"
SERVER_BINARY="${SERVER_ROOT}/bin/brainco_hand_server"
SERVER_LIBRARY="${SERVER_ROOT}/lib/aarch64"
NETWORK_INTERFACE="${G1_NETWORK_INTERFACE:-eth0}"

source "${ROS_SETUP}"
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi
if [[ ! -x "${SERVER_BINARY}" || \
      ! -f "${SERVER_LIBRARY}/libbc_stark_sdk.so" ]]; then
    echo "ERROR: validated BrainCo runtime is incomplete in ${SERVER_ROOT}." >&2
    exit 1
fi
if [[ -f "${SERVER_ROOT}/SHA256SUMS" ]]; then
    (cd "${SERVER_ROOT}" && sha256sum -c SHA256SUMS)
fi
if pgrep -af '[b]rainco_hand_server|[h]andshake_node|[g]1_hardware_bridge' \
        >/dev/null; then
    echo "ERROR: handshake processes are already running." >&2
    pgrep -af '[b]rainco_hand_server|[h]andshake_node|[g]1_hardware_bridge' >&2
    exit 1
fi

brainco_pid=""
launch_pid=""

stop_group() {
    local signal="$1"
    local pid="$2"
    local privileged="${3:-false}"
    [[ -n "${pid}" ]] || return 0
    if [[ "${privileged}" == true ]]; then
        sudo kill -"${signal}" -- "-${pid}" 2>/dev/null || true
    else
        kill -"${signal}" -- "-${pid}" 2>/dev/null || true
    fi
}

wait_for_exit() {
    local pid="$1"
    local attempts="$2"
    for ((count = 0; count < attempts; count++)); do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 0.2
    done
    return 1
}

cleanup() {
    local status=$?
    trap - INT TERM EXIT

    # Keep BrainCo alive until ROS has neutralized the hand and lowered the arm.
    if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
        stop_group INT "${launch_pid}"
        if ! wait_for_exit "${launch_pid}" 60; then
            stop_group TERM "${launch_pid}"
        fi
        wait "${launch_pid}" 2>/dev/null || true
    fi
    if [[ -n "${brainco_pid}" ]] && kill -0 "${brainco_pid}" 2>/dev/null; then
        stop_group TERM "${brainco_pid}" true
        if ! wait_for_exit "${brainco_pid}" 30; then
            stop_group KILL "${brainco_pid}" true
        fi
        wait "${brainco_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup INT TERM EXIT

sudo -v
setsid sudo env \
    LD_LIBRARY_PATH="${SERVER_LIBRARY}:/usr/local/lib" \
    "${SERVER_BINARY}" --network "${NETWORK_INTERFACE}" &
brainco_pid=$!

echo "Waiting for the BrainCo right hand..."
if ! env ROS_DOMAIN_ID=0 python3 -m \
        g1_hardware_bridge.brainco_hand_helper state \
        --hand right --interface "${NETWORK_INTERFACE}" --timeout 15 \
        >/dev/null; then
    echo "ERROR: BrainCo right hand did not become ready." >&2
    exit 1
fi

echo "BrainCo ready; starting handshake nodes."
setsid ros2 launch g1_hardware_bridge hardware_handshake.launch.py &
launch_pid=$!
wait "${launch_pid}"
