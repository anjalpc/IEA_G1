#!/usr/bin/env bash

# One operator entry point for the complete physical handshake/card stack.

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
export ROS_DOMAIN_ID="${G1_APPLICATION_DOMAIN_ID:-42}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

if [[ ! -x "${SERVER_BINARY}" || \
      ! -f "${SERVER_LIBRARY}/libbc_stark_sdk.so" ]]; then
    echo "ERROR: BrainCo runtime is incomplete in ${SERVER_ROOT}." >&2
    exit 1
fi

if [[ -f "${SERVER_ROOT}/SHA256SUMS" ]]; then
    (
        cd "${SERVER_ROOT}"
        sha256sum -c SHA256SUMS
    )
fi


# ---------------------------------------------------------------------------
# Prevent duplicate stack
# ---------------------------------------------------------------------------

process_pattern='[b]rainco_hand_server|[h]andshake_node|[g]1_hardware_bridge'
process_pattern+='|[h]and_presence_node|[u]nitree_video_helper'
process_pattern+='|[e]vent_card_presentation'

if pgrep -af "${process_pattern}" >/dev/null; then
    echo "ERROR: G1 interaction processes are already running." >&2
    pgrep -af "${process_pattern}" >&2
    exit 1
fi


brainco_pid=""
launch_pid=""


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

stop_group() {
    local signal="$1"
    local pid="$2"

    [[ -n "${pid}" ]] || return 0

    kill -"${signal}" -- "-${pid}" 2>/dev/null || true
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


# ---------------------------------------------------------------------------
# BrainCo lifecycle
# ---------------------------------------------------------------------------

start_brainco() {
    setsid env \
        LD_LIBRARY_PATH="${SERVER_LIBRARY}:/usr/local/lib" \
        "${SERVER_BINARY}" \
        --network "${NETWORK_INTERFACE}" &

    brainco_pid=$!
}


stop_brainco() {
    if [[ -z "${brainco_pid}" ]]; then
        return 0
    fi

    if kill -0 "${brainco_pid}" 2>/dev/null; then
        echo "Stopping BrainCo server..."

        stop_group TERM "${brainco_pid}"

        if ! wait_for_exit "${brainco_pid}" 30; then
            echo "WARN: BrainCo did not stop cleanly; forcing exit." >&2
            stop_group KILL "${brainco_pid}"
        fi
    fi

    wait "${brainco_pid}" 2>/dev/null || true
    brainco_pid=""
}


wait_for_hand() {
    local hand="$1"
    local timeout_s="$2"

    # The helper itself has a timeout, but a shell-level timeout is also
    # required because a missing BrainCo worker can otherwise leave it stuck.
    timeout \
        --foreground \
        --signal=INT \
        --kill-after=1s \
        "${timeout_s}s" \
        env ROS_DOMAIN_ID=0 \
        python3 -m g1_hardware_bridge.hand.helper state \
        --hand "${hand}" \
        --interface "${NETWORK_INTERFACE}" \
        --timeout "${timeout_s}" \
        >/dev/null 2>&1
}


brainco_both_hands_ready() {
    local timeout_s="$1"

    echo "Waiting for BrainCo left hand..."

    if ! wait_for_hand left "${timeout_s}"; then
        echo "WARN: BrainCo left hand is not ready." >&2
        return 1
    fi

    echo "BrainCo left hand: READY"

    echo "Waiting for BrainCo right hand..."

    if ! wait_for_hand right "${timeout_s}"; then
        echo "WARN: BrainCo right hand is not ready." >&2
        return 1
    fi

    echo "BrainCo right hand: READY"

    return 0
}


start_brainco_with_retry() {
    local attempt

    for attempt in 1 2 3; do
        echo
        echo "Starting BrainCo server (attempt ${attempt}/3)..."

        start_brainco

        # Allow vendor serial enumeration to complete.
        sleep 1

        if brainco_both_hands_ready 4; then
            echo
            echo "BrainCo left hand: READY"
            echo "BrainCo right hand: READY"
            echo "BrainCo subsystem: READY"
            return 0
        fi

        echo
        echo "WARN: BrainCo did not detect both hands on attempt ${attempt}." >&2

        stop_brainco

        if [[ ${attempt} -lt 3 ]]; then
            echo "Retrying BrainCo initialization..."
            sleep 1
        fi
    done

    echo
    echo "ERROR: both BrainCo hands were not ready after three attempts." >&2
    return 1
}


# ---------------------------------------------------------------------------
# Safe cleanup
# ---------------------------------------------------------------------------

cleanup() {
    local status=$?

    trap - INT TERM EXIT

    #
    # Shutdown order is intentional:
    #
    # 1. ROS/hardware bridge first
    #    - neutralize BrainCo hand
    #    - retract/release card arm if needed
    #    - release Unitree arm authority
    #
    # 2. BrainCo server afterwards
    #

    if [[ -n "${launch_pid}" ]] && \
       kill -0 "${launch_pid}" 2>/dev/null; then

        echo
        echo "Stopping ROS interaction stack..."

        stop_group INT "${launch_pid}"

        # Up to 40 seconds for bridge-safe shutdown.
        if ! wait_for_exit "${launch_pid}" 200; then
            echo "WARN: ROS stack did not stop after SIGINT." >&2
            stop_group TERM "${launch_pid}"

            if ! wait_for_exit "${launch_pid}" 25; then
                echo "WARN: forcing ROS stack shutdown." >&2
                stop_group KILL "${launch_pid}"
            fi
        fi

        wait "${launch_pid}" 2>/dev/null || true
        launch_pid=""
    fi

    stop_brainco

    exit "${status}"
}


trap cleanup INT TERM EXIT


# ---------------------------------------------------------------------------
# Start BrainCo
# ---------------------------------------------------------------------------

if ! start_brainco_with_retry; then
    exit 1
fi


# ---------------------------------------------------------------------------
# Start ROS stack
# ---------------------------------------------------------------------------

echo
echo "Starting G1 ROS interaction stack..."

setsid ros2 launch \
    g1_bringup \
    event_interaction.launch.py &

launch_pid=$!


# Give ros2 launch a brief chance to fail immediately if configuration
# or package setup is broken.
sleep 1

if ! kill -0 "${launch_pid}" 2>/dev/null; then
    echo "ERROR: ROS interaction stack failed during startup." >&2
    exit 1
fi


echo
echo "=========================================="
echo "G1 INTERACTION STACK: READY"
echo "BrainCo left hand:      READY"
echo "BrainCo right hand:     READY"
echo "ROS application domain: ${ROS_DOMAIN_ID}"
echo "Waiting for card/handshake triggers..."
echo "=========================================="
echo


# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------

# BrainCo and ROS are both required.
# If either exits unexpectedly, tear down the surviving sibling.

set +e

wait -n \
    "${brainco_pid}" \
    "${launch_pid}"

status=$?

set -e


echo
echo "ERROR: a required G1 interaction component exited." >&2


# An unexpected status 0 is still considered a stack failure.
if [[ ${status} -eq 0 ]]; then
    status=1
fi

exit "${status}"