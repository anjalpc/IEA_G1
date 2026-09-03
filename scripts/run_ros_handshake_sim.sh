#!/usr/bin/env bash

# Run both simulation nodes under one fail-together ROS launch supervisor.
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f ros2_ws/install/setup.bash ]]; then
    echo "ERROR: ROS workspace is not built. Run:" >&2
    echo "  cd ros2_ws && colcon build --symlink-install" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

# Keep simulation isolated from physical robot communication.
export ROS_DOMAIN_ID="${G1_SIMULATION_DOMAIN_ID:-77}"

# Preserve command-line arguments for the simulation node inside the launch file.
printf -v G1_SIMULATION_ARGUMENTS '%q ' "$@"
export G1_SIMULATION_ARGUMENTS

launch_pid=""
trigger_pid=""

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

    if [[ -n "${trigger_pid}" ]] && kill -0 "${trigger_pid}" 2>/dev/null; then
        kill -TERM "${trigger_pid}" 2>/dev/null || true
    fi

    if [[ -n "${launch_pid}" ]] && kill -0 "${launch_pid}" 2>/dev/null; then
        kill -INT "${launch_pid}" 2>/dev/null || true

        if ! wait_for_exit "${launch_pid}" 25; then
            kill -TERM "${launch_pid}" 2>/dev/null || true
        fi

        if ! wait_for_exit "${launch_pid}" 15; then
            kill -KILL "${launch_pid}" 2>/dev/null || true
        fi
    fi

    [[ -z "${trigger_pid}" ]] || wait "${trigger_pid}" 2>/dev/null || true
    [[ -z "${launch_pid}" ]] || wait "${launch_pid}" 2>/dev/null || true

    exit "${status}"
}

trap cleanup INT TERM EXIT

ros2 launch g1_simulation_bridge simulation_handshake.launch.py &
launch_pid=$!

(
    deadline=$((SECONDS + 30))

    until ros2 node list | grep -qx '/g1_simulation_bridge' &&
          ros2 service type /event/handshake/start >/dev/null 2>&1; do
        if ! kill -0 "${launch_pid}" 2>/dev/null; then
            echo "ERROR: simulation launch exited before becoming ready." >&2
            exit 1
        fi

        if ((SECONDS >= deadline)); then
            echo "ERROR: simulation stack did not become ready." >&2
            exit 1
        fi

        sleep 0.2
    done

    ros2 service call \
        /event/handshake/start \
        std_srvs/srv/Trigger "{}"

    if [[ -n "${G1_AUTO_CANCEL_AFTER_S:-}" ]]; then
        sleep "${G1_AUTO_CANCEL_AFTER_S}"

        ros2 service call \
            /event/handshake/cancel \
            std_srvs/srv/Trigger "{}"
    else
        sleep "${G1_AUTO_GRIP_AFTER_S:-0.5}"

        ros2 service call \
            /event/handshake/grip \
            std_srvs/srv/Trigger "{}"
    fi
) || {
    echo "ERROR: automatic simulation command sequence failed." >&2
    kill -TERM "${launch_pid}" 2>/dev/null || true
} &

trigger_pid=$!

# The ROS launch process owns both required nodes. If either crashes, its
# OnProcessExit handler shuts down the sibling and makes this wait return.
wait "${launch_pid}"
