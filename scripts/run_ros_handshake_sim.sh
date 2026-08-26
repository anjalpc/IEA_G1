#!/usr/bin/env bash

# ROS 2 setup scripts legitimately inspect variables that may be unset, so
# nounset (-u) is intentionally not enabled here.
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
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

node_pid=""
trigger_pid=""

cleanup() {
    trap - INT TERM EXIT
    [[ -z "${trigger_pid}" ]] || kill "${trigger_pid}" 2>/dev/null || true
    [[ -z "${node_pid}" ]] || kill -- "-${node_pid}" 2>/dev/null || true
    [[ -z "${trigger_pid}" ]] || wait "${trigger_pid}" 2>/dev/null || true
    [[ -z "${node_pid}" ]] || wait "${node_pid}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

setsid ros2 launch g1_robot_bridge event_handshake.launch.py \
    simulation_enabled:=true &
node_pid="$!"

(
    sleep 2
    ros2 service call /event/handshake/operator_enable \
        std_srvs/srv/SetBool "{data: true}"
    ros2 service call /event/handshake/start std_srvs/srv/Trigger "{}"
) &
trigger_pid="$!"

ros2 run g1_simulation_bridge simulation_bridge "$@"
