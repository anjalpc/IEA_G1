#!/usr/bin/env bash

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${G1_APPLICATION_DOMAIN_ID:-42}"

echo "Processes:"
pgrep -af \
    '[b]rainco_hand_server|[r]os2 launch g1_bringup|[h]ardware_bridge|[h]andshake_node|[h]and_presence_node|[r]gb_helper|[c]amera_bridge|[c]ard_presentation|[c]ard_handover_detector|[a]rm.state_helper|[e]vent_gateway.py' \
    || true
echo
echo "Supervisor PID records:"
if [[ -d "${G1_RUNTIME_DIR:-/tmp/g1_interaction_stack}" ]]; then
    for pid_file in "${G1_RUNTIME_DIR:-/tmp/g1_interaction_stack}"/*.pid; do
        [[ -f "${pid_file}" ]] || continue
        echo "$(basename "${pid_file}"): $(cat "${pid_file}")"
    done
else
    echo "none"
fi
echo
echo "Dashboard event socket:"
if [[ -S "${G1_EVENT_SOCKET:-/tmp/g1_event_control.sock}" ]]; then
    echo "READY: ${G1_EVENT_SOCKET:-/tmp/g1_event_control.sock}"
else
    echo "MISSING: ${G1_EVENT_SOCKET:-/tmp/g1_event_control.sock}"
fi
echo
echo "ROS nodes:"
ros2 node list
