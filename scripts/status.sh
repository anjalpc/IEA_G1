#!/usr/bin/env bash

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${G1_APPLICATION_DOMAIN_ID:-42}"

echo "Processes:"
pgrep -af 'brainco_hand_server|handshake_node|g1_hardware_bridge' || true
echo
echo "ROS nodes:"
ros2 node list
