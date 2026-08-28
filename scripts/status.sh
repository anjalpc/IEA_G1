#!/usr/bin/env bash

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u

echo "Processes:"
pgrep -af 'brainco_hand_server|handshake_node|g1_hardware_bridge' || true
echo
echo "ROS nodes:"
ros2 node list
