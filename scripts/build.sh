#!/usr/bin/env bash

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
set -u
cd "${PROJECT_ROOT}/ros2_ws"
colcon build --symlink-install --packages-select \
    g1_robot_interfaces \
    g1_handshake \
    g1_hardware_bridge \
    g1_card_presentation \
    g1_bringup
