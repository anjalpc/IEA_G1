#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${IEA_G1_DEPLOY_ROOT:-/home/unitree/telefonica/event_handshake}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

source "${ROS_SETUP}"
source "${DEPLOY_ROOT}/ros2_ws/install/setup.bash"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
