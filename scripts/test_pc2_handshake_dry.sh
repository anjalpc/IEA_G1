#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${IEA_G1_DEPLOY_ROOT:-/home/unitree/telefonica/event_handshake}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"
LOG_FILE="${TMPDIR:-/tmp}/iea_g1_handshake_dry.log"

cd "${DEPLOY_ROOT}/ros2_ws"
source "${ROS_SETUP}"
source install/setup.bash
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

ros2 launch g1_robot_bridge event_handshake.launch.py \
  hardware_enabled:=false \
  >"${LOG_FILE}" 2>&1 &
node_pid=$!

cleanup() {
  kill -INT "${node_pid}" 2>/dev/null || true
  wait "${node_pid}" 2>/dev/null || true
}
trap cleanup EXIT

sleep 2
ros2 service call /event/handshake/operator_enable \
  std_srvs/srv/SetBool "{data: true}"
ros2 service call /event/handshake/start std_srvs/srv/Trigger "{}"
sleep 3

echo "=== node log ==="
sed -n '1,160p' "${LOG_FILE}"
