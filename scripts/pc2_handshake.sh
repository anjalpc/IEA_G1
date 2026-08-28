#!/usr/bin/env bash
# Foxy setup scripts inspect variables that may not exist yet, so nounset is
# enabled only after both environment files have been sourced.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

source "${ROS_SETUP}"
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

case "${1:-}" in
  start)
    ros2 service call /event/handshake/start std_srvs/srv/Trigger "{}"
    ;;
  grip)
    ros2 service call /event/handshake/grip std_srvs/srv/Trigger "{}"
    ;;
  cancel)
    ros2 service call /event/handshake/cancel std_srvs/srv/Trigger "{}"
    ;;
  state)
    timeout 2 ros2 topic echo /event/handshake/state || status=$?
    if [[ ${status:-0} -ne 0 && ${status:-0} -ne 124 ]]; then
      exit "${status}"
    fi
    ;;
  *)
    echo "Usage: $0 {start|grip|cancel|state}" >&2
    exit 2
    ;;
esac
