#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

source "${ROS_SETUP}"
source "${PROJECT_ROOT}/ros2_ws/install/setup.bash"
set -u

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${G1_APPLICATION_DOMAIN_ID:-42}"

group="${1:-}"
command="${2:-}"

case "${group}:${command}" in

handshake:start)
    ros2 service call /event/handshake/start std_srvs/srv/Trigger "{}"
    ;;

handshake:grip)
    ros2 service call /event/handshake/grip std_srvs/srv/Trigger "{}"
    ;;

handshake:cancel)
    ros2 service call /event/handshake/cancel std_srvs/srv/Trigger "{}"
    ;;

handshake:state)
    ros2 topic echo /event/handshake/state std_msgs/msg/String
    ;;

card:grip)
    ros2 service call /event/card/grip std_srvs/srv/Trigger "{}"
    ;;

card:release)
    ros2 service call /event/card/release std_srvs/srv/Trigger "{}"
    ;;

card:present)
    ros2 service call /event/card/present std_srvs/srv/Trigger "{}"
    ;;

card:retract)
    ros2 service call /event/card/retract std_srvs/srv/Trigger "{}"
    ;;

card:cancel)
    ros2 service call /event/card/cancel std_srvs/srv/Trigger "{}"
    ;;

card:state)
    ros2 topic echo /event/card/state std_msgs/msg/String
    ;;

*)
    echo "Usage:"
    echo "  $0 handshake {start|grip|cancel|state}"
    echo "  $0 card {grip|release|present|retract|cancel|state}"
    exit 2
    ;;

esac