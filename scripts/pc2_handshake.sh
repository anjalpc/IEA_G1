#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/pc2_event_common.sh"

case "${1:-}" in
  enable)
    ros2 service call /event/handshake/operator_enable \
      std_srvs/srv/SetBool "{data: true}"
    ;;
  disable)
    ros2 service call /event/handshake/operator_enable \
      std_srvs/srv/SetBool "{data: false}"
    ;;
  start)
    ros2 service call /event/handshake/start std_srvs/srv/Trigger "{}"
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
    echo "Usage: $0 {enable|disable|start|cancel|state}" >&2
    exit 2
    ;;
esac
