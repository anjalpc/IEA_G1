#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pc2_event_common.sh"

if pgrep -af '[h]andshake_node' >/dev/null; then
  echo "Refusing to start: a handshake_node process is already running." >&2
  pgrep -af '[h]andshake_node' >&2
  exit 1
fi

echo "Starting the event handshake node in DRY RUN mode."
exec ros2 launch g1_robot_bridge event_handshake.launch.py \
  hardware_enabled:=false \
  network_interface:="${G1_NETWORK_INTERFACE:-eth0}"
