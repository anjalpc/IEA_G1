#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pc2_event_common.sh"

if pgrep -af '[h]andshake_node' >/dev/null; then
  echo "Refusing to start: a handshake_node process is already running." >&2
  pgrep -af '[h]andshake_node' >&2
  exit 1
fi

echo "Starting the event handshake node in HARDWARE mode."
echo "The operator remains disabled until pc2_handshake.sh enable is run."
echo "BrainCo commands remain dry-run unless hand_control_enabled is explicitly set."
exec ros2 launch g1_robot_bridge event_handshake.launch.py \
  hardware_enabled:=true \
  hand_control_enabled:=false \
  network_interface:="${G1_NETWORK_INTERFACE:-eth0}"
