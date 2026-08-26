# PC2 event-handshake deployment

The scripts default to this deployment root:

```text
/home/unitree/telefonica/event_handshake
```

For another location, export `IEA_G1_DEPLOY_ROOT`. The scripts also accept
`ROS_SETUP` (default `/opt/ros/foxy/setup.bash`) and `G1_NETWORK_INTERFACE`
(default `eth0`). Keeping these settings outside tracked files prevents local
machine paths from becoming project requirements.


Use ROS domain `42` for the event-facing ROS API. Domain `0` contains Unitree's
native DDS traffic and is intentionally isolated from the ROS API participant.
The hardware backend independently initializes Unitree SDK domain `0` on the
configured robot network interface.

```bash
export IEA_G1_DEPLOY_ROOT=/home/unitree/telefonica/event_handshake
cd "$IEA_G1_DEPLOY_ROOT/ros2_ws"
source /opt/ros/foxy/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

Dry run (recommended through the reviewed wrapper):

```bash
"$IEA_G1_DEPLOY_ROOT/scripts/pc2_run_handshake_dry.sh"
```

The repository configuration always keeps `hardware_enabled: false`. Physical
testing must use an explicit per-run override after completing the safety setup:

```bash
"$IEA_G1_DEPLOY_ROOT/scripts/pc2_run_handshake_hardware.sh"
```

Never store `hardware_enabled: true` in the deployment configuration.

BrainCo hand commands have a second independent interlock. They remain dry-run
unless `hand_control_enabled:=true` is supplied on the hardware invocation.
The integrated sequence opens the right hand before arm action 99 on normal
completion, cancellation, disable, shutdown, and error cleanup. Keep both
flags false for local and PC2 dry runs.

Use the consolidated operator command from another terminal:

```bash
./scripts/pc2_handshake.sh state
./scripts/pc2_handshake.sh enable
./scripts/pc2_handshake.sh start
./scripts/pc2_handshake.sh cancel
./scripts/pc2_handshake.sh disable
```
