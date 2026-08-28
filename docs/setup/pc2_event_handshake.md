# PC2 event-handshake deployment

The robot uses the same reviewed Git history as development but checks out only
the production files. Create a partial sparse clone:

```bash
cd /home/unitree
git clone --filter=blob:none --no-checkout \
  <COMPANY_GITLAB_REPOSITORY_URL> IEA_G1
cd /home/unitree/IEA_G1
git sparse-checkout init --no-cone
git sparse-checkout set --no-cone \
  '/docs/setup/pc2_event_handshake.md' \
  '/ros2_ws/src/g1_robot_interfaces/' \
  '/ros2_ws/src/g1_handshake/' \
  '/ros2_ws/src/g1_hardware_bridge/' \
  '/scripts/build.sh' \
  '/scripts/start.sh' \
  '/scripts/status.sh' \
  '/scripts/pc2_handshake.sh' \
  '/vendor/brainco_hand_server/'
git checkout main
./scripts/build.sh
```

The PC2 working tree now contains only this guide, the three production
packages, four runtime scripts, and the BrainCo runtime. Git remembers the
sparse selection for subsequent pulls.

The supported robot checkout is:

```text
/home/unitree/IEA_G1
```

`ROS_SETUP` defaults to `/opt/ros/foxy/setup.bash` and
`G1_NETWORK_INTERFACE` defaults to `eth0`.


Use ROS domain `42` for the event-facing ROS API. Domain `0` contains Unitree's
native DDS traffic and is intentionally isolated from the ROS API participant.
The hardware backend independently initializes Unitree SDK domain `0` on the
configured robot network interface.

```bash
cd /home/unitree/IEA_G1/ros2_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

Start BrainCo and both ROS nodes with one command:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh
```

The hardware bridge directly owns the reviewed arm and BrainCo hand
backends. The integrated sequence restores the calibrated neutral hand posture
before arm action 99 on completion, cancellation, timeout, shutdown, and error
cleanup.

Use the operator command from another terminal:

```bash
./scripts/pc2_handshake.sh state
./scripts/pc2_handshake.sh start
./scripts/pc2_handshake.sh grip
./scripts/pc2_handshake.sh cancel
```

`start` executes action 27 and waits. Issue `grip` only after the arm is
presented: the hand closes to the validated target, holds for five seconds,
opens, and action 99 releases the arm. A 15-second wait timeout performs the
same open-and-release cleanup without gripping.
