# PC2 interaction-stack deployment

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
  '/config/udev/70-brainco-hand.rules' \
  '/ros2_ws/src/g1_bringup/' \
  '/ros2_ws/src/g1_robot_interfaces/' \
  '/ros2_ws/src/g1_handshake/' \
  '/ros2_ws/src/g1_card_presentation/' \
  '/ros2_ws/src/g1_hardware_bridge/' \
  '/scripts/build.sh' \
  '/scripts/start.sh' \
  '/scripts/event_control.sh' \
  '/scripts/status.sh' \
  '/scripts/pc2_handshake.sh' \
  '/vendor/brainco_hand_server/'
git checkout main
./scripts/build.sh
```

The PC2 working tree now contains only this guide, the restricted udev rule,
the bringup package, the four runtime packages, the operator scripts, and the
BrainCo runtime. Git remembers the sparse selection for subsequent
pulls.

The supported robot checkout is:

```text
/home/unitree/IEA_G1
```

`ROS_SETUP` defaults to `/opt/ros/foxy/setup.bash` and
`G1_NETWORK_INTERFACE` defaults to `eth0`.

## BrainCo serial permissions

The BrainCo server runs as `unitree` without `sudo`. Install the restricted
device rule once:

```bash
sudo groupadd --system brainco 2>/dev/null || true
sudo usermod -aG brainco unitree
sudo install -m 0644 \
  config/udev/70-brainco-hand.rules \
  /etc/udev/rules.d/70-brainco-hand.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --subsystem-match=tty
```

Start a fresh login session and verify:

```bash
id
ls -l /dev/ttyUSB*
```

The login must include group `brainco`, and the matching FTDI ports must belong
to `root:brainco`. No interactive password is required by `start.sh`.

The runtime scripts select ROS domain `42` automatically for the event-facing
API. Domain `0` contains Unitree's native DDS traffic and is intentionally
isolated from the ROS API participant. The hardware backend independently uses
Unitree SDK domain `0` on the configured robot network interface.

The following environment setup is required only for direct `ros2` diagnostic
commands:

```bash
cd /home/unitree/IEA_G1/ros2_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

Start BrainCo and the supervised ROS stack with one command:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh
```

The hardware bridge directly owns the reviewed arm and BrainCo hand
backends. The integrated sequence restores the calibrated neutral hand posture
before arm action 99 on completion, cancellation, timeout, shutdown, and error
cleanup.

The hardware package also owns the physical camera adapters. The standalone
Unitree RGB helper remains isolated on native DDS domain `0` and writes its
latest frame to shared memory. The ROS-facing camera bridge is the sole
RealSense owner, reads that RGB frame, and publishes application-facing data
on domain `42`:

```text
/g1/camera/rgb/compressed
/g1/camera/depth/image_raw
```

The handshake presence node subscribes to these topics and never opens camera
devices. This permits preview, recording, and future perception consumers to
share the same frames without competing for `/dev/video*`.

The bridge also watches Unitree's event-driven `/arm/action/state` topic on
native DDS domain `0`. Silence at startup is the robot's normal idle behavior.
An observed external nonzero action blocks new handshake commands and makes an
active handshake open only its hand while leaving the externally controlled
arm untouched. A final action ID `0` restores availability. If the monitor
process exits, physical arm commands remain blocked.

Use the operator command from another terminal:

```bash
./scripts/pc2_handshake.sh state
./scripts/pc2_handshake.sh start
./scripts/pc2_handshake.sh grip
./scripts/pc2_handshake.sh cancel
```

The first `start` call executes action 27 and starts a 15-second presentation
timeout. A second `start` call while the state is `ARM_PRESENTED` cancels that
timer, restores the hand to neutral, and releases the arm with action 99. If no
second call or grip follows, the timeout performs the same safe release. `grip`
closes to the validated target, holds for five seconds, returns to neutral, and
executes action 99.
