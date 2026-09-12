# Robot deployment (PC2)

This guide installs and runs the physical handshake and card-presentation stack directly on the G1's PC2 computer.

## Expected platform

- user: `unitree`
- checkout: `/home/unitree/IEA_G1`
- Ubuntu 20.04 ARM64
- ROS 2 Foxy: `/opt/ros/foxy/setup.bash`
- robot network interface: `eth0` by default
- Unitree SDK2 Python available to `/usr/bin/python3`
- BrainCo hands connected and accessible by the `unitree` user
- RealSense Python module and native library available on the robot

The vendored BrainCo ARM64 server is included in the repository. The Unitree system libraries and RealSense installation are robot-local dependencies and are not bundled.

## 1. Clone

On PC2:

```bash
cd /home/unitree
git clone <REPOSITORY_URL> IEA_G1
cd /home/unitree/IEA_G1
```

For an existing checkout:

```bash
cd /home/unitree/IEA_G1
git pull --ff-only
```

Do not overwrite `ros2_ws/install` with build output from another computer. Build on PC2 after updating source.

## 2. Check BrainCo device permissions

The ROS packages do not require a udev rule. The BrainCo server only requires permission to open its USB serial device. If it already runs as the `unitree` user, skip this step.

`config/udev/70-brainco-hand.rules.example` is an optional template for a machine where normal users cannot open the device. It matches one FTDI vendor/product pair and must be updated with the serial number reported by `udevadm info`; do not install it unchanged on another robot.

If a rule is needed, the administrator must also create its `brainco` group, add the operator to that group, install the customized rule under `/etc/udev/rules.d/`, reload udev, and reconnect the device. This is operating-system configuration, not part of the ROS runtime.

## 3. Verify robot-local dependencies

```bash
source /opt/ros/foxy/setup.bash
/usr/bin/python3 -c 'import unitree_sdk2py; print("Unitree SDK2 Python: OK")'
```

The default RealSense locations used by `scripts/start.sh` are:

```text
/home/unitree/librealsense-install/lib/python3.8/site-packages
/home/unitree/librealsense-install/lib
```

Verify the dependency in a clean, non-interactive environment:

```bash
env -i HOME=/home/unitree \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -c '
    source /opt/ros/foxy/setup.bash
    export PYTHONPATH=/home/unitree/librealsense-install/lib/python3.8/site-packages
    export LD_LIBRARY_PATH=/home/unitree/librealsense-install/lib
    /usr/bin/python3 -c "import pyrealsense2; print(\"RealSense: OK\")"
  '
```

If the installation is elsewhere, set `REALSENSE_PYTHON_PATH` and `REALSENSE_LIBRARY_PATH` when starting the stack.

Verify the bundled BrainCo files:

```bash
cd /home/unitree/IEA_G1/vendor/brainco_hand_server
sha256sum -c SHA256SUMS
```

## 4. Build on PC2

```bash
cd /home/unitree/IEA_G1
chmod +x scripts/*.sh scripts/event_gateway.py
./scripts/build.sh
```

This builds only the physical packages required by PC2. The simulation bridge is not needed.

## 5. Start and verify

For a direct operator test:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh
```

Leave it running and inspect from another terminal:

```bash
cd /home/unitree/IEA_G1
./scripts/status.sh
```

A healthy result includes:

- BrainCo server and both hands ready;
- the ROS launch and all required nodes;
- `event_gateway.py` and `/tmp/g1_event_control.sock`;
- handshake and card ROS services on domain 42.

Stop cleanly:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh stop
```

## Manual checks

On PC2:

```bash
pgrep -af 'start.sh|event_gateway.py|brainco_hand_server|ros2 launch g1_bringup'
test -S /tmp/g1_event_control.sock && echo GATEWAY_OK || echo GATEWAY_MISSING
```

Manual ROS actions:

```bash
cd /home/unitree/IEA_G1
./scripts/event_control.sh handshake start
./scripts/event_control.sh handshake grip
./scripts/event_control.sh handshake cancel
./scripts/event_control.sh card hold
./scripts/event_control.sh card present
./scripts/event_control.sh card cancel
```

## Logs and recovery

Runtime PID files are under `/tmp/g1_interaction_stack`. When running interactively, startup errors are printed in the `start.sh` terminal. If output is redirected by a service, inspect the log path configured by that service.

If startup fails:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh stop
./scripts/status.sh
```

Fix the first process failure in the log. Because the launch is supervised, one required camera, detector, bridge, or gateway failure intentionally takes the whole interaction stack offline.
