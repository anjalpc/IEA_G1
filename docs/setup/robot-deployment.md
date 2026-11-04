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
- pinned ROS 2 Foxy RealSense driver packages installed on the robot

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

The physical stack uses the pinned ROS 2 Foxy RealSense packages rather than
loading the earlier custom Python installation into the runtime environment.
Verify the installed packages and executable:

```bash
dpkg-query -W \
  ros-foxy-librealsense2 \
  ros-foxy-realsense2-camera-msgs \
  ros-foxy-realsense2-camera \
  ros-foxy-realsense2-description \
  ros-foxy-xacro

ros2 pkg prefix realsense2_camera
ros2 pkg executables realsense2_camera
```

The custom installation under `/home/unitree/librealsense-install` remains
available only as rollback. `scripts/start.sh` intentionally does not prepend
it to `PYTHONPATH` or `LD_LIBRARY_PATH`, preventing it from overriding the
pinned Foxy driver library.

Verify the bundled BrainCo files:

```bash
cd /home/unitree/IEA_G1/third_party/brainco_hand_server
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
- `/d435/d435` as the sole D435i owner with the configured sensor streams;
- `/g1_hardware_state` publishing measured waist and pelvis IMU state;
- `event_gateway.py` and `/tmp/g1_event_control.sock`;
- handshake and card ROS services on domain 42.

Stop cleanly:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh stop
```

## Verify physical LowState

On PC2, with the supported stack running:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/IEA_G1/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

ros2 node list | grep '^/g1_hardware_state$'
ros2 topic info /joint_states --verbose
ros2 topic echo --once /joint_states
ros2 topic info /g1/state/imu_raw --verbose
ros2 topic echo --once /g1/state/imu_raw
```

The message must contain exactly these measured names in order:

```text
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
```

`/joint_states` must have one physical publisher with reliable, volatile,
depth-one QoS.

`/g1/state/imu_raw` must have one physical publisher with best-effort,
volatile, depth-one QoS. Its frame must be `imu_in_pelvis`. The message
contains measured orientation, angular velocity, and linear acceleration from
the same accepted LowState tick as the waist positions. All covariance arrays
are zero to represent unknown covariance; no sensor accuracy is invented.

The validated-state boundary intentionally rejects this unknown covariance
until physical calibration or estimator-specific qualification is completed.
Do not start a second physical LowState source.

On the workstation, use the same reachable ROS network, application
domain, and RMW implementation. In the development container:

```bash
cd /workspace/IEA_G1/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

ros2 launch g1_robot_description robot_state.launch.py \
  use_sim_time:=false
```

In another container terminal, verify that remote measurements reach
`robot_state_publisher` and drive the downstream body tree:

```bash
cd /workspace/IEA_G1/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42

ros2 topic info /joint_states
ros2 run tf2_ros tf2_echo pelvis d435_link
```


## Guarded physical MID-360 odometry

Physical navigation keeps raw acquisition on PC2 and computation on the
workstation. It is independent of the handshake/card interaction stack.

PC2 runs the isolated Livox driver on domain `42` with
`/home/unitree/IEA_G1/livox_ws/config/cyclonedds_pc2_wired.xml` and
`mid360_pc2.json`. The DDS file must use `<Domain id="42">`; an
`id="any"` configuration would also affect the child native Unitree
domain-0 runtime.

The Livox driver remaps its outputs to
`/g1/sensors/lidar/points` and `/g1/sensors/lidar/imu_raw`. Exactly one
Livox process may own the MID-360. Separately, the read-only
`g1_hardware_bridge hardware_state` process reads native LowState on
domain `0` and publishes measured `/joint_states` and
`/g1/state/imu_raw` on domain `42`.

The workstation development container uses the installed
`cyclonedds_laptop_wired.xml` and starts:

```bash
ros2 launch g1_navigation_bringup physical_odometry.launch.py
```

This launch starts CPU GLIM, `g1_glim_health`, `g1_glim_planar`,
`g1_physical_body`, and the physical `robot_state_publisher`. The health
boundary publishes unchanged qualified samples on
`/g1/navigation/glim/odometry_validated`. The planar staging node corrects
the inverted MID-360 axes and publishes start-relative planar samples on
`/g1/navigation/glim/odometry_planar`.

The physical-body staging node buffers up to 100 pelvis IMU samples and 20
pending planar samples. It uses the nearest header timestamps only when their
separation is at most 0.05 seconds, then publishes approximately 10 Hz
`odom -> pelvis` messages on
`/g1/navigation/physical/odometry_staged`. The verified standing height is
`0.785 m`. This staged topic publishes no TF and has no navigation authority.

The robot-state runtime consumes measured waist joints and publishes
the `pelvis -> torso_link -> mid360_link` subtree. The physical
MID-360 edge has 180 degrees of local-X roll and the official
2.3-degree pitch.

GLIM runs with `publish_imu2lidar` set to `false` and owns only
`glim_odom -> livox_frame`. The GLIM and physical robot-state trees remain
intentionally isolated; do not add `livox_frame -> mid360_link` because that
would create duplicate TF parentage.

None of the staged topics is `/g1/state/odometry`. This launch does
not publish navigation-authority TF, start mapping or Nav2, or command the
robot. Restarting the launch deliberately establishes a new planar origin.

Before starting Livox, verify that PC2 system time is synchronized. If the clock changes substantially after the driver starts, stop and restart the driver. Qualify the cloud before GLIM: healthy output is about 10 Hz with roughly 20,000 points per cloud. Packet-sized output near 96 points at approximately 2 kHz is invalid.

After staging has been qualified, run the explicit authority proof:

```bash
ros2 launch g1_navigation_bringup physical_odometry_authority.launch.py
```

This adds `g1_physical_odometry_authority`. After strict warmup and plausibility checks pass, it publishes planar `/g1/state/odometry` with `frame_id=odom` and `child_frame_id=base_footprint`, plus the matching `odom -> base_footprint -> pelvis` TF chain. Motion is limited by both per-update bounds and source-time rates of `0.5 m/s` and 60 degrees/s; a source gap above `2.0 s` fails closed. It starts no mapping, Nav2, localization, locomotion, or robot commands. Run it in the foreground; one `Ctrl+C` cleanly stops the workstation stack.

To start read-only physical mapping after the authority proof succeeds:

```bash
ros2 launch g1_navigation_bringup physical_mapping.launch.py
```

This foreground launch adds the MID-360 `/scan` projection and asynchronous
SLAM Toolbox. Verify `/scan`, `/map`, and `map -> odom` before moving the
robot. It starts no localization, Nav2, locomotion, velocity command, or
robot-command node. Press `Ctrl+C` to stop mapping and workstation odometry
authority together.

During physical motion, keep RViz PointCloud2 disabled. Rendering the full
MID-360 cloud alongside CPU GLIM and SLAM caused multi-second estimator
stalls during validation. Map, LaserScan, TF, and RobotModel remain suitable
live displays.

The robot may remain secured in the gantry for stationary qualification.
Before navigation promotion, place it safely on the ground with an operator
and validate forward, lateral, and yaw direction, metric scale, estimator
continuity, and the inverted MID-360 mounting. Stop immediately on any axis
inversion, discontinuity, stale stream, or tracking loss.

The accepted grounded proof measured 0.4678 m for a marked 0.50 m forward
movement and 81.184 degrees for a physical 90-degree left turn. The staging
boundary therefore preserves forward positive X and converts physical left
turning to positive robot yaw without adding empirical scale factors.

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

## Read-only physical navigation-state probe

The `physical_state_probe` command is a manual, time-bounded, read-only PC2
inspection tool. It subscribes to approved Unitree state candidates on DDS
domain `0`. It creates no command publisher and never starts automatically.

Before running it:

1. Secure the G1 in its harness with an operator present.
2. Stop body-policy, Sport, teleoperation, and other command processes.
3. Use the robot-facing network interface explicitly.
4. Run:

```bash
ros2 run g1_hardware_bridge physical_state_probe \
  --interface <ROBOT_NETWORK_INTERFACE> \
  --duration-seconds 10 \
  --minimum-samples 20
```

A normal run prints exactly one line beginning with
`PHYSICAL_STATE_PROBE=`. Retain only that sanitized JSON line. It contains
availability, sample count, cadence, field lengths, progress status, and a
reason code. It contains no raw sensor values, interface names, addresses,
robot identifiers, or exception details.

`VALID` proves only that a candidate was available, finite, sufficiently
sampled, and advancing during the probe. It does not approve that candidate
as navigation odometry.
