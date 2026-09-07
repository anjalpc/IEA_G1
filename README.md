# IEA G1 ROS 2 workspace

This repository contains ROS 2 applications, hardware integration, and MuJoCo
simulation for the Unitree G1. Application logic is kept separate from robot
drivers so the same workflow can use either the physical or simulated bridge.

The currently validated physical workflow presents the right arm with Unitree
action 27, detects a person's hand using RGB and depth camera data, curls the
BrainCo right hand, holds for the configured duration, opens the hand, and
releases the arm with action 99. Manual grip remains available as a fallback.

## Supported environments

| Environment | Platform | ROS 2 | Purpose |
| --- | --- | --- | --- |
| Development container | Ubuntu 22.04 x86_64 | Humble | Build, tests, and MuJoCo development |
| Unitree PC2 | Ubuntu 20.04 aarch64 | Foxy | Physical robot runtime |

Packages deployed to PC2 must remain compatible with ROS 2 Foxy and Python
3.8. Development and automated tests use ROS 2 Humble and Python 3.10.

## Package architecture

The workspace contains six ROS 2 packages:

```text
ros2_ws/src/
├── g1_robot_interfaces/
├── g1_hardware_bridge/
├── g1_handshake/
├── g1_card_presentation/
├── g1_simulation_bridge/
└── g1_bringup/
```

### `g1_robot_interfaces`

Defines the backend-neutral, acknowledged command and feedback transport used
between applications and bridges. It also owns shared command validation.

### `g1_hardware_bridge`

Owns all physical hardware access on PC2:

```text
g1_hardware_bridge/
├── arm/           Unitree actions, state monitoring, and ownership guard
├── hand/          BrainCo commands and helper process
├── camera/
│   ├── node.py           ROS camera publishers and coordination
│   ├── depth.py          RealSense depth acquisition
│   ├── rgb_transport.py  shared-memory RGB transport
│   └── rgb_helper.py     isolated Unitree RGB acquisition process
├── experimental/  disabled left-arm calibration and shadow helpers
├── hardware_bridge_node.py
└── settings.py
```

The package installs two ROS nodes:

- `/g1_hardware_bridge` handles guarded arm and hand commands.
- `/g1_camera_bridge` publishes shared camera frames.

`/g1_camera_bridge` is a node inside `g1_hardware_bridge`; there is no
`g1_camera_bridge` package. Camera settings are stored in
`g1_hardware_bridge/config/camera.yaml`, while arm and hand settings are in
`g1_hardware_bridge/config/hardware.yaml`.

### `g1_handshake`

Owns the handshake state machine, public services, timeouts, automatic
hand-presence detection, and handshake-specific configuration. It consumes
generic bridge commands and camera topics; it does not open physical devices
or import camera drivers.

Configuration files:

- `config/handshake.yaml`: sequence, grip, hold, timeout, and release values.
- `config/hand_presence.yaml`: RGB/depth region and trigger thresholds.

### `g1_card_presentation`

Owns the separate left-hand card-holding workflow. Its verified hand target is
kept separate from the right-hand handshake. Physical left-arm presentation
remains disabled unless explicitly enabled and validated.

### `g1_simulation_bridge`

Implements the same generic bridge operations for MuJoCo. Simulation-specific
targets are stored in `config/simulation.yaml`. This package is not deployed
as part of the physical PC2 runtime.

### `g1_bringup`

Contains composition launch files only. The physical handshake launch starts
and supervises these required processes:

1. Unitree RGB helper;
2. camera bridge;
3. hardware bridge;
4. handshake node;
5. hand-presence node.

If a required process exits, launch shuts down the complete stack so the
hardware bridge can perform its guarded cleanup.

## Camera data flow

Camera acquisition is centralized in `g1_hardware_bridge`:

```text
Unitree RGB camera (native DDS domain 0)
  -> Unitree video helper
  -> shared-memory frame transport
  -> /g1_camera_bridge
  -> /g1/camera/rgb/compressed

RealSense depth camera
  -> /g1_camera_bridge (sole device owner)
  -> /g1/camera/depth/image_raw

RGB + depth topics
  -> /hand_presence
  -> /event/handshake/grip
```

The depth image uses `sensor_msgs/msg/Image` with `16UC1` millimetre values.
The RGB stream uses `sensor_msgs/msg/CompressedImage`. Applications must
subscribe to these ROS topics instead of opening the cameras directly. This
also provides one reusable camera boundary for future navigation and vision
features.

Camera topics:

| Topic | Type | Purpose |
| --- | --- | --- |
| `/g1/camera/rgb/compressed` | `sensor_msgs/msg/CompressedImage` | Shared RGB frames |
| `/g1/camera/depth/image_raw` | `sensor_msgs/msg/Image` | Shared depth frames in millimetres |
| `/event/handshake/hand_presence/debug/compressed` | `sensor_msgs/msg/CompressedImage` | Annotated handshake detector view |

## Handshake workflow

The operator-facing workflow uses two commands:

1. `start` presents the right arm. Calling `start` again while presented opens
   the hand and releases the arm.
2. `grip` curls the right hand, holds it for the configured duration, opens it,
   and releases the arm.

When automatic grip is enabled, the hand-presence node calls `grip` only while
the handshake is in `ARM_PRESENTED` and both configured gates are stable:

- the RGB hand landmark is inside the configured palm region;
- the depth region contains a sufficiently close object;
- the required number of recent samples agree;
- no grip request is already pending or completed for that presentation.

A hand already inside the robot palm when presentation finishes is allowed to
trigger. The region is calibrated for the current manually positioned robot
head. Moving or tilting the head changes the camera geometry and can make the
fixed region inaccurate; recalibrate `config/hand_presence.yaml` after such a
change.

The Unitree arm ownership guard prevents application commands from starting
while another Unitree arm action is active. If controller takeover is detected
during the workflow, the application opens the BrainCo hand but does not issue
action 99 over the external controller action.

## ROS interfaces

Bridge topics:

```text
/g1/bridge/hardware/command
/g1/bridge/hardware/feedback
/g1/bridge/hardware/arm_ownership
/g1/bridge/simulation/command
/g1/bridge/simulation/feedback
```

Handshake API:

| Type | Name | Purpose |
| --- | --- | --- |
| Service | `/event/handshake/start` | Present the arm or toggle an active presentation off |
| Service | `/event/handshake/grip` | Run the timed grip and release sequence |
| Service | `/event/handshake/cancel` | Request safe cancellation and release |
| Topic | `/event/handshake/state` | Application state plus arm ownership information |

Card-presentation API:

| Type | Name | Purpose |
| --- | --- | --- |
| Service | `/event/card/present` | Request the left-card presentation sequence |
| Service | `/event/card/release` | Open the left hand and release presentation state |

## ROS domains

| Traffic | Domain | Selected by |
| --- | --- | --- |
| Physical application API | `42` | `scripts/start.sh`, `scripts/pc2_handshake.sh`, and `scripts/status.sh` |
| ROS simulation | `77` | `scripts/run_ros_handshake_sim.sh` |
| Native Unitree and BrainCo helpers | `0` | Isolated helper processes |

Runtime scripts select the correct domain automatically. Manual exports are
only needed when running direct `ros2` diagnostics.

For PC2 diagnostics:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/IEA_G1/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

Do not use domain `0` for application services.

## Development environment

Requirements are Linux, Git, Docker Engine, Docker Compose v2, and X11 plus
`/dev/dri` access for GUI simulation.

```bash
git clone <COMPANY_GITLAB_REPOSITORY_URL> IEA_G1
cd IEA_G1
./scripts/dev.sh
```

Open another shell in the running container with:

```bash
./scripts/dev.sh shell
```

Start or recreate it with GUI access when required:

```bash
./scripts/dev.sh --gui
```

The repository is mounted at `/workspace/IEA_G1`. Detailed setup is in
[`docs/setup/docker_setup.md`](docs/setup/docker_setup.md).

## Build and test

Inside the development container:

```bash
cd /workspace/IEA_G1/ros2_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select \
  g1_robot_interfaces \
  g1_hardware_bridge \
  g1_handshake \
  g1_card_presentation \
  g1_simulation_bridge \
  g1_bringup

source install/setup.bash

colcon test --packages-select \
  g1_robot_interfaces \
  g1_hardware_bridge \
  g1_handshake \
  g1_card_presentation \
  g1_simulation_bridge \
  g1_bringup \
  --event-handlers console_direct+

colcon test-result --verbose
```

The final test summary must contain zero errors and zero failures.

## Simulation

From `/workspace/IEA_G1` inside the container:

```bash
python3 simulation/validate_config.py
python3 simulation/run_g1_model.py --validate-only
python3 simulation/run_handshake_demo.py --validate-only
python3 simulation/run_handshake_demo.py --headless --duration 0.5
./scripts/run_ros_handshake_sim.sh --headless
```

For the native model and Unitree balance controller:

```bash
python3 simulation/run_g1_model.py --native --balance
```

The native Unitree binaries require the pinned Unitree runtime libraries in
`LD_LIBRARY_PATH`; use the repository scripts instead of invoking those
binaries directly. See [`simulation/README.md`](simulation/README.md).

## PC2 deployment and operation

The physical runtime selection is:

```text
config/udev/70-brainco-hand.rules
docs/setup/pc2_event_handshake.md
ros2_ws/src/g1_robot_interfaces/
ros2_ws/src/g1_hardware_bridge/
ros2_ws/src/g1_handshake/
ros2_ws/src/g1_bringup/
scripts/build.sh
scripts/start.sh
scripts/status.sh
scripts/pc2_handshake.sh
vendor/brainco_hand_server/
```

Build and start on PC2:

```bash
cd /home/unitree/IEA_G1
./scripts/build.sh
source /opt/ros/foxy/setup.bash
source ros2_ws/install/setup.bash
./scripts/start.sh
```

From another PC2 terminal:

```bash
cd /home/unitree/IEA_G1
./scripts/status.sh
./scripts/pc2_handshake.sh state
./scripts/pc2_handshake.sh start
./scripts/pc2_handshake.sh grip
./scripts/pc2_handshake.sh cancel
```

The full installation, udev, network, deployment, update, and shutdown
procedure is maintained in
[`docs/setup/pc2_event_handshake.md`](docs/setup/pc2_event_handshake.md).

## Manual-grip fallback and checkpoints

Automatic grip is an extension of the previously validated manual workflow.
The Git checkpoints are:

- `manual-grip-baseline`: validated manual `start` and `grip` workflow;
- `autonomous-grip-v1`: first RGB-depth automatic-grip implementation.

Use the manual `/event/handshake/grip` service if automatic detection is
disabled or unsuitable. Do not switch Git revisions on a powered robot without
first stopping the running stack and confirming that both arms and hands are in
a safe state.

## Safety constraints

- Keep people clear of both arms during startup, shutdown, and fault recovery.
- Do not operate the application and Unitree controller arm actions
  simultaneously.
- Treat controller input as external ownership; the guard blocks or aborts the
  application rather than competing for the arm.
- Keep the RealSense device owned by the camera bridge only.
- Revalidate camera regions whenever the robot head or camera mounting changes.
- Test configuration and state-machine changes in simulation before physical
  deployment.
- Keep physical left-arm card presentation disabled until separately validated.

## Repository layout

```text
config/udev/                physical-device permission rules
docker/                     reproducible Humble development image
docs/setup/                 workstation and PC2 procedures
ros2_ws/src/                ROS 2 packages
scripts/                    development, deployment, and operator commands
simulation/                 MuJoCo models and validation tools
vendor/brainco_hand_server/ physical BrainCo runtime and provenance
```

Generated `build`, `install`, `log`, caches, recordings, and machine-specific
files must not be committed.

## Contribution checklist

Before committing:

```bash
git status --short
git diff --check
```

Build and test every affected package, run the relevant simulation or physical
validation, and document new nodes, topics, services, parameters, dependencies,
and safety effects. Never commit credentials or captured personal data.

Stop the shared development container when finished:

```bash
./scripts/dev.sh stop
```
