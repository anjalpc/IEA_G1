# IEA G1 ROS 2 development workspace

This repository is the shared ROS 2 workspace for developing applications,
hardware integrations, reusable interfaces, and simulation support for the
Unitree G1 robot.

Applications are separated from physical hardware drivers so that new robot
behaviours can reuse the same interfaces and can be tested against an
alternative backend before physical deployment. The handshake is one current
application in this workspace; additional applications can be added as
separate ROS 2 packages.

## Supported environments

| Environment | Platform | ROS 2 | Purpose |
| --- | --- | --- | --- |
| Development container | Ubuntu 22.04 x86_64 | Humble | Builds, tests and MuJoCo development |
| Unitree PC2 | Ubuntu 20.04 aarch64 | Foxy | Selected physical-robot packages |

Shared packages are tested on both Humble and Foxy in `.gitlab-ci.yml`.
Hardware-only dependencies are isolated from application packages.
Development uses ROS 2 Humble. Selected physical packages run on ROS 2 Foxy on
PC2.

## ROS 2 package architecture

The source workspace is under `ros2_ws/src`:

```text
ros2_ws/src/
├── g1_robot_interfaces/
├── g1_handshake/
├── g1_card_presentation/
├── g1_hardware_bridge/
└── g1_simulation_bridge/
```

### `g1_robot_interfaces`

Backend-neutral, acknowledged command transport shared by applications and
bridges. It defines correlated command/reply behaviour plus shared command
validation without importing physical or simulated implementations.

New application packages should depend on this interface layer rather than
importing Unitree or BrainCo drivers directly.

### `g1_handshake`

Application-level state machine for the current two-button handshake workflow.
It owns the sequence, public ROS services, state publication, cancellation,
timeouts, and application-level runtime configuration:

```text
ros2_ws/src/g1_handshake/config/handshake.yaml
```

It communicates only through `g1_robot_interfaces` and contains no physical
driver implementation.

### `g1_card_presentation`

Separate application for the event's verified left-hand card grip and guarded
left-arm presentation request. It does not change or reuse the right-hand
action-27 handshake profile. Development defaults to the simulation bridge;
the physical backend is disabled unless an explicit shadow-mode configuration
is selected.

### `g1_hardware_bridge`

Physical backend for PC2. It is the only package that owns Unitree arm actions
and BrainCo hand commands. Hardware allowlists, physical neutral calibration,
safe-shutdown ramp, and network values are owned by
`g1_hardware_bridge/config/hardware.yaml`. It contains no handshake grip target
or handshake timing. Native SDK operations run in isolated helper processes on
DDS domain `0`, while the ROS-facing bridge remains on the application domain.

### `g1_simulation_bridge`

Development backend that adapts the same generic arm and hand operations to
the MuJoCo implementation. Model-specific neutral targets are owned by
`g1_simulation_bridge/config/simulation.yaml`. It contains no handshake grip
profile and is not selected by the PC2 production build.

## Repository layout

```text
config/udev/                restricted physical-device permission rules
docker/                     reproducible ROS Humble development image
docs/setup/                 workstation and PC2 setup instructions
launch/                     physical deployment launch files
ros2_ws/src/                ROS 2 source packages
scripts/                    development, validation and robot commands
simulation/                 MuJoCo models, configuration and validation
vendor/brainco_hand_server/ physical BrainCo runtime and provenance
```

Generated `build`, `install`, `log`, cache, local output and environment files
are excluded by `.gitignore` and must not be committed.

## Reproduce the development environment

Requirements on the workstation:

- Linux;
- Git;
- Docker Engine;
- Docker Compose v2;
- internet access while pinned upstream dependencies are downloaded.

Clone the complete repository:

```bash
git clone <COMPANY_GITLAB_REPOSITORY_URL> IEA_G1
cd IEA_G1
./scripts/dev.sh
```

`dev.sh` starts one shared development container and opens a shell in it.
Additional terminals enter the same container with the same command. The
checkout is mounted at `/workspace/IEA_G1`.

Inside the container, confirm the environment:

```bash
echo "$ROS_DISTRO"
python3 --version
pwd
```

Expected values are ROS 2 `humble`, Python `3.10.x`, and
`/workspace/IEA_G1`.

Detailed Docker, GPU and X11 instructions are in
[`docs/setup/docker_setup.md`](docs/setup/docker_setup.md).

## Build and test the ROS workspace

Inside the development container:

```bash
cd /workspace/IEA_G1/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select \
  g1_robot_interfaces \
  g1_handshake \
  g1_card_presentation \
  g1_hardware_bridge \
  g1_simulation_bridge \
  --event-handlers console_direct+
colcon test-result --verbose
```

A reproducible change must build successfully and report no test failures.

## Run validation and simulation

From `/workspace/IEA_G1` inside the container:

```bash
python3 simulation/validate_config.py
python3 simulation/run_g1_model.py --validate-only
python3 simulation/run_g1_model.py --headless --steps 1000
python3 simulation/run_handshake_demo.py --validate-only
python3 simulation/run_handshake_demo.py --headless --duration 0.05
python3 simulation/run_handshake_demo.py --headless \
  --duration 0.5 --cancel-after 0.05
python3 simulation/run_g1_model.py --native --balance --card-presentation
```

Run the ROS-integrated handshake backend with:

```bash
./scripts/run_ros_handshake_sim.sh --headless
```

Interactive rendering requires an active X11 session and `/dev/dri`:

```bash
./scripts/dev.sh --gui
python3 simulation/run_g1_model.py --native
```

Simulation details are documented in
[`simulation/README.md`](simulation/README.md).

## ROS communication model

Applications communicate with an explicitly named backend through acknowledged
bridge channels. Current bridge topics are:

```text
/g1/bridge/hardware/command
/g1/bridge/hardware/feedback
/g1/bridge/simulation/command
/g1/bridge/simulation/feedback
```

Every command includes a client identifier and command identifier. A response
is accepted only by the requesting client and only while that command is
pending.

The current handshake application exposes:

| Type | Name | Purpose |
| --- | --- | --- |
| Service | `/event/handshake/start` | Toggle arm presentation with automatic timeout |
| Service | `/event/handshake/grip` | Execute an independent timed grip sequence |
| Service | `/event/handshake/cancel` | Cancel and request safe release |
| Topic | `/event/handshake/state` | Publish application state and bridge |

The independent left-card application exposes:

| Type | Name | Purpose |
| --- | --- | --- |
| Service | `/event/card/present` | Grip and simulate/request the left-card pose |
| Service | `/event/card/release` | Open the left hand and release presentation state |

Future applications should expose their own application-level APIs while
reusing the shared backend boundary where appropriate.

## Adding another ROS 2 application

Create a separate package under `ros2_ws/src` and keep responsibilities clear:

1. application sequencing belongs in the application package;
2. reusable request/reply transport belongs in `g1_robot_interfaces`;
3. physical SDK imports belong in `g1_hardware_bridge` or another dedicated
   hardware package;
4. simulated implementation belongs in a simulation backend;
5. runtime values belong in one installed package configuration;
6. tests must not require connected physical hardware;
7. Foxy compatibility is required for any package deployed to PC2.

Document new packages, nodes, topics, services, parameters, dependencies, and
safety effects in the same change.

## PC2 deployment model

PC2 uses the same Git repository but checks out only production files through
Git sparse checkout. Development-only directories do not appear in the PC2
working tree.

The current production selection is:

```text
docs/setup/pc2_event_handshake.md
config/udev/70-brainco-hand.rules
launch/hardware_handshake.launch.py
ros2_ws/src/g1_robot_interfaces/
ros2_ws/src/g1_handshake/
ros2_ws/src/g1_hardware_bridge/
scripts/build.sh
scripts/start.sh
scripts/status.sh
scripts/pc2_handshake.sh
vendor/brainco_hand_server/
```

The complete Foxy prerequisites, sparse-clone commands, domain setup, build,
operation, shutdown and update procedure are maintained in
[`docs/setup/pc2_event_handshake.md`](docs/setup/pc2_event_handshake.md).

## ROS domains

The development environment and PC2 application API use ROS domain `42`.
Native Unitree arm and BrainCo hand helper processes use domain `0` internally.

On PC2, operator terminals use:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/IEA_G1/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

Do not use domain `0` for application service calls.

## BrainCo runtime

`vendor/brainco_hand_server` records the physical hand runtime and its
provenance:

```text
bin/brainco_hand_server
lib/aarch64/libbc_stark_sdk.so
patches/preserve-startup-and-neutral-shutdown.patch
SOURCE_COMMIT
SOURCE_REPOSITORY
SHA256SUMS
```

The server executable is the validated patched aarch64 runtime. The BrainCo
SDK `.so` is an unmodified proprietary library. Confirm company policy before
publishing these binaries; use Git LFS or an approved internal artifact
registry when required.

## Versioning and contribution

Before creating a commit:

```bash
git status --short
git diff --check
```

Build and test the affected packages, run the relevant validation gates, and
commit one coherent reviewed change. Do not commit credentials, captured logs,
generated workspace outputs, local environment files, or machine-specific
paths.

After work is complete, stop the shared development container from the host:

```bash
./scripts/dev.sh stop
```
