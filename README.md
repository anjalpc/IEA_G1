# IEA G1 interaction stack

ROS 2 packages and operator scripts for running handshake and card-presentation interactions on a Unitree G1 with BrainCo Revo2 hands.

This repository has two distinct uses:

- **Robot runtime (PC2):** ROS 2 Foxy nodes run on the robot and own the arms, hands, and cameras.
- **Workstation development:** the repository's Docker container provides ROS 2 Humble, MuJoCo, Unitree SDKs, and test tools without changing the host machine.

## Start here

The consolidated hardware, SDK, simulation, and stair-training reference is
[IEA G1 system contract](docs/reference/g1_system_contract.md).

Choose the workflow that matches your machine:

| Goal | Where it runs | Guide |
|---|---|---|
| Deploy or operate handshake/card | G1 robot (PC2) | [Robot deployment](docs/setup/robot-deployment.md) |
| Build, test, or simulate | Linux workstation | [Development container](docs/setup/development-container.md) |
| Tune card presentation | Robot and simulator | [Card presentation](docs/setup/card-presentation.md) |
| Understand or run simulation | Development container | [Simulation guide](simulation/mujoco/README.md) |

Do not run the physical robot stack from the development container. The container is for development and simulation; the Foxy runtime executes directly on PC2.

## Repository map

```text
IEA_G1/
├── config/
│   ├── cyclonedds_humble.xml       # ROS discovery for the dev container
│   └── udev/*.rules.example        # optional BrainCo permission template
├── docker/                          # container definitions
├── docs/setup/                      # focused setup and deployment guides
├── ros2_ws/src/                     # ROS 2 source packages
├── scripts/                         # supported operator entry points
├── simulation/mujoco/                      # models, mappings, validation, demos
└── vendor/brainco_hand_server/      # pinned ARM64 BrainCo runtime for PC2
```

Generated `ros2_ws/build`, `ros2_ws/install`, and `ros2_ws/log` directories are local outputs and are intentionally not committed.

## ROS 2 packages

### `g1_robot_interfaces`

Defines the internal command and feedback contract shared by the physical and simulated backends. Commands use validated JSON carried on ROS topics so the interaction state machines do not depend directly on a hardware SDK.

### `g1_hardware_bridge`

The hardware boundary for the physical robot. It is the only package that should directly command the G1 arms or BrainCo hands.

It contains:

- `arm/`: Unitree arm ownership, command validation, motion execution, state monitoring, and safe release;
- `hand/`: BrainCo hand commands and state helpers;
- `camera/`: Unitree RGB shared-memory access and RealSense depth-camera access;
- `hardware_bridge`: translates validated ROS commands into physical arm/hand operations;
- `camera_bridge`: publishes camera data used by the interaction detectors;
- `rgb_helper`: isolates access to the Unitree RGB camera;
- `config/hardware.yaml` and `config/camera.yaml`: physical limits and device settings.

Keeping hardware access here prevents handshake and card code from competing for arm ownership.

### `g1_handshake`

Implements the handshake state machine and hand-presence detector. It exposes start, grip, and cancel services and publishes handshake state.

### `g1_card_presentation`

Implements the left-arm card workflow: hold, present, detect a handover, release, and return safely. It exposes hold, present, and cancel services and publishes card/arm state.

### `g1_simulation_bridge`

Connects the same application-level command contract to simulation. It is a development package and is not required on PC2.

### `g1_bringup`

Owns the supported physical launch file, `event_interaction.launch.py`. It launches and supervises the hardware bridge, cameras, handshake nodes, card nodes, and detectors as one stack. If a required node exits, the launch shuts down rather than leaving a partially working robot stack.

## Supported scripts

| Script | Purpose | Run on |
|---|---|---|
| `scripts/build.sh` | Build the physical Foxy packages | PC2 |
| `scripts/start.sh` | Start/stop BrainCo, ROS nodes, and the persistent event gateway | PC2 |
| `scripts/status.sh` | Show processes, PID records, gateway socket, and ROS nodes | PC2 |
| `scripts/event_control.sh` | Manual/fallback handshake and card ROS service calls | PC2 |
| `scripts/event_gateway.py` | Persistent Unix-socket bridge for low-latency local commands | PC2, started automatically |
| `scripts/dev.sh` | Build and enter the development container | Workstation |
| `scripts/run_ros_handshake_sim.sh` | Run the real ROS handshake state machine against MuJoCo | Development container |

There is one supported physical launch path: `scripts/start.sh` -> `g1_bringup/event_interaction.launch.py`.

## Physical runtime lifecycle

On PC2, after completing the deployment guide:

```bash
cd /home/unitree/IEA_G1
./scripts/build.sh
./scripts/start.sh
```

In another terminal:

```bash
cd /home/unitree/IEA_G1
./scripts/status.sh
```

Stop cleanly with:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh stop
```

`start.sh` verifies the vendored BrainCo runtime, starts the hand server, checks both hands, launches all physical ROS nodes, and starts the persistent event gateway. It records process IDs in `/tmp/g1_interaction_stack` and uses ROS domain `42` by default.

Manual diagnostics:

```bash
./scripts/event_control.sh handshake start
./scripts/event_control.sh handshake grip
./scripts/event_control.sh handshake cancel
./scripts/event_control.sh card hold
./scripts/event_control.sh card present
./scripts/event_control.sh card cancel
```

## Development container

From a Linux workstation with Docker Engine and the Compose plugin:

```bash
git clone <REPOSITORY_URL> IEA_G1
cd IEA_G1
./scripts/dev.sh shell --headless
```

The first build is large because it installs ROS Humble, MuJoCo, Unitree SDK2, Unitree MuJoCo, Unitree RL Lab, ONNX Runtime, and test dependencies. Later starts reuse Docker's cache.

Inside the container:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
pytest -q
python3 simulation/mujoco/validate_config.py
```

Use `./scripts/dev.sh shell --gui` on a Linux X11 host when you need the MuJoCo viewer.

## What `simulation/mujoco/` is for

`simulation/mujoco/` is a safe development and validation environment. It lets contributors check configuration, robot/hand model composition, state-machine behavior, joint mappings, motion limits, and policy compatibility before touching hardware.

It contains:

- redistributable Unitree G1 and BrainCo Revo2 descriptions/meshes under `simulation/mujoco/vendor/`;
- configuration contracts and joint mappings under `simulation/mujoco/config/`;
- a small surrogate model for fast tests under `simulation/mujoco/models/`;
- model composition, policy inspection, validation, and demo tools;
- provenance and license information for vendored assets.

Simulation is not copied to PC2 for normal operation. It does not prove a motion is safe on the physical robot; hardware promotion still requires review and low-risk testing.

## Safety rules

- Use only `g1_hardware_bridge` to command physical arms and hands.
- Keep the configured joint, velocity, and workspace limits enabled.
- Test state-machine and mapping changes in simulation first.
- Keep an operator and emergency stop ready during physical tests.
- Stop with `scripts/start.sh stop` so the bridge can neutralize the hand and release arm authority.
- Never run two physical interaction stacks at the same time.

## Before opening a change

```bash
bash -n scripts/*.sh
python3 -m compileall -q scripts simulation ros2_ws/src
python3 simulation/mujoco/validate_config.py
```

Then run the package tests in the development container and document any physical validation separately.
