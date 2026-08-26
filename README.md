# IEA G1 robot integration

ROS 2 integration and simulation tooling for a guarded Unitree G1 handshake
workflow with BrainCo Revo2 hands. The repository is self-contained except for
the upstream dependencies downloaded while building the development image.

Hardware commands are disabled in the committed configuration. Simulation or
dry-run success does not authorize operation on a physical robot.

## What is included

- `ros2_ws/src/g1_robot_bridge/`: application node, handshake sequencing,
  hardware backends, simulation client, and tests.
- `ros2_ws/src/g1_simulation_bridge/`: ROS adapter that validates application
  commands and dispatches them to the MuJoCo backend.
- `ros2_ws/src/g1_robot_bridge/launch/`: reproducible ROS startup with hardware
  and hand control disabled by default.
- `simulation/`: MuJoCo model composition, headless gates, interactive tools,
  configuration contracts, and licensed upstream robot descriptions.
- `docker/`: Ubuntu 22.04, Python 3.10, ROS 2 Humble, MuJoCo, and pinned
  Unitree dependencies.
- `scripts/`: development, simulation, and guarded deployment entry points.
- `docs/setup/`: complete workstation and deployment instructions.

Third-party sources, revisions, and licenses are documented in
[`simulation/PROVENANCE.md`](simulation/PROVENANCE.md).

## Prerequisites

For the reproducible development workflow, use a Linux host with:

- Git;
- Docker Engine with the Compose v2 plugin;
- internet access while the image downloads pinned upstream sources;
- at least 10 GB of free space for the image and build products.

An X11 display and `/dev/dri` are needed only for the interactive viewer.
Headless builds and tests do not require host ROS, Python, or graphics support.

## Reproduce a clean checkout

```bash
git clone <COMPANY_GITLAB_REPOSITORY_URL> IEA_G1
cd IEA_G1
./scripts/dev.sh
```

Inside the container, run the same core gates used by GitLab CI:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select g1_robot_bridge g1_simulation_bridge \
  --event-handlers console_direct+
colcon test-result --verbose

python3 simulation/validate_config.py
python3 simulation/run_g1_model.py --validate-only
python3 simulation/run_g1_model.py --headless --steps 1000
python3 simulation/run_handshake_demo.py --validate-only
python3 simulation/run_handshake_demo.py --headless --duration 0.05
python3 simulation/run_handshake_demo.py --headless --duration 0.5 \
  --cancel-after 0.05
```

For the interactive native viewer, leave the headless container and start a
GUI-enabled one:

```bash
./scripts/dev.sh --gui
python3 simulation/run_g1_model.py --native
```

Detailed setup, GPU checks, rebuild instructions, and troubleshooting are in
[`docs/setup/docker_setup.md`](docs/setup/docker_setup.md). Simulation controls
and safety gates are in [`simulation/README.md`](simulation/README.md).

## ROS interface

The node is `event_handshake`. Start it with safe dry-run defaults after
sourcing `ros2_ws/install/setup.bash`:

```bash
ros2 launch g1_robot_bridge event_handshake.launch.py
```

Its public application interface is:

| Type | Name | Purpose |
| --- | --- | --- |
| Service | `/event/handshake/operator_enable` | Enable or disable requests |
| Service | `/event/handshake/start` | Start one guarded handshake |
| Service | `/event/handshake/cancel` | Cancel and return to the safe state |
| Topic | `/event/handshake/state` | Publish state and errors |

The simulation adapter uses `/simulation/g1/command` and
`/simulation/g1/feedback`. It currently accepts only reviewed `arm_action` and
right-hand `hand_position` commands. See `simulation/README.md` for exact
limits and behavior.

Run the adapter directly after sourcing the workspace:

```bash
ros2 run g1_simulation_bridge simulation_bridge --headless
```

## ROS commands and shell scripts

Use ROS-native commands for ROS processes and interfaces:

- `ros2 launch g1_robot_bridge event_handshake.launch.py` starts the node with
  declared parameters and safe defaults;
- `ros2 service call` operates the enable, start, and cancel services;
- `ros2 topic echo` observes handshake state.

The `scripts/` directory is limited to tasks outside a single ROS process:

- `dev.sh` prepares Docker, host identity, and optional display/GPU access;
- `run_ros_handshake_sim.sh` supervises the application launch and the
  `g1_simulation_bridge` adapter, including cleanup;
- `pc2_event_common.sh` loads the deployment machine's ROS/DDS environment;
- `pc2_run_handshake_{dry,hardware}.sh` apply operator safeguards before the
  ROS launch command;
- `pc2_handshake.sh` is one convenience wrapper for enable, disable, start,
  cancel, and state operations;
- `test_pc2_handshake_dry.sh` is a repeatable deployment smoke test.

These scripts do not implement robot behavior. That behavior remains in ROS
nodes and the launch file; the scripts only prepare or coordinate processes.

## Safety and deployment

Changes pass through these gates in order:

1. Build and test the ROS package in the Humble development image.
2. Validate the vendored models, mappings, policy contract, and MuJoCo gates.
3. Run the ROS node in simulation and dry-run modes.
4. Review the exact commit and CI result.
5. Deploy reviewed source and configuration to the robot computer.
6. Build under the deployment ROS distribution and repeat dry-run testing.
7. Perform guarded hardware testing with an authorized operator present.

Never commit `hardware_enabled: true` or `hand_control_enabled: true`. Only one
process may own low-level robot motion. The deployment procedure and all
per-run interlocks are documented in
[`docs/setup/pc2_event_handshake.md`](docs/setup/pc2_event_handshake.md).

## Contributing

Create a short-lived branch for one change, keep generated build/output files
out of Git, run the gates above, and submit a GitLab merge request. Document
new topics, services, actions, configuration, dependencies, and safety effects
in the same change. Do not commit credentials, local environment files,
captured logs, policies, or machine-specific paths.
