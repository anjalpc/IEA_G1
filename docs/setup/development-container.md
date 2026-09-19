# Development container

This guide creates the workstation environment used to build, test, inspect policies, and run MuJoCo. It does not replace the ROS 2 Foxy installation on the robot.

## Requirements

- Linux host with Docker Engine and Docker Compose v2
- Git
- About 20 GB of free disk space for the first image build
- X11 and `/dev/dri` only when using the graphical MuJoCo viewer

## Clone and enter the container

```bash
git clone <REPOSITORY_URL> IEA_G1
cd IEA_G1
./scripts/dev.sh shell --headless
```

The first run downloads and builds many dependencies. This is expected: the image pins ROS 2 Humble, MuJoCo, Unitree SDK2 (C++ and Python), Unitree MuJoCo, Unitree RL Lab, ONNX Runtime, and Python test tools. Docker caches completed layers for later runs.

The repository is mounted at `/workspace/IEA_G1`, so source edits on the host are immediately visible in the container. ROS build outputs remain local and are ignored by Git.

## Build and test

Run these commands inside the container:

```bash
cd /workspace/IEA_G1
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
pytest -q
python3 simulation/mujoco/validate_config.py
```

To build selected packages:

```bash
cd ros2_ws
colcon build --symlink-install --packages-select \
  g1_robot_interfaces \
  g1_simulation_bridge \
  g1_handshake \
  g1_card_presentation
cd ..
```

## Start, re-enter, and stop

Start a background container:

```bash
./scripts/dev.sh start --headless
```

Open another shell without rebuilding:

```bash
./scripts/dev.sh shell --headless --no-build
```

Stop it:

```bash
./scripts/dev.sh stop
```

Rebuild after changing `docker/development/Dockerfile` or its overlay files:

```bash
./scripts/dev.sh shell --headless
```

## Graphical MuJoCo

On a Linux X11 host:

```bash
./scripts/dev.sh shell --gui
```

The GUI mode passes X11 authentication and GPU devices through to the container. If the host has no active `DISPLAY`, use `--headless` and follow [the simulation guide](../../simulation/mujoco/README.md) for non-visual checks.

## What the image contains

`docker/development/Dockerfile` is the reproducible dependency definition. `docker/development/docker-compose.yml` supplies host networking and mounts the checkout. `docker/development/docker-compose.gui.yml` adds X11 and graphics devices. `scripts/dev.sh` is the supported wrapper around those Compose files.

The image is intentionally larger than the codebase because it includes compilers, simulators, robot SDKs, policy runtime libraries, ROS, and test tooling. None of these image layers are copied into Git.

## Troubleshooting

Check Docker and Compose:

```bash
docker version
docker compose version
```

Inspect the container:

```bash
docker compose -f docker/development/docker-compose.yml ps
docker compose -f docker/development/docker-compose.yml logs --tail=100
```

If a cached image is genuinely stale, rebuild explicitly:

```bash
docker compose -f docker/development/docker-compose.yml build --no-cache iea_g1_dev
```

Avoid `--no-cache` during normal development because it redownloads every dependency.


## Proxy configuration

Proxy use is optional.

Normal network:

    ./scripts/dev.sh build

Proxy-restricted network:

    ./scripts/dev.sh build --proxy

Store machine-local proxy values in:

    docker/development/.env.local

The file is ignored by Git. Without `--proxy`, proxy variables are cleared.

Start an already-built container with:

    ./scripts/dev.sh start --headless --no-build
