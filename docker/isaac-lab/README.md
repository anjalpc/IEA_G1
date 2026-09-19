# Isaac Lab AWS Container

This directory defines the Isaac Lab environment used for G1 reinforcement-learning experiments.

## Validated stack

- Isaac Sim: `5.1.0`
- Isaac Lab: `v2.3.0`
- Isaac Lab commit: `3c6e67bb5c7ada942a6d1884ab69338f57596f77`
- Fixed image: `isaac-lab-base-fixed:2.3.0`
- Runtime container: `isaac-lab-fixed`

The fixed image installs the missing core `isaaclab` package and its required `flatdict==4.0.1` dependency using `--no-build-isolation`.

## Expected workspace

    <workspace>/
    ├── IEA_G1/
    ├── src/
    │   ├── IsaacLab/
    │   └── unitree_ros/
    └── integration/
        └── unitree_rl_lab/

The workspace root is derived automatically from the location of `IEA_G1`.

Optional overrides:

    IEA_WORKSPACE=/path/to/workspace
    ISAACLAB_SOURCE=/path/to/IsaacLab
    UNITREE_RL_HOST=/path/to/unitree_rl_lab
    UNITREE_ROS_HOST=/path/to/unitree_ros
    PROJECT_HOST=/path/to/IEA_G1

## Network configuration

Proxy use is optional.

Normal network:

    docker/isaac-lab/build.sh
    docker/isaac-lab/run.sh

Proxy-restricted network:

    docker/isaac-lab/build.sh --proxy
    docker/isaac-lab/run.sh --proxy

Proxy values belong in `docker/isaac-lab/.env.local`.
This file is ignored by Git. Do not commit proxy addresses or credentials.

## Build

From the repository root:

    docker/isaac-lab/build.sh

Expected image:

    isaac-lab-base-fixed:2.3.0

## Run

    docker/isaac-lab/run.sh

This recreates the `isaac-lab-fixed` container with:

    IEA_G1                -> /workspace/IEA_G1              (read-write)
    src/unitree_rl_lab     -> /workspace/unitree_rl_lab      (read-only)
    src/unitree_ros        -> /workspace/unitree_ros         (read-only)

Isaac Lab source, scripts, tools, caches, logs, and data are inherited from the validated `isaac-lab-base` container.

## Validate

    docker exec isaac-lab-fixed /workspace/isaaclab/isaaclab.sh -p -c \
      "import isaaclab; print('Isaac Lab import: PASS')"

    docker exec isaac-lab-fixed nvidia-smi \
      --query-gpu=name,memory.total \
      --format=csv,noheader

## G1 training

Use:

    simulation/isaac_lab/g1_stairs/scripts/simctl

Scientific experiment parameters belong in committed experiment YAML files, not Docker runtime configuration.
