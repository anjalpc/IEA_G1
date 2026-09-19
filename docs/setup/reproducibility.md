# Isaac Lab stair-training reproducibility

This is the supported setup path for the Unitree G1 stair-training work.

## Quick start

```bash
./scripts/setup_workspace.sh
docker/isaac-lab/build.sh
docker/isaac-lab/run.sh
simulation/isaac_lab/g1_stairs/scripts/simctl train experiments/g1_stairs/configs/004_experiment_b_smoke.yaml
```

Check the run:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl status
```

Expected infrastructure result:

```text
Status: completed
Exit code: 0
```

## Workspace

```text
workspace/
├── IEA_G1/
└── src/
    ├── IsaacLab/
    ├── unitree_rl_lab/
    └── unitree_ros/
```

Run `./scripts/setup_workspace.sh` to create or verify the pinned dependencies.

Pinned revisions:
- Isaac Lab: `3c6e67bb5c7ada942a6d1884ab69338f57596f77`
- Unitree RL Lab: `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- Unitree ROS: `7d6075f7f58588b189b940130e3edab3c839b2df`
- Isaac Sim: 5.1.0
- RSL-RL: 3.0.1

All project-specific changes belong in `IEA_G1`. The repositories under `src/` stay pristine.

The stair task uses `simulation/isaac_lab/g1_stairs/robot_cfg.py` to load the pinned mode-5 G1 URDF from Unitree ROS.

Recorded runs live under `experiments/g1_stairs/runs/`. Each run records the exact IEA_G1 Git commit, branch, clean/dirty state, runtime versions, and GPU information.

For deeper Docker details, see `docker/isaac-lab/README.md`.
