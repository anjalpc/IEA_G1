# G1 three-step Isaac Lab task

This package contains the Isaac Lab stair-climbing task for the Unitree G1 EDU U7.

Current thesis contract:

- mode-machine 5
- 29 body-DoF locomotion actions
- fixed start, centered and facing the stairs
- three 0.20 m risers with 0.25 m treads
- continuous upper platform
- at least 2.0 m forward travel after the third step

Randomization is introduced only after the nominal task works.

## Dependencies

Upstream dependencies remain pristine under the workspace `src/` directory:

```text
workspace/
├── IEA_G1/
└── src/
    ├── IsaacLab/
    ├── unitree_rl_lab/
    └── unitree_ros/
```

Do not modify Unitree RL Lab or Unitree ROS for project-specific changes.

The project-specific G1 asset override is:

`simulation/isaac_lab/g1_stairs/robot_cfg.py`

It loads the pinned mode-5 URDF from Unitree ROS while preserving the upstream G1 actuator and joint configuration.

## Setup

From a fresh IEA_G1 clone:

```bash
./scripts/setup_workspace.sh
docker/isaac-lab/build.sh
docker/isaac-lab/run.sh
```

Run the committed infrastructure smoke test:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl \
  train experiments/g1_stairs/configs/004_experiment_b_smoke.yaml
```

Check it with:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl status
```

Expected result: `Status: completed` and `Exit code: 0`.
The one-iteration smoke may not produce a resolved checkpoint.

## Tests

Run the stair package tests inside the Isaac container:

```bash
docker exec isaac-lab-fixed bash -lc 'cd /workspace/IEA_G1 && /workspace/isaaclab/isaaclab.sh -p -m pytest simulation/isaac_lab/g1_stairs/tests -q'
```

Current expected result: `39 passed`.

## Experiments and provenance

Experiment definitions live under `experiments/g1_stairs/configs/`.
Generated runs, logs, checkpoints, videos, and evaluation outputs are local artifacts.

Recorded runs capture the exact IEA_G1 Git commit and branch, clean/dirty state, resolved experiment configuration, Isaac Lab revision, runtime package versions, and GPU/host information.

Training is refused when the IEA_G1 worktree is dirty.

Use committed YAML files for scientific parameters rather than overriding recorded experiment settings ad hoc on the command line.

## BrainCo hands

The locomotion policy controls the 29 G1 body joints only.
BrainCo Revo2 finger joints are not locomotion actions.

Future Isaac assets that include the physical hands must include their mass, inertia, collision geometry, and neutral pose without changing the 29-action locomotion contract.

See `docs/setup/reproducibility.md` for the complete fresh-workspace setup.
