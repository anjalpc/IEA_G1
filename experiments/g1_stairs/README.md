# G1 stair-climbing experiments

This directory contains the reproducible experiment definitions, archived
experiment evidence, and generated run data for the Unitree G1 EDU U7
stair-climbing project.

The physical robot contract remains `mode_machine: 5`. Policies and assets for
the mode-15 G1 must not be substituted.

## Directory layout

- `configs/`: version-controlled experiment definitions.
- `manifests/`: repository and dependency freeze records.
- `checkpoints/`: checkpoint documentation; checkpoint binaries are ignored.
- `archive/`: committed experiment evidence and provenance snapshots.
- `runs/`: generated recorded-run metadata and metrics; ignored by Git.
- `logs/`: generated RSL-RL training logs and checkpoints; ignored by Git.
- `outputs/`: generated videos and evaluation outputs; ignored by Git.

Executable stair-training and evaluation code lives under
`simulation/isaac_lab/g1_stairs/`:

- `evaluation/` contains outcome evaluation and metrics implementation.
- `scripts/training/` contains preparation and recorded-training launchers.
- `scripts/evaluation/` contains checkpoint evaluation commands.
- `scripts/visualization/` contains video recording tools.
- `runtime_contract.py` validates the recorded runtime dependency contract.

The `experiments/g1_stairs/` tree therefore stores experiment definitions and
evidence, while `simulation/isaac_lab/g1_stairs/` stores executable simulation,
training, evaluation, and analysis code.

## Experiment progression

The committed YAML files form a small progression of reproducible checks and experiments:

1. `001_upstream_flat_ground.yaml`
   - One-iteration infrastructure check based on the upstream G1 task.
   - Validates the Isaac Lab, Unitree RL Lab, RSL-RL, and GPU stack.

2. `002_stairs_smoke.yaml`
   - One-iteration smoke test for the project stair task.
   - Validates task registration, G1 asset loading, nominal staircase geometry, and stepping.

3. `003_nominal_reward_debug.yaml`
   - Nominal fixed-stair training configuration.
   - Uses the exact three-step course and the current progress-plus-upright reward stage.

4. `004_experiment_b_smoke.yaml`
   - One-iteration reward and infrastructure smoke derived from the nominal configuration.
   - Use this after environment or runtime changes before launching longer training.

Historical experiment evidence belongs under `archive/`. Do not rewrite archived paths or results to match later repository reorganizations.

## Nominal staircase

- Direction: positive X.
- Riser height: 0.20 m.
- Tread depth: 0.25 m.
- Number of steps: 3.
- Total rise: 0.60 m.
- Continuous upper platform after the third step.
- Required post-stair continuation: 2.00 m.
- Initial base position: `(1.00, 0.00, 0.80)` m.
- Initial position and yaw randomization: disabled.

## Running an experiment

Use `simctl` as the normal operator entry point:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl \
  train experiments/g1_stairs/configs/004_experiment_b_smoke.yaml
```

Check the selected run:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl status
```

List recorded runs:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl runs
```

Scientific parameters belong in committed experiment YAML files. Do not change a recorded experiment by casually appending different training parameters on the command line.

Canonical training records the resolved configuration, exact IEA_G1 Git commit and branch, clean/dirty state, Isaac Lab source revision, runtime package versions, and host/GPU information.

The lower-level preparation and AWS wrapper scripts under `simulation/isaac_lab/g1_stairs/scripts/training/` remain implementation details used by `simctl` and by debugging workflows.
