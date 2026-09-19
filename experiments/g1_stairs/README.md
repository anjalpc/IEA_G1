# G1 stair-climbing experiments

This directory contains the reproducible experiment definitions and the
simulator-independent evaluation framework for the Unitree G1 EDU U7
stair-climbing project.

The physical robot contract remains `mode_machine: 5`. Policies and assets for
the mode-15 G1 must not be substituted.

## Directory layout

- `configs/`: version-controlled experiment definitions.
- `manifests/`: repository and dependency freeze records.
- `evaluation/`: independent outcome evaluation, metrics, and tests.
- `checkpoints/`: local generated model files; checkpoint binaries are ignored.
- `/runs/`: generated run directories at the repository root; ignored by Git.

The experiment framework lives outside `simulation/isaac_lab/g1_stairs/` intentionally:

- `simulation/isaac_lab/g1_stairs/` defines the Isaac Lab task and simulator adapter.
- `experiments/` defines simulator-independent experiment records, acceptance
  criteria, outcomes, and metrics.
- `scripts/` contains command-line orchestration shared by the experiment
  framework and Isaac Lab.

This separation allows the evaluator and reproducibility records to be tested
without importing Isaac Sim.

## Experiment progression

1. `001_upstream_flat_ground.yaml`
   - Reproduce the unmodified upstream G1 flat-ground task.
   - Uses 32 environments and one learning iteration.
   - Validates the installed Isaac Lab, Unitree RL Lab, RSL-RL, and GPU stack.

2. `002_stairs_smoke.yaml`
   - Load the project stair task.
   - Uses the fixed initial position, 32 environments, and one iteration.
   - Validates task registration, asset loading, staircase geometry, stepping,
     logging, and checkpoint creation.

3. `003_nominal_reward_debug.yaml`
   - First meaningful fixed-stair learning experiment.
   - Targets the exact three-step staircase.
   - Remains launch-blocked until every launch gate has been validated.

Do not bypass `launch_gates.allow_launch: false` simply to start training.

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

## Preparing a run

From the repository root:

```bash
python3 simulation/isaac_lab/g1_stairs/scripts/training/prepare_experiment.py \
  experiments/g1_stairs/configs/001_upstream_flat_ground.yaml \
  --run-id upstream_flat_ground_001
```

Preparation-only runs may omit AWS provenance because they do not launch
training. Canonical AWS launches must use the host-side wrapper so the source
checkout identity is passed into the container:

```bash
simulation/isaac_lab/g1_stairs/scripts/training/run_aws_recorded_experiment.sh \
  experiments/g1_stairs/configs/003_nominal_reward_debug.yaml \
  --run-id experiment_a_seed42
```

The wrapper reads tag and commit from
`/home/ubuntu/workspace/src/IsaacLab`. The container records those values as
source provenance and records `isaaclab`, `isaaclab-rl`, and `rsl-rl-lib`
distribution versions separately as runtime metadata.
