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

## Active experiment profiles

The active config directory is intentionally small and configuration-driven:

- `nominal_stairs_base.yaml`: shared reproducibility, robot, timing, action, and terrain contract.
- `specialist_base.yaml`: reusable near-riser specialist skill profile.
- `specialist_smoke.yaml`: one-iteration integration gate.

Historical experiment definitions 001 through 050 and their associated
controller/reward implementations are preserved under
`archive/pre_specialist_20260925/`. They are not imported by the active
runtime.

## Thesis target staircase

- Direction: positive X.
- Riser height: 0.20 m.
- Tread depth: 0.25 m.
- Number of steps: 3.
- Total rise: 0.60 m.
- Continuous upper platform after the third step.
- Required post-stair continuation: 2.00 m.

The specialist curriculum begins from a simpler one-step geometry and scales
toward this target. Its staging distribution is defined in the active YAML,
not hard-coded here.

## Running an experiment

Use `simctl` as the normal operator entry point:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl train \
  experiments/g1_stairs/configs/specialist_smoke.yaml
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

## Active specialist path

The active specialist stair-climbing path is configuration-driven. It does not
create experiment-number-specific Python implementations.

- `configs/specialist_base.yaml` defines the reusable specialist skill profile.
- `configs/specialist_smoke.yaml` is the one-iteration integration gate.
- `specialist_stair.enabled: true` selects the official USD/deployment-PD robot
  profile and near-riser staging distribution.
- `policy_initialization.type: random` is mandatory for this profile.
- Legacy `physical_step`, `structured_step`, teacher, and halfway-reset runtimes
  are rejected if accidentally mixed into a specialist experiment.

Historical 001-050 configs and obsolete runtime implementations are now
archived under `archive/pre_specialist_20260925/`. The active specialist path
retains all generic numerical diagnostics, replay/evaluation tooling, and
reference-motion infrastructure.
