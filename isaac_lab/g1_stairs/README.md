# G1 three-step Isaac Lab preparation

This package fixes the task geometry and robot/deployment contracts before an
Isaac Lab cloud workstation is available. It does not claim that training can
run in the repository's current ROS/MuJoCo container.

The nominal course is three 0.20 m risers with 0.25 m treads, followed directly
by a 0.60 m-high continuing platform. `geometry.py` is simulator-independent;
`isaac_terrain.py` adapts it to Isaac Lab's `SubTerrainBaseCfg` API.
The initial base pose is fixed 1.0 m before the first riser, centered laterally
and facing the staircase. Start-pose randomization remains disabled until the
nominal motion is learned.

## Local checks

```bash
python3 -m unittest discover -s isaac_lab/g1_stairs/tests -v
python3 isaac_lab/g1_stairs/audit_assets.py
```

## First cloud milestones

1. Install the pinned compatible Isaac Sim, Isaac Lab, Unitree RL Lab, and RSL-RL stack below.
2. Reproduce `Unitree-G1-29dof-Velocity` unchanged.
3. Convert or compose an Isaac asset containing the G1 and both BrainCo hands.
4. Measure and replace the provisional wrist transforms in
   `simulation/model_composition.py::HAND_MOUNTS`.
5. Register `ThreeStepStairsCfg` in a copied G1 velocity environment and run a
   32-environment, one-iteration training smoke test before large-scale training.
6. Add stair-ascent motion data and tracking rewards before PPO robustness fine-tuning.

## Cloud launch

Compatibility target: Isaac Sim 5.1.0, Isaac Lab 2.3.0, Unitree RL Lab commit
`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, and RSL-RL 3.0.1 exactly.

Install this repository and the pinned Unitree RL Lab checkout into the Isaac
Lab Python environment. In Unitree RL Lab's
`unitree_rl_lab/assets/robots/unitree.py`, replace the placeholder
`UNITREE_MODEL_DIR` with the absolute path to a downloaded `unitree_model`
checkout. Then set `UNITREE_RL_LAB_ROOT`. The wrapper registers the local task
before handing control to Unitree's unmodified RSL-RL scripts:

```bash
export UNITREE_RL_LAB_ROOT=/path/to/unitree_rl_lab
python3 scripts/run_isaac_stairs.py train --headless --num_envs 32 --max_iterations 1
python3 scripts/run_isaac_stairs.py train --headless --num_envs 4096
```

`play` requires a checkpoint, so run it only after the smoke training creates
one:

```bash
python3 scripts/run_isaac_stairs.py play --num_envs 32 --checkpoint /absolute/path/to/model.pt
```

The first slice intentionally uses a deterministic 0.35 m/s forward command,
fixed start, no pushes, and no dynamics randomization. Its stop condition is a
successful 32-environment import/reset/play smoke test. Human motion tracking,
task-completion rewards, and domain randomization follow only after that proof.

The BrainCo URDFs help because they contain visual/collision meshes, inertial
data, and joint limits. The G1 source under `simulation/vendor/unitree_g1` is
MJCF, not URDF; for Isaac Lab, use Unitree's matching 29-DoF USD/URDF as the
authoritative base and validate it against this repository's joint contract.

Unitree also publishes `g1_29dof_mode_15_brainco_hand.urdf`. It is a valuable
combined-model reference, but this project currently declares mode-machine 5,
so it must not replace the deployed model until the hardware revision and mode
compatibility are confirmed. See `sources.yaml` for immutable source revisions.
BrainCo's newer multi-format repository includes ready-made Revo2 USD files,
but the inspected revision has no repository license file; it is therefore a
reference only until BrainCo clarifies redistribution terms. The already
vendored Apache-2.0 Revo2 description remains the usable project source.
