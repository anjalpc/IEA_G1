# IEA G1 system contract

This is the local quick reference for the physical robot, simulation, SDK,
BrainCo hands, and stair-climbing RL task. It is intended to eliminate routine
dependence on upstream websites. The machine-readable YAML files linked below
remain authoritative when this summary and a configuration disagree.

Last reviewed: 2026-09-13

## Robot identity

| Property | Project value |
|---|---|
| Manufacturer | Unitree |
| Family | G1 |
| Variant | G1 EDU U7 |
| Machine type (`mode_machine`) | **5** |
| Body degrees of freedom | 29 |
| Hip pitch / roll gear ratio | 14.3 / 22.5 |
| Wrist motor generation | 4010 |
| Waist | 3 active joints |
| Hand system | Dual BrainCo Revo2 Basic |
| Hand channels | 6 per hand; 12 total |
| Locomotion policy outputs | 29 body joints only |

`mode_machine` is a physical hardware identifier read from
`unitree_hg/LowState`; it is not a software-selectable locomotion mode. A
mode-15 policy or robot model must not be substituted for this mode-5 robot.
Mode 15 uses 22.5/22.5 hip gearing and newer 5010 wrist motors.

Authoritative local configuration:

- [`simulation/mujoco/config/robot.yaml`](../../simulation/mujoco/config/robot.yaml)
- [`simulation/mujoco/config/joint_mapping.yaml`](../../simulation/mujoco/config/joint_mapping.yaml)
- [`simulation/isaac_lab/g1_stairs/sources.yaml`](../../simulation/isaac_lab/g1_stairs/sources.yaml)

## Timing, frames, and policy interface

| Contract | Value |
|---|---|
| Physics rate | 200 Hz (`dt = 0.005 s`) |
| Policy/control rate | 50 Hz (`step_dt = 0.02 s`) |
| Policy input | `[1, 480]` for the pinned reference velocity policy |
| Policy output | `[1, 29]` |
| World frame | Right-handed, +Z up |
| IMU quaternion order | `wxyz` |
| Angular velocity frame | Body |
| Initial velocity command | Zero |
| Body message family | `unitree_hg` |
| Body command topic | `rt/lowcmd` |
| Body state topic | `rt/lowstate` |
| Motor mode | 1 |

The trained stair policy must retain the 29-output action order below and must
export to ONNX before MuJoCo sim-to-sim evaluation.

## Body joint order

`Policy` is the neural-network action index. `SDK` is the index in Unitree
`LowCmd.motor_cmd` and `LowState.motor_state`.

| Policy | SDK | Joint |
|---:|---:|---|
| 0 | 0 | `left_hip_pitch_joint` |
| 1 | 6 | `right_hip_pitch_joint` |
| 2 | 12 | `waist_yaw_joint` |
| 3 | 1 | `left_hip_roll_joint` |
| 4 | 7 | `right_hip_roll_joint` |
| 5 | 13 | `waist_roll_joint` |
| 6 | 2 | `left_hip_yaw_joint` |
| 7 | 8 | `right_hip_yaw_joint` |
| 8 | 14 | `waist_pitch_joint` |
| 9 | 3 | `left_knee_joint` |
| 10 | 9 | `right_knee_joint` |
| 11 | 15 | `left_shoulder_pitch_joint` |
| 12 | 22 | `right_shoulder_pitch_joint` |
| 13 | 4 | `left_ankle_pitch_joint` |
| 14 | 10 | `right_ankle_pitch_joint` |
| 15 | 16 | `left_shoulder_roll_joint` |
| 16 | 23 | `right_shoulder_roll_joint` |
| 17 | 5 | `left_ankle_roll_joint` |
| 18 | 11 | `right_ankle_roll_joint` |
| 19 | 17 | `left_shoulder_yaw_joint` |
| 20 | 24 | `right_shoulder_yaw_joint` |
| 21 | 18 | `left_elbow_joint` |
| 22 | 25 | `right_elbow_joint` |
| 23 | 19 | `left_wrist_roll_joint` |
| 24 | 26 | `right_wrist_roll_joint` |
| 25 | 20 | `left_wrist_pitch_joint` |
| 26 | 27 | `right_wrist_pitch_joint` |
| 27 | 21 | `left_wrist_yaw_joint` |
| 28 | 28 | `right_wrist_yaw_joint` |

Safety invariants: commands require an explicit motor mode; unknown names,
duplicate SDK indices, and non-finite values are rejected.

## BrainCo hands

The BrainCo hands affect whole-body mass, inertia, collision, and arm-swing
dynamics, but their 12 channels are not locomotion-policy actions. During the
initial stair task they remain in a neutral pose.

Each vendored hand declares 22 links and 0.390 kg of link mass. Five fixed tip
links per side have no separate inertial mass and must be checked after Isaac
URDF import/fixed-joint merging.

| Channel | Function | Right URDF joint | Left URDF joint | Range (rad) |
|---:|---|---|---|---:|
| 0 | Thumb flexion | `right_thumb_proximal_joint` | `left_thumb_proximal_joint` | 0–1.0472 |
| 1 | Thumb auxiliary | `right_thumb_metacarpal_joint` | `left_thumb_metacarpal_joint` | 0–1.55334 |
| 2 | Index | `right_index_proximal_joint` | `left_index_proximal_joint` | 0–1.41 |
| 3 | Middle | `right_middle_proximal_joint` | `left_middle_proximal_joint` | 0–1.41 |
| 4 | Ring | `right_ring_proximal_joint` | `left_ring_proximal_joint` | 0–1.41 |
| 5 | Pinky | `right_pinky_proximal_joint` | `left_pinky_proximal_joint` | 0–1.41 |

Hand DDS uses hardware domain 0. Commands are normalized to 0–1 at the project
boundary. Both hands are physically verified. The right hand records firmware
`1.0.16.U` and device ID 127. The left hand was validated through the guarded
physical card-presentation workflow, including gripping and releasing the
card; its firmware and device ID were not recorded and remain `null` rather
than being inferred.

The current MuJoCo attachment transforms in
[`simulation/mujoco/model_composition.py`](../../simulation/mujoco/model_composition.py) are
simulation-derived. Physical wrist-adapter translation, orientation, and mass
must be measured before the Isaac asset is considered deployment-grade.

Authoritative hand contract:
[`simulation/mujoco/config/brainco_revo2.yaml`](../../simulation/mujoco/config/brainco_revo2.yaml).

## SDK, runtime, and DDS isolation

| Component | Pinned version/revision |
|---|---|
| Unitree SDK2 | `9754cd153af3da471b0fe5f3aa535e426fb11db3` |
| Unitree SDK2 Python | `65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` |
| Unitree MuJoCo | `ae6a8403e272733e9996ef59990880330496177f` |
| Unitree RL Lab deployment subset | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` |
| MuJoCo | 3.3.6 |
| ONNX Runtime | 1.20.1 Python; pinned Unitree native runtime for deployment |
| NumPy | 1.26.4 |
| ROS development environment | ROS 2 Humble |
| Physical PC2 environment | ROS 2 Foxy |

DDS/network separation is mandatory:

| Environment | Domain | Interface |
|---|---:|---|
| Physical Unitree/BrainCo traffic | 0 | Deployment-specific physical NIC |
| Native MuJoCo simulation | 1 | Loopback (`lo`) |
| Project ROS application stack | 42 | Configured CycloneDDS interface |

Never run simulation on hardware domain 0. Never run two low-level physical
controllers simultaneously. The physical hardware bridge is the only project
component permitted to command the robot directly.

Dependency definition: [`docker/development/Dockerfile`](../../docker/development/Dockerfile).

## Model and policy assets

### Active local assets

- G1 mode-5-compatible reference: vendored Unitree 29-DoF MJCF and meshes under
  `simulation/mujoco/vendor/unitree_g1/`.
- BrainCo Revo2: vendored URDF/Xacro and meshes under
  `simulation/mujoco/vendor/brainco_revo2/`.
- Combined MuJoCo model: constructed at runtime by
  `simulation/mujoco/model_composition.py`.
- Reference balance/velocity policy: installed into the development image,
  not stored in Git.

### Isaac Lab asset decision

Use the official `g1_29dof_rev_1_0` mode-5 body and compose both BrainCo Revo2
hands. Do not use Unitree's ready-made
`g1_29dof_mode_15_brainco_hand.urdf` as the training body: its gearing and
wrist motors describe different hardware.

Exact upstream provenance and candidate paths are recorded locally in
[`simulation/isaac_lab/g1_stairs/sources.yaml`](../../simulation/isaac_lab/g1_stairs/sources.yaml).
Licenses and vendored revisions are documented in
[`simulation/mujoco/PROVENANCE.md`](../../simulation/mujoco/PROVENANCE.md).

## Stair-climbing task

| Parameter | Initial value |
|---|---:|
| Direction | Ascending, +X |
| Riser height | 0.20 m |
| Tread depth | 0.25 m |
| Steps | 3 |
| Total rise | 0.60 m |
| Stair run | 0.75 m |
| Width | 1.20 m |
| Approach length | 2.00 m |
| Upper platform length in training scene | 3.00 m |
| Required post-stair continuation | 2.00 m |

Initial fixed base pose:

- position: `(1.00, 0.00, 0.80)` m;
- roll/pitch/yaw: `(0, 0, 0)` rad;
- first-riser distance: 1.00 m;
- position/yaw randomization: disabled until nominal ascent is learned.

The initial executable training and visual-inspection task uses the exact
`0.20 m` riser height in every environment. The planned `0.05–0.20 m`
riser-height curriculum remains disabled until the nominal staircase passes
its smoke and evaluation gates. During this phase, observation corruption,
external-force resets, terrain progression, and velocity progression are also
disabled. Proprioception retains five control steps of history, while the local
terrain height scan uses only its current frame.

Training uses PPO, initialized from human stair-ascent motion tracking. The
policy observes proprioception, commands, previous actions, and a local terrain
height scan. Training-only privileged observations may not become undeclared
deployment inputs.

Phase 12 Experiment A used integrated forward +X progress and a one-time reward
for each newly reached stair level. Experiment B starts from a fresh policy and
adds only Isaac Lab's `flat_orientation_l2` term at weight `-1.0`. No inherited
flat-ground shaping, completion, safety, foot-control, constraint, power,
action-rate, or jerk rewards are active in Experiment B.

Acceptance gates:

- at least 99% completion on the exact nominal staircase;
- at least 95% completion over the approved randomized evaluation set;
- no torso, pelvis, knee, forearm, or hand contacts in nominal evaluation;
- stable forward continuation for at least 2 m on the top platform;
- ONNX export passes the 29-output contract;
- independent MuJoCo sim-to-sim validation passes before hardware work.

Task configuration:
[`simulation/isaac_lab/g1_stairs/task.yaml`](../../simulation/isaac_lab/g1_stairs/task.yaml).

Isaac training integration contract:

- task ID: `Unitree-IEA-G1-29dof-Stairs-v0` (the `Unitree` prefix is required
  by the pinned Unitree RL Lab launcher task filter);
- compatibility target: Isaac Sim 5.1.0, Isaac Lab 2.3.0, Unitree RL Lab
  revision `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, and RSL-RL 3.0.1 exactly;
- the AWS host supplies Isaac Lab source tag `v2.3.0` and immutable commit
  `3c6e67bb5c7ada942a6d1884ab69338f57596f77` to the container; canonical
  launches reject absent or mismatched provenance and never require `.git`
  metadata inside `/workspace/isaaclab`;
- Unitree RL Lab remains pristine. The project-specific G1 asset override lives
  in `simulation/isaac_lab/g1_stairs/robot_cfg.py` and loads the pinned mode-5
  `g1_29dof_rev_1_0.urdf` from the read-only Unitree ROS checkout;
- the supported infrastructure gate is the committed
  `experiments/g1_stairs/configs/004_experiment_b_smoke.yaml` run through
  `simctl`. It uses one training iteration to validate the complete runtime
  path and may complete without producing a resolved checkpoint.

## Experiment recording and independent evaluation

Version-controlled experiment definitions live under `experiments/g1_stairs/configs/`.
Generated run artifacts live under `experiments/g1_stairs/runs/` and
must not be committed to Git.

Every retained experiment run must contain:

- the recursively resolved experiment configuration;
- a run manifest recording Git, host, GPU, package, and configuration facts;
- the complete console log;
- training and reward-component metrics;
- independent per-episode evaluation outcomes;
- an evaluation summary;
- checkpoint-associated evaluation videos when recording is enabled.

The recorded-run launcher rejects dirty worktrees by default. The
`--allow-dirty` option is permitted only for local infrastructure testing and
must not be used for thesis experiment results.

Training reward and task termination are not accepted as independent evidence
of stair completion. The simulator-independent evaluator determines success
and assigns a deterministic primary failure reason.

Nominal success requires:

- completion of the approach;
- all three stair levels reached;
- entry onto the continuous upper platform;
- at least 2.00 m of forward platform continuation;
- at least 1.00 s of upright dwell;
- no earlier fall, forbidden contact, joint, torque, velocity, direction,
  slip, stall, invalid-state, or timeout failure.

The metrics contract produces:

- `training_metrics.csv`;
- `reward_components.csv`, including raw, weight, and weighted values;
- `evaluation_episodes.csv`;
- `evaluation_summary.json`, including a 95% Wilson success-rate interval.

Experiment A uses the project-owned `StairPPORunnerCfg` rather than relying on
unverified upstream defaults. Each RSL-RL iteration writes structured training,
active reward-component, constraint, motion, PPO, performance, GPU-memory, and
non-finite-state metrics into the recorded run directory. RSL-RL 3.0.1 does
not expose approximate KL or clip fraction through its runner loss dictionary;
those two diagnostics are explicitly deferred until PPO ablation work rather
than being represented as measured values.

The independent evaluator and metrics writer have simulator-independent unit
coverage. Their Isaac Lab adapter and live logging integration passed the
one-iteration AWS metrics smoke, so the nominal Experiment A launch gate is
open.

Experiment 021 is a bounded 5 cm, single-step initial-state curriculum. It
starts from the confirmed nominal G1 standing pose, applies deterministic
two-link sagittal IK only to the leading leg, places the root immediately
before the riser, zeros all velocities, and settles under the normal actuator
configuration. Training is fail-closed unless every environment physically
validates leading-foot upper-tread support, trailing-foot lower-floor support,
upright posture, finite observations, and absence of forbidden contact. Its
reward state begins at `FIRST_SUPPORT`; no first-support reward is available.
Only one-time `BASE_ABOVE`, `BOTH_FEET_SUPPORTED`, and `COMPLETED` transitions
augment the pinned locomotion-preservation rewards. Experiment 020 remains a
rejected historical run and is never used for policy initialization.

## Safety and promotion sequence

1. Validate configuration and all 29 body-joint mappings.
2. Reproduce the unmodified Unitree flat-ground policy in Isaac Lab.
3. Validate the composed mode-5 G1 plus BrainCo asset at rest.
4. Train and evaluate the stair policy in Isaac Lab.
5. Export ONNX and inspect input/output shapes and finite zero inference.
6. Run the policy in MuJoCo on DDS domain 1 with the elastic harness attached.
7. Test adverse cases and confirm fall/limit/stale-state transitions stop the controller.
8. Require explicit human review before releasing the simulated harness.
9. Design a separate, supervised physical promotion plan with an operator and emergency stop.

Simulation success is not evidence that physical execution is safe. The
physical controller must start from fresh state, preserve exclusive low-level
ownership, and transition to passive on stale state, fall, limit violation, or
non-finite command.

## Local validation commands

Run inside the project development container unless stated otherwise:

```bash
python3 simulation/mujoco/validate_config.py
python3 simulation/mujoco/run_g1_model.py --validate-only
python3 simulation/mujoco/inspect_policy.py "$UNITREE_BALANCE_POLICY" \
  --require-g1-velocity-contract --zero-inference
python3 -m unittest discover -s simulation/isaac_lab/g1_stairs/tests -v
python3 simulation/isaac_lab/g1_stairs/audit_assets.py
pytest -q
```

Native harness-assisted balance validation:

```bash
python3 simulation/mujoco/run_g1_model.py --native --balance
```

Keep the harness attached while loading or changing a controller. Harness
release remains an explicit operator decision.

## Maintenance rule

### Opt-in 043 physical single-step smoke (not hardware promotion)

`experiments/g1_stairs/configs/043a_physical_single_step_smoke.yaml` starts
from the original Unitree ONNX actor, the normal fixed lower-floor reset,
one 6 cm riser and a 25 cm first landing region. The unchanged 5.75 m tile
has a 3.5 m effective continuation platform. This diagnostic is not the
final three-step 20 cm thesis course. Historical 042 and earlier configs
are unchanged.

The command slows from 0.50 to 0.18 m/s; episode length is 20 s. Actor,
critic and action dimensions remain 485/687/29. G1 mode_machine 5, control
timing, policy/action order and PPO settings remain those of 042. BrainCo
hands are still absent from this diagnostic asset and required before
final training/deployment. No frozen-teacher loss is introduced here.

043 uses transformed ankle-roll collision sphere samples from the pinned
mode-5 URDF, vertical load relative to actual model weight, foot speed,
angular speed, orientation, tread containment, forbidden-body contact and
0.12 s support dwell. Footprint coverage is a geometric proxy, not measured
centre of pressure. First support requires the first tread rectangle;
subsequent support may move onto the continuous platform. Both feet must
establish support, not stand motionless together throughout the gait.
These thresholds are explicit initial diagnostic choices, not a claim of
validated dynamic feasibility or a guarantee of natural-looking motion.

The snapshot boundary follows [Isaac Lab v2.3.0's step order](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/envs/manager_based_rl_env.py):
termination, reward, then automatic reset. Foot poses and velocities use the
[pinned link-frame data API](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py),
not offsets mixed between the link and centre-of-mass frames.

Physical state updates once per control step through termination checks,
independently of reward weights/order. Explicit per-environment reset clears
state while retaining the last pre-autoreset snapshot for metrics. Event
payouts are 2/3/3/5 for first support/body transfer/trailing recovery/completion;
raw event terms divide by dt to cancel RewardManager's dt multiplication.
Positive preparation budgets are capped at 0.25/0.25/0.50 per episode.

The smoke is 32 environments, one iteration, seed 42. Tensor counterexample
tests and an actual Isaac smoke are mandatory before enabling longer runs.
`physical_step043_summary.yaml` records event counts and their environment-
step denominator; training metrics persist `step043_*` in `extra_json` and
TensorBoard. Zero completed episodes is not a measured zero success rate.
These are training-detector metrics, not independent proof of successful
climbing. Independent replay/video acceptance and walking-preservation
ablation remain separate work; the base config is launch-blocked.

`043b_approach_preservation_smoke.yaml` is an opt-in, one-iteration extension
of 043A. It adds a separate 480-D observation group matching the original
Unitree locomotion contract and evaluates the frozen original ONNX actor as a
training-only reference. A smooth distance gate applies an action-reference
penalty only on the far approach, fading from full strength at 0.40 m to zero
at 0.25 m before the current riser. The deployed student remains the same
single 485-input, 29-output actor. This smoke does not add early-lift, stance,
swing, placement, or support mechanisms and is not evidence of preserved gait
or learned climbing.

Update this document in the same change whenever any of these change:

- physical G1 machine type or motor hardware;
- joint order, policy shape, timing, gains, or observation contract;
- hand firmware, mass, adapter, mounting transform, or verification status;
- SDK, simulator, model, or policy revision;
- DDS domain/interface ownership;
- stair geometry, fixed start pose, curriculum, or evaluation gates.

After such a change, run `simulation/mujoco/validate_config.py`, the stair geometry
tests, and `graphify update .`.
