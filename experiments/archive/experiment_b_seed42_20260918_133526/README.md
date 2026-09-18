# Experiment B

Experiment: B - progress rewards plus upright-posture penalty

Run ID:
experiment_b_seed42_20260918_133526

Training log directory:
logs/rsl_rl/unitree_iea_g1_29dof_stairs_v0/2026-09-18_13-35-34

Training:
- Seed: 42
- Environments: 128
- Iterations: 2000
- Final iteration: 1999

Reward terms:
- forward_progress: +1.0
- stair_progress: +1.0
- upright / flat_orientation_l2: -1.0

Final selected metrics:
- episode return mean: 5.3834188968
- maximum X mean: 5.8780918121 m
- maximum step reached: 3
- platform reached rate: 1.0
- joint-limit violations: 14414
- velocity-limit violations: 587
- torque-limit violations: 0
- action-rate mean: 4.0220549901
- joint-jerk mean: 19231.751953125
- mechanical-power mean: 4836.9698486328125

Interpretation:
The upright term preserved and eventually improved task progress while reducing
action rate and velocity violations relative to Experiment A. It did not produce
controlled human-like stair climbing. Final behavior remained highly dynamic and
joint-limit violations increased substantially.

Primary checkpoint:
model_1999.pt

Representative videos:
- model_b_400
- model_b_800
- model_b_1999
