# G1 Stair Imitation Stack

This tree is intentionally separate from the pure-RL stair implementation.

- Pure RL: `simulation/isaac_lab/g1_stairs/` and `experiments/g1_stairs/`
- Imitation/reference-guided work: `motion_imitation/`

The imitation stack may import validated robot/terrain components from the
pure-RL stack, but the pure-RL stack must never depend on imitation code.

Current canonical reference:

`motion_imitation/outputs/g1_stair_reference_v1/g1_stair_reference_20cm_v001.json`

Current gate sequence:

1. reference export
2. reference contract preflight
3. Isaac alignment
4. physics preflight
5. reward/observation preflight with zero imitation weights
6. tiny learning test
7. short diagnostic run
8. meaningful training
9. evaluation/video/metrics
10. sim2sim and real-robot gates

`imitationctl` is the operational entry point for this stack.  It must use only
imitation-specific run/process state and must never stop or mutate the pure-RL
training process.

## Operational gates

CPU-only reference checks:

```bash
./motion_imitation/imitationctl preflight
./motion_imitation/imitationctl test
```

One-environment Isaac plant/alignment check (no PPO and no `env.step()`):

```bash
./motion_imitation/imitationctl isaac-preflight
```

Patch 002 still installs no imitation reward terms and no training command.
