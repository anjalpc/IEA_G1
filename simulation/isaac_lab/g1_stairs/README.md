# G1 specialist stair-climbing task

This package contains the active Isaac Lab task for the Unitree G1 EDU U7
specialist stair-climbing policy.

## Active architecture

The specialist skill is configuration-driven. Python modules implement generic
contracts and reusable mechanics; experiment YAML selects geometry, staging,
rewards, curriculum, and execution scope. Do not create experiment-number-specific
Python modules.

The active specialist foundation uses:

- 29 body-DoF joint-position-offset actions;
- the official 29-DoF Unitree USD with deployment-matched PD gains;
- random policy initialization, with no locomotion teacher or actor transfer;
- near-riser standing resets;
- explicit stair geometry in the actor observation;
- generic numerical diagnostics and recorded run provenance.

Reference-motion modules remain active because they may be reused for
reference-guided specialist training. Historical 001-050 controller/reward
implementations are preserved under
`experiments/g1_stairs/archive/pre_specialist_20260925/` and are not executable
through the active training/replay path.

## Diagnostics

The cleanup does not remove the glass-box diagnostic architecture. Active tools
include run manifests, resolved configs, reward distributions, action-group
statistics, contact/safety metrics, checkpoint evaluation, posture diagnostics,
video replay, and `simctl`.

## Tests

Run the complete active package suite inside the Isaac container:

```bash
docker exec isaac-lab-fixed bash -lc '
cd /workspace/IEA_G1
export PYTHONPATH=/workspace/IEA_G1:${PYTHONPATH}
/workspace/isaaclab/isaaclab.sh -p -m pytest -q -p no:cacheprovider \
  simulation/isaac_lab/g1_stairs/tests
'
```

## Operator entry point

Run the specialist integration gate with:

```bash
simulation/isaac_lab/g1_stairs/scripts/simctl train \
  experiments/g1_stairs/configs/specialist_smoke.yaml
```

Scientific parameters belong in committed YAML. Recorded runs capture the
resolved configuration, IEA_G1 commit, runtime dependency revisions, package
versions, and host/GPU provenance.
