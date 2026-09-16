# Experiment checkpoints

This directory is reserved for local model checkpoints during development.

Checkpoint binaries are intentionally excluded from Git because they are large
generated artifacts. Every retained checkpoint must be associated with a run
directory containing:

- `resolved_config.yaml`
- `run_manifest.yaml`
- `console.log`
- training metrics
- reward-component metrics
- independent evaluation results
- evaluation summary
- evaluation videos, when enabled

Durable AWS checkpoints must be copied from instance storage to persistent EBS
and then uploaded to the approved artifact storage location before an instance
is terminated.

A checkpoint is not eligible for deployment merely because training completed.
It must pass independent Isaac evaluation, ONNX contract validation, and
MuJoCo sim-to-sim evaluation.