# Simulation provenance

The files in this directory were authored specifically for the IEA G1 company
simulation gate:

- `run_handshake_demo.py`: project-authored behavior harness;
- `models/handshake_surrogate.xml`: project-authored generic geometry;
- `README.md`: project-authored operating and safety instructions;
- `requirements.txt`: dependency declaration only;
- `config/`: project-authored robot, hand, joint, simulator, and safety contracts;
- `validate_config.py`: project-authored contract and provenance validator;
- `run_g1_model.py`: project-authored headless validator, model exporter, and
  Python fallback harness;
- `inspect_policy.py`: project-authored ONNX contract inspector.
- `../ros2_ws/src/g1_simulation_bridge/`: project-authored ROS adapter for the
  simulation backend.

Project-authored files are kept separate from redistributable robot-description
assets under `vendor/`. No external controller source, checkpoint, dataset, or
generated experiment artifact is presented as company-authored work.

`vendor/unitree_g1/` contains only the G1 29-DoF model, required meshes, joint
reference, and scene from `unitreerobotics/unitree_mujoco` revision
`ae6a8403e272733e9996ef59990880330496177f`. Its BSD-3-Clause license is retained
in that directory.

`vendor/brainco_revo2/` contains the Revo2 URDFs, meshes, ROS package metadata,
and build metadata from `BrainCoTech/revo2_description` revision
`92cc697c7fa691db59404ce52344f3969a5ef7a6`. Its Apache-2.0/BSD license text is
retained in that directory.

MuJoCo, NumPy, PyYAML, and ONNX Runtime are installed in Docker and are not
vendored here. The generic surrogate remains an application test fixture and
must not be presented as an accurate G1 model.

The Docker image fetches Unitree's reference G1 velocity policy from
`unitreerobotics/unitree_rl_lab` revision
`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`. The policy stays in the image and
is not committed to Git or presented as company-authored work.

The Docker image also fetches and builds `unitree_sdk2` revision
`9754cd153af3da471b0fe5f3aa535e426fb11db3`, `unitree_mujoco` revision
`ae6a8403e272733e9996ef59990880330496177f`, and MuJoCo `3.3.6`. Those sources
and binaries remain image dependencies; they are not copied into this Git
repository or presented as company-authored work.

The image builds Unitree's G1 29-DoF deployment controller from the pinned
`unitree_rl_lab` revision. Its image-only simulation adaptations are limited
to DDS domain `1`, the installed SDK/CycloneDDS include paths, automatic
zero-velocity `Velocity` startup, and removal of hardware/gamepad-only FSM
transitions. The policy, observation history, action processing, gains, joint
mapping, ONNX runtime, and controller implementation remain Unitree's pinned
upstream artifacts and are not committed as company-authored source.
