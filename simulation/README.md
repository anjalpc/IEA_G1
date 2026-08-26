# G1 simulation and policy inspection

This directory is the simulation-first entry point for company development.
Use it to validate robot and hand descriptions, inspect compatible policies,
load the G1 in MuJoCo, exercise the elastic safety harness, and test application
behavior before any PC2 or physical-robot deployment.

The standard workflow runs entirely inside the project Docker container. A
separate host Python virtual environment is not required.

## Contents

- `config/`: project integration contracts for the G1, BrainCo Revo2 hands,
  29-joint ordering, MuJoCo isolation, and balance promotion gates.
- `vendor/unitree_g1/`: the redistributable Unitree G1 29-DoF MJCF and meshes,
  with the upstream BSD-3-Clause license.
- `vendor/brainco_revo2/`: BrainCo Revo2 URDFs and meshes, with the upstream
  Apache-2.0/BSD license notice.
- `model_composition.py`: attaches both BrainCo hands to the G1 wrist-yaw
  links, replaces the rubber-hand placeholders, and adds hand controls.
- `run_g1_model.py`: headless validation and integrated-model export.
- `run_g1_model.py --native`: native Unitree C++ timing/viewer with the
  integrated G1 and dual-BrainCo model.
- `inspect_policy.py`: safe ONNX input/output and zero-inference inspection.
- `run_handshake_demo.py`: one application-level sequencing example.
- `models/handshake_surrogate.xml`: small CI model for the handshake example.
- `validate_config.py`: validates contracts, descriptions, mappings, licenses,
  and deployment-safe defaults.

See `PROVENANCE.md` before updating third-party robot-description files.

## Start the Docker environment

From the repository root:

```bash
./scripts/dev.sh
```

The image contains ROS 2 Humble, MuJoCo, PyYAML, NumPy, ONNX Runtime, and a
pinned Unitree G1 reference velocity/balance policy. The repository is mounted
at `/workspace/IEA_G1`.

All commands below are run inside the container.

## 1. Validate the robot configuration

```bash
python3 simulation/validate_config.py
python3 simulation/run_g1_model.py --validate-only
```

This checks:

- the G1 body-policy interface remains exactly 29 actuators;
- the combined model contains 41 actuators: 29 body, 6 right hand, and 6 left
  hand;
- policy and SDK joint orders form a valid mapping;
- both BrainCo bases are mounted on the correct G1 wrist-yaw links;
- BrainCo descriptions define six channels per hand, the configured joints
  exist, all mesh paths resolve, and five distal joints per hand mimic their
  corresponding proximal joints;
- simulation uses DDS domain `1` and loopback, never hardware domain `0`;
- the elastic harness and balance safety gate are enabled;
- required third-party license notices are present.

## 2. Load the G1 and BrainCo hands in MuJoCo

For a headless smoke test:

```bash
python3 simulation/run_g1_model.py --headless --steps 1000
python3 simulation/run_g1_model.py --headless --steps 1000 --hand-position 0.5
```

Start the GUI-enabled development container from the host:

```bash
./scripts/dev.sh --gui
```

Then run the interactive viewer inside that container:

```bash
python3 simulation/run_g1_model.py --native
```

This exports the composed model, verifies the OpenGL renderer with `glxinfo`,
then opens Unitree's pinned native C++ simulator. Physics and rendering use
separate native loops; Python is not in the interactive real-time loop.
The simulator runs in the same development container. The Python entry point
forwards `Ctrl+C` so the native viewer returns to the Docker shell.

The harness starts attached. Viewer controls are:

- `9`: attach or release the elastic harness;
- `8`: lengthen the tether by 10 cm, lowering the robot;
- `7`: shorten the tether by 10 cm, raising the robot.

The full Unitree scene is loaded, including its checkerboard floor, contact
plane, lighting, and gravity (`0 0 -9.81 m/s^2`). The native harness uses the
original Unitree parameters: it applies a 200 N/m, 100 N s/m elastic tether at
`torso_link`, toward the world anchor `(0, 0, 3)`.

The viewer replaces Unitree's fixed rubber-hand placeholders with the vendored
BrainCo Revo2 geometry. The hands are mounted at the placeholder attachment
origins, with the BrainCo base `+Z` axes aligned to the G1 wrist `+X` axes. Hand
actuators remain outside the 29-element body-policy output: indices `0..28`
are the G1 body and indices `29..40` are the right and left hand channels.
`--hand-position` accepts a normalized `0..1` target for a guarded articulation
smoke test; it never changes the 29 body controls.

The native simulator advances MuJoCo in its physics thread and renders in its
viewer thread, synchronizing simulation time to the wall clock without a
Python frame loop.

The mounting transform is a simulation reference derived from the two vendor
descriptions. Confirm it against the physical wrist adapter dimensions before
using simulation contact locations as metrology.

Keep the harness attached while loading or changing a controller. Release it
only after the controller has stabilized and the checks in
`config/balance_gate.yaml` have passed.

### Run Unitree's balancing policy

Inside the development container, start the native simulator and pinned
Unitree G1 29-DoF velocity policy together:

```bash
python3 simulation/run_g1_model.py --native --balance
```

The controller uses Unitree's deployment observation history, joint mapping,
gains, action processing, 50 Hz ONNX runner, and keyboard velocity observation.
It begins at zero velocity. Both
the simulator and controller are restricted to DDS domain `1` on loopback.
The elastic tether starts attached. Keep it attached until the robot reaches a
visibly stable standing state; releasing it with `9` remains a manual operator
decision. `Ctrl+C` terminates both controller and simulator.
The launcher also restores the terminal's canonical and echo settings if the
native C++ keyboard reader is interrupted or the simulator closes.

Keep the terminal focused and hold a key to command motion:

- `W` / `S`: forward / backward at up to 1.0 m/s;
- `A` / `D`: move left / right at up to 0.8 m/s;
- `Q` / `E`: turn left / right at up to 1.0 rad/s.

Releasing the key returns the command to zero. Begin with one axis at a time
and keep the harness attached. The MuJoCo viewer has separate focus-dependent
controls: `9` toggles the tether, while `8` and `7` lower and raise it.

The balancing controller owns the 29 body actuators only. The 12 BrainCo hand
actuators remain outside the Unitree policy action vector.

The GUI Compose override passes `/dev/dri`, the host `video` and `render` group IDs,
the X11 socket, and the current Xauthority cookie into the container. It does
not use `llvmpipe`. `glxinfo -B` must report a hardware renderer; the vendor
name depends on the workstation. Stop if it reports software rendering.
Headless validation does not require a display.

## 3. Inspect policies

The Docker image exposes Unitree's pinned reference policy at:

```text
$UNITREE_BALANCE_POLICY
```

Inspect its input/output contract and run a zero-input inference:

```bash
python3 simulation/inspect_policy.py "$UNITREE_BALANCE_POLICY" \
  --require-g1-velocity-contract --zero-inference
```

Inspect another ONNX policy by passing any path mounted inside the container:

```bash
python3 simulation/inspect_policy.py /workspace/IEA_G1/policies/candidate.onnx \
  --zero-inference
```

Inspection proves that a model loads, declares understandable tensors, and
returns finite output. It does not prove dynamic stability or authorize
closed-loop control. A policy intended for this G1 velocity contract must
accept `[1, 480]` observations and produce `[1, 29]` actions.

The pinned policy is fetched into the Docker image from Unitree's repository;
the binary is not committed to this Git repository or presented as
company-authored work.

## 4. Test application behavior

The handshake is an example of an application using the shared simulation
pipeline, not the purpose of the whole directory:

```bash
python3 simulation/run_handshake_demo.py --duration 5.0
python3 simulation/run_handshake_demo.py --headless --duration 0.25
python3 simulation/run_handshake_demo.py --headless --duration 0.5 \
  --cancel-after 0.05
```

By default this demo uses the integrated G1 with both BrainCo hands. Simulated
action `27` presents the real right-arm joint chain and closes all six right
BrainCo channels. During the configured hold interval, the elbow, shoulder,
and wrist follow a bounded `1.4 Hz` handshake cycle. Cleanup smoothly opens
that hand before simulated action `99` returns the arm. The unused left hand is
actively held open, and the elastic harness remains attached throughout.

Perfect the hand profile without the approximate arm action first:

```bash
python3 simulation/run_handshake_demo.py --hand-only --curl-amount 0.12 \
  --duration 2.0
```

This uses the same `plan_curl()` channel ratios as the hardware helper and the
same 1.5-second smooth close/open duration. `--curl-amount` accepts `0..0.25`
in simulation; the guarded hardware command remains independently limited to
`0.15`.

To tune the six channels independently in simulation, pass them in manufacturer
order (`thumb thumb_aux index middle ring pinky`):

```bash
python3 simulation/run_handshake_demo.py --hand-only --duration 2.0 \
  --curl-target 0.048 0.030 0.120 0.120 0.120 0.120
```

An explicit simulation target never changes the guarded hardware limits or
configuration. Promote a tuned profile only after review.

The small generic fixture is never selected implicitly. It can be requested
only for lightweight debugging with `--surrogate`.

`Ctrl+C` requests cancellation, opens the hand, returns the arm, closes the
viewer, and exits with status `130`.

Additional applications should add their own scenario runner and tests while
reusing the robot, hand, mapping, policy-inspection, and safety contracts.

### Run the actual ROS 2 handshake through MuJoCo

The application demo above directly shares the sequencing module. The generic
ROS bridge test goes further: it runs the real `HandshakeNode`, calls its real
ROS services, sends acknowledged ROS commands to MuJoCo, and publishes the
normal handshake state topic. Hardware remains disabled.

Inside the development container, build and source the workspace:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd ..
```

Then run the complete example:

```bash
./scripts/run_ros_handshake_sim.sh
```

For an integration run without the viewer:

```bash
./scripts/run_ros_handshake_sim.sh --headless
```

The runner starts the node in explicit simulation mode, enables it through
`/event/handshake/operator_enable`, triggers `/event/handshake/start`, and
keeps the integrated G1 + BrainCo viewer open. `Ctrl+C` stops the node and
bridge. If the bridge does not acknowledge a command, the ROS node enters its
existing error and cleanup path rather than pretending the action succeeded.

The reusable client is implemented in `g1_robot_bridge/simulation_client.py`;
the adapter node is the separate `g1_simulation_bridge` ROS package.
Applications communicate through
`/simulation/g1/command` and `/simulation/g1/feedback`. The bridge currently
implements the `arm_action` and `hand_position` interfaces. Future packages can
use the same acknowledged transport, but every new capability—velocity,
sensors, navigation state, or another actuator—must add and test a named bridge
handler rather than sending arbitrary unvalidated commands.

The adapter loads the project MuJoCo backend from `simulation/` while keeping
application packages isolated from simulator implementation details. The
current arm handler intentionally supports only actions `27` and `99`, and
the hand handler supports six-channel right-hand positions. This is an
application integration test; the visible
action-27 arm trajectory is still the reviewed MuJoCo approximation because
Unitree does not expose the firmware trajectory implementation as simulation
joint targets.

## 5. Promotion toward hardware

Before moving a change to PC2:

1. Validate all configuration and description files.
2. Inspect the candidate policy and record its exact checksum and contract.
3. Run headless simulation tests.
4. Run the interactive G1 simulation with the harness attached.
5. Test cancellation, stale-state handling, limits, and passive fallback.
6. Review the exact deployment diff.
7. Copy only reviewed source/configuration to PC2 and build under Foxy.
8. Perform guarded hardware tests with an operator present.

Simulation success never authorizes physical robot commands.
