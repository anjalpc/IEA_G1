# Left-card presentation integration status

Physical card presentation remains `disabled` by default. The terminal
controller's physical results are supplied by the operator's handoff; they do
not validate this integrated helper or its new failure handling.

## Measured Action 27 choreography (2026-09-08)

The card helper now defaults to 1.2 seconds of visible motion: 0.30 seconds
from the fresh post-acquisition start pose to bilateral companion, then 0.90
seconds from companion to the existing validated left-card delta target. The
right arm holds companion throughout extension and presentation. The firmware
handshake Action 27 path is unchanged.

Full-fraction companion angles, in shoulder pitch/roll/yaw, elbow, wrist
roll/pitch/yaw order (radians):

- Left: `[0.286, 0.218, -0.009, 0.978, 0.144, 0.006, -0.004]`
- Right: `[0.286, -0.218, 0.009, 0.978, -0.144, 0.006, 0.004]`

`move_seconds` is total motion time (1.2–10 seconds), split 25%/75%; larger
values allow slower staged validation. `motion_fraction` scales both companion
displacements from captured start and the original left-card delta. The left
card target remains start-relative, not companion-relative.

Release uses fresh measured arm positions: reverse extension to companion,
then both arms home, home settling, and authority ramp-down. Release during
acquisition/preparation returns directly home without proceeding to extension.
Complete joint 0–28 initialization, safety thresholds, telemetry checks,
persistent hold, stdin RELEASE, and disabled/live gating remain in place.

Local regression evidence: 20 controller tests passed with SDK/IDL/CRC and DDS
test doubles, plus 35 backend/card/handshake sequence tests. Real SDK/CRC and
ROS integration were not tested: host dependencies are absent and container
access was declined. These results do not establish physical tracking or
stability at the faster timing. No physical commands were sent.

Before physical validation, complete the real SDK/container gates below.
With the existing live-mode and ownership gates and an operator ready to stop,
validate arm-only acquire/return, then 25%, 50%, 75%, and 100% gestures, first
at slower timing and then at 1.2 seconds. Check waist/IMU and both arms at each
stage; exercise RELEASE in preparation, extension, and hold before integrating
the hand. Prior single-arm tests do not validate this new bilateral motion.

## Bridge changes

- Complete commands initialize joints 0–28 from measured state. Motion uses
  the post-acquisition baseline and the handed-off mirrored deltas and gains.
- DDS callbacks retain advancing lowstate ticks. Samples older than 0.5 seconds
  and nonfinite measured positions/IMU values are rejected.
- RELEASE, parent stdin EOF, SIGINT, and SIGTERM request graceful release.
  Stdin commands are checked during initial state waiting, acquisition,
  movement, and holding, without blocking on partial input lines.
- Cleanup runs even after acquisition or safety errors. Retraction uses fresh
  measured state. If telemetry is stale, retraction is refused and authority
  is ramped down using the last command and current blend weight. Partial
  acquisition cleanup does not raise authority to 1.0.
- The backend continuously drains stdout/stderr, retains result markers across
  process exit, and serializes concurrent release requests. Release must be
  acknowledged and the helper must exit successfully before its ownership
  reference is cleared. Timeout does not kill or forget a live helper.
- Generic release routes an active presentation through its own helper, without
  Action 99. Bridge shutdown attempts local presentation cleanup even when the
  firmware ownership monitor is unavailable. Card acquisition checks native
  ownership before dispatch.
- Client timeouts include acquisition and failed-present cleanup, or retraction,
  settling, authority release, and process exit, as appropriate.

The helper's graceful release is a best-effort software action. Publication
failure is reported even if later zero-weight writes succeed. SIGKILL or loss
of the host process cannot execute Python cleanup. Firmware state monitoring
does not yet provide full custom Arm SDK ownership/takeover semantics.

## Verification status (2026-09-08)

Operator follow-up: SDK and helper imports passed. The focused suite reported
58 passed and 11 setup errors, all caused by missing `crc_amd64.so` when the
controller constructs `CRC()`. The pip wheel did not retain that native asset.
The Dockerfile now retains a pinned editable SDK checkout and checks an actual
HG command checksum during image build. Repairing the running container and
rerunning the tests remain pending. These errors occurred before controller
test execution; they do not establish a controller behavior failure or pass.

Before these edits, 40 targeted backend/card/handshake tests passed in the
development container. The pinned official SDK2 Python package installed
successfully in that running container. The Dockerfile now records the same
installation for future builds.

Post-edit Python syntax compilation and `git diff --check` passed. Container
regression execution was declined; the new tests have not been run. The image
has not been rebuilt, the full workspace has not been rebuilt/tested, and the
SDK/helper import-only checks remain pending. No physical commands were sent.

Run the following inside the development container when execution is enabled:

```bash
source /opt/ros/humble/setup.bash
cd /workspace/IEA_G1/ros2_ws
colcon build --symlink-install
source install/setup.bash
python3 -c "import unitree_sdk2py"
python3 -c "from g1_hardware_bridge.arm.presentation_helper import PresentationController; print('imports OK')"
python3 -c "from unitree_sdk2py.utils.crc import CRC; from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_; print('SDK CRC OK:', CRC().Crc(unitree_hg_msg_dds__LowCmd_()))"
python3 -m pytest -q \
  src/g1_hardware_bridge/test/test_presentation_helper.py \
  src/g1_hardware_bridge/test/test_interruptible_backends.py \
  src/g1_hardware_bridge/test/test_hardware_bridge_contract.py \
  src/g1_robot_interfaces/test/test_bridge_backends.py
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

The controller tests replace DDS initialization, subscribers, and publishers
with test doubles. Backend lifecycle tests use real local subprocess pipes
running synthetic protocol scripts, with no SDK or robot access.

## Remaining gates

1. Complete the offline checks above and rebuild the development image.
2. Validate arm-only bridge presentation/release on the robot with an operator.
3. Validate interruption during acquisition, outward motion, holding, and bridge
   shutdown, including the new stale-telemetry behavior.
4. Refactor the card application using the handshake worker/state/coordinator
   pattern, then integrate the hand. The existing card node still blocks its
   single-threaded ROS executor while waiting for bridge feedback; it is not
   ready for end-to-end live operation.

Retain subprocess isolation, the validated hand targets, and the existing
handshake Action 27 path. Do not launch the card application as a substitute for
the arm-only bridge validation.
