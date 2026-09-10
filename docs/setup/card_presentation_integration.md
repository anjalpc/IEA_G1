# Left-card presentation

The left-card workflow is physically validated on the Unitree G1. PC2 runs the
card node through the shared hardware bridge with `arm_motion_mode` set
to `live`.

## Dashboard behavior

The dashboard exposes two normal controls:

| Control | Behavior |
| --- | --- |
| Hold | Grip when open; release when gripping |
| Present | Raise when down; lower when presented |

The hand and arm remain independent:

- gripping does not raise the arm;
- presenting does not change the hand;
- lowering while gripping preserves the grip;
- releasing while presented opens the hand and then retracts automatically.

The cancel service remains available for emergency cleanup and supervision but
does not need to appear as a normal dashboard button.

## ROS interfaces

```text
/event/card/hold
/event/card/present
/event/card/cancel
/event/card/state
```

The internal `/event/card/handover` service is called only by the camera
detector. It is accepted only while the hand is `GRIPPED` and the arm is
`PRESENTED`; it opens the hand, waits 1.5 seconds, and retracts the arm.

All services use `std_srvs/srv/Trigger`. State is published as JSON in a
`std_msgs/msg/String` and includes `hand_state`, `arm_state`, `state`,
`arm_ownership`, and `bridge`.

The dashboard must disable the relevant controls while a subsystem is moving,
preparing, cleaning up, unknown, or in an error state.

## Validated physical motion

### Restricted motion boundary

The behavior package owns `config/card_arm_motion.yaml`: the original rest and
Action-27 calibration, mirror signs, bilateral preparation, phase durations,
and recovery waypoint order. `motion.py` converts this recipe into a neutral
version-1 request. No card pose name or mirror calculation is sent to hardware.

The physical bridge accepts `arm_motion` with `motion` and the compatibility
field `execution_timeout_s`. It validates the full request before starting the native worker,
then the worker validates it again before DDS initialization. The initial
capability permits fourteen arm preparation positions, seven left-arm offsets,
two fixed phases (0.30/0.90 seconds), and recovery through
preparation to the captured baseline. Component amplitude bounds derive from
the existing measured recipe; these are deliberately restricted capability
bounds, not general vendor joint limits or proof that every in-envelope motion
is safe. The gate also preserves each component's validated direction, so a
sign-flipped request is rejected. Post-acquisition displacement limits require
known-good physical traces and remain outstanding. Changing the recipe still
requires physical validation.

The worker captures the actual post-acquisition baseline, resolves the supplied
waypoints, interpolates, monitors telemetry, and performs measured-state
recovery and authority release even if the application exits. The complete
recovery specification is loaded before acquisition. Cancellation before phase
two still returns directly to baseline without first extending outward.

Authority handling, control gains, safety checks, and the worker's
`PRESENTED=`/`RELEASED=` acknowledgements retain their existing behavior.
The generic capability is enabled with `arm_motion_mode`. Named `arm_present`
requests are no longer accepted; deploy the card package, interfaces, and
bridge together. The worker is `arm_motion_helper` and accepts the validated
motion JSON. Do not launch it manually on hardware for testing.

Physical trajectory duration is the sum of explicit phase durations.
`execution_timeout_s` controls only client and backend watchdog deadlines. The
simulation and physical worker both derive trajectory duration from the motion's
explicit phase durations.

Offline regression tests freeze the checkpoint recipe independently of YAML,
exercise executor recovery phases, and reject invalid requests before worker
startup. Physical validation at 0.25, 0.50, and full scale confirmed equivalent
presentation and measured-state recovery. The deployed interface now exposes
only the complete calibrated motion; validation scaling and endpoint tracing
were removed after that acceptance run.

The live helper publishes a complete Unitree HG `LowCmd` with a CRC on every
write. It keeps non-arm joints at the captured post-acquisition position and
moves both arms through the measured bilateral companion posture.

Visible presentation motion uses fixed validated phases:

```text
bilateral preparation: 0.30 seconds
left-arm extension:    0.90 seconds
```

The reverse path uses the same phases. Authority acquisition and release use
smooth ramps. Full-fraction tests confirmed smooth presentation and recovery,
stable waist position, and no balance shift or oscillation.

`trajectory_duration` is the sum of the fixed validated phase durations; it is
independent of the execution watchdog.

## State and safety behavior

- Card arm commands are rejected while an external Unitree arm action owns the
  arm.
- Arm preparation atomically acquires a bridge-client lease. Only that client
  can execute, interrupt, recover, or release the arm until cleanup completes.
- Cancellation interrupts the active backend, waits for its worker to finish,
  and attempts each required cleanup once.
- A cleanup failure publishes `UNKNOWN` instead of claiming that the hand is
  open or the arm is down.
- Releasing while presented opens the hand before automatic retraction.
- Explicit lowering never changes the hand state.
- RGB and depth handover detection is armed only while a gripped card is
  presented. Enable its physical trigger only after the left-hand region in
  `g1_card_presentation/config/card_presentation.yaml` is calibrated.

## Handover calibration

The detector reads only the bridge topics
`/g1/camera/rgb/compressed` and `/g1/camera/depth/image_raw`. Set
`show_calibration_windows: true` while keeping `enabled: false`, rebuild, and
start the normal stack. With the card gripped and presented, drag a box around
the transfer area in both OpenCV windows. The terminal prints the selected RGB
and depth coordinates once, then prints the RGB-change ratio, depth median,
near-25 depth, depth delta, and stability hits once per second. Copy the
selected coordinates and observed thresholds into `card_presentation.yaml`,
then return `show_calibration_windows` to `false`.

An approaching hand uses the fast RGB-plus-depth stability window. A hand that
is already waiting when presentation completes uses the longer absolute-depth
window, so it is not incorrectly recorded as the empty RGB baseline. Both
paths remain gated by `hand_state=GRIPPED` and `arm_state=PRESENTED`.
- Graceful stack shutdown neutralizes bridge-owned hands and returns or
  releases bridge-owned arm control before stopping the BrainCo server.
- Abrupt electrical power loss cannot execute software cleanup. Remove held
  objects and use the normal stack shutdown before powering off the robot.

## PC2 operation

Start the complete interaction stack:

```bash
cd /home/unitree/IEA_G1
./scripts/start.sh
```

Use the operator wrapper from another PC2 shell:

```bash
cd /home/unitree/IEA_G1
./scripts/event_control.sh card hold
./scripts/event_control.sh card present
./scripts/event_control.sh card cancel
./scripts/event_control.sh card state
```

Stop the stack with one `Ctrl+C` in its owning terminal, or run
`./scripts/start.sh stop` from another PC2 terminal. Wait for ROS recovery,
authority release, and BrainCo cleanup messages before powering off.

## Verification

The workstation development image uses ROS 2 Humble. PC2 uses ROS 2 Foxy and
its native CycloneDDS library.

Workstation tests run inside the development container:

```bash
cd /workspace/IEA_G1/ros2_ws
source /opt/ros/humble/setup.bash
colcon test --packages-select \
  g1_robot_interfaces \
  g1_hardware_bridge \
  g1_card_presentation
colcon test-result --verbose
```

For direct PC2 helper imports or tests, place its native CycloneDDS library
before Foxy's library:

```bash
export UNITREE_CYCLONEDDS_LIB=\
/home/unitree/cyclonedds_ws/install/cyclonedds/lib
export LD_LIBRARY_PATH=\
"$UNITREE_CYCLONEDDS_LIB:${LD_LIBRARY_PATH:-}"
```
