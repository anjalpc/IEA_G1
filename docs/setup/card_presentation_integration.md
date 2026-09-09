# Left-card presentation

The left-card workflow is physically validated on the Unitree G1. PC2 runs the
card node through the shared hardware bridge with `card_presentation_mode` set
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

`move_seconds` remains in the bridge protocol for compatibility, but the live
helper uses the fixed validated phase durations.

## State and safety behavior

- Card arm commands are rejected while an external Unitree arm action owns the
  arm.
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

Stop the stack with one `Ctrl+C` in its owning terminal. Wait for ROS and
BrainCo cleanup messages before powering off.

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
