# Card-presentation interaction

The card service uses the G1's left arm and left BrainCo hand to hold a card, present it, detect when a visitor takes it, release it, and return the arm. It shares the same hardware bridge and supervised launch as handshake.

## Commands

| Action | ROS service | Result |
|---|---|---|
| Hold | `/event/card/hold` | Moves to the configured pickup pose and grips the card |
| Present | `/event/card/present` | Raises/presents the held card and enables handover detection |
| Cancel | `/event/card/cancel` | Stops the interaction and returns safely |

State is published on `/event/card/state` as JSON. The persistent event gateway caches this state so local clients do not need to launch a new ROS CLI process for every poll.

## Package ownership

- `g1_card_presentation/card_presentation`: state machine and service endpoints
- `g1_card_presentation/card_handover_detector`: physical handover detection
- `g1_hardware_bridge`: all actual arm and hand commands
- `g1_bringup/event_interaction.launch.py`: starts and supervises the complete interaction stack
- `scripts/event_gateway.py`: low-latency local bridge for commands and state
- `scripts/event_control.sh`: manual and compatibility fallback

Card code must not call the Unitree or BrainCo SDK directly. It publishes the validated application command contract and lets `g1_hardware_bridge` enforce ownership and limits.

## Configuration

The source configuration is:

```text
ros2_ws/src/g1_card_presentation/config/card_presentation.yaml
```

After changing it, rebuild on PC2 with `./scripts/build.sh`; the runtime reads the installed copy under `ros2_ws/install`.

Important groups include:

- hold and presentation poses;
- gripper positions and timing;
- state timeouts;
- physical handover detector thresholds;
- safe return/cancel behavior.

Change one parameter group at a time and record the previous value. Validate pose and limit changes in simulation before testing on hardware.

## State lifecycle

A normal operator sequence is:

```text
idle -> holding -> presenting -> released/returning -> idle
```

Cancel and timeout paths must also end at `idle`. Clients should render controls from the current published state, not only from the last requested command. This matters when automatic handover detection completes the interaction without another command.

If the physical action completes but the buttons remain stuck, check the state source first:

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/IEA_G1/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic echo /event/card/state std_msgs/msg/String
```

Then confirm the gateway socket exists and `/event/card/state` returns to `active: false` with the expected neutral hand/arm state.

## Operation

The supported stack start is described in [Robot deployment](robot-deployment.md). Once healthy, manual calls are:

```bash
cd /home/unitree/IEA_G1
./scripts/event_control.sh card hold
./scripts/event_control.sh card present
./scripts/event_control.sh card cancel
```

Do not start a separate card launch beside `scripts/start.sh`; the card nodes are already part of `event_interaction.launch.py`.

## Calibration checklist

Before a physical test:

1. Confirm the workspace is clear and an operator has the emergency stop.
2. Confirm the card can be held without exceeding the configured grip.
3. Test `hold`, `present`, and `cancel` without a visitor.
4. Test timeout and cancellation from every active phase.
5. Test handover detection with slow, deliberate removal.
6. Confirm the final state returns to idle.

Treat simulation as validation of logic and mappings, not proof of physical safety.
