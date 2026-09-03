"""Calculate left-card commands from real G1 state without publishing."""

import argparse
import json
import time

from g1_hardware_bridge.experimental.left_arm_safety import (
    ArmState,
    captured_arm_positions,
    interpolate,
    presentation_target,
    validate_command_step,
    validate_state,
)


CONTROL_DT = 0.02
REQUIRED_STABLE_SAMPLES = 25


def _vector(value, attribute, fallback=(0.0, 0.0, 0.0)):
    candidate = getattr(value, attribute, fallback)
    return tuple(float(item) for item in candidate)


def observed_state(message, received_at):
    """Convert a Unitree DDS message into the pure controller state."""
    imu = getattr(message, "imu_state", None)
    rpy = _vector(imu, "rpy") if imu is not None else (0.0, 0.0, 0.0)
    omega = (
        _vector(imu, "gyroscope") if imu is not None
        else (0.0, 0.0, 0.0)
    )
    return ArmState(
        received_at=float(received_at),
        mode_machine=int(message.mode_machine),
        positions=tuple(float(motor.q) for motor in message.motor_state),
        velocities=tuple(float(motor.dq) for motor in message.motor_state),
        rpy=rpy,
        omega=omega,
    )


def run(interface, fraction, move_seconds):
    """Calculate a real-time shadow trajectory without creating a publisher."""
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()

    stable = []
    deadline = time.monotonic() + 5.0
    while len(stable) < REQUIRED_STABLE_SAMPLES:
        if time.monotonic() >= deadline:
            raise RuntimeError("no continuous stable G1 low state received")
        message = subscriber.Read()
        if message is None:
            time.sleep(CONTROL_DT)
            continue
        now = time.monotonic()
        state = observed_state(message, now)
        validate_state(state, now)
        stable.append(state)
        time.sleep(CONTROL_DT)

    captured = captured_arm_positions(stable[-1])
    target = presentation_target(captured, fraction)
    started = time.monotonic()
    samples = 0
    maximum_step = 0.0
    previous = captured
    previous_velocity = (0.0,) * len(captured)
    maximum_velocity = 0.0
    maximum_acceleration = 0.0
    while True:
        now = time.monotonic()
        phase = min(1.0, samples * CONTROL_DT / move_seconds)
        proposed = interpolate(captured, target, phase)
        velocity = validate_command_step(
            previous, proposed, previous_velocity, CONTROL_DT)
        acceleration = tuple(
            (current - prior) / CONTROL_DT
            for current, prior in zip(velocity, previous_velocity)
        )
        maximum_velocity = max(
            maximum_velocity, max(abs(value) for value in velocity))
        maximum_acceleration = max(
            maximum_acceleration, max(abs(value) for value in acceleration))
        maximum_step = max(
            maximum_step,
            max(abs(value - initial)
                for value, initial in zip(proposed, captured)),
        )
        previous = proposed
        previous_velocity = velocity
        samples += 1
        if phase >= 1.0:
            break
        message = subscriber.Read()
        if message is None:
            raise RuntimeError("G1 low state disappeared during shadow run")
        validate_state(observed_state(message, now), now)
        sleep_until = started + samples * CONTROL_DT
        time.sleep(max(0.0, sleep_until - time.monotonic()))

    return {
        "mode": "offline_shadow",
        "published": False,
        "pose": "left_card_presentation",
        "fraction": fraction,
        "move_s": move_seconds,
        "samples": samples,
        "mode_machine": stable[-1].mode_machine,
        "captured_dual_arm": captured,
        "proposed_dual_arm": target,
        "maximum_displacement_rad": maximum_step,
        "maximum_velocity_rad_s": maximum_velocity,
        "maximum_acceleration_rad_s2": maximum_acceleration,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface")
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--move-seconds", type=float, required=True)
    parsed = parser.parse_args(args)
    if not 0.0 < parsed.fraction <= 1.0:
        parser.error("fraction must be between 0 and 1")
    if not 2.0 <= parsed.move_seconds <= 10.0:
        parser.error("move-seconds must be between 2 and 10")
    result = run(parsed.interface, parsed.fraction, parsed.move_seconds)
    print("SHADOW_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
