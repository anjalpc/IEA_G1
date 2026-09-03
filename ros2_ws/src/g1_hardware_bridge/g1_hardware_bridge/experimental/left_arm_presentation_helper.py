"""Guarded physical 10-percent left-arm presentation trial."""

import argparse
import json
import math
import time

from g1_hardware_bridge.experimental.left_arm_safety import (
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    PresentationLimits,
    captured_arm_positions,
    presentation_target,
    validate_state,
)
from g1_hardware_bridge.experimental.arm_shadow_helper import observed_state


CONFIRMATION = "CONFIRM_LEFT_ARM_100_PERCENT"

CONTROL_DT = 0.02
STABLE_SAMPLES = 25

WAIST_INDICES = (12, 13, 14)
UPPER_BODY_INDICES = WAIST_INDICES + LEFT_ARM_INDICES + RIGHT_ARM_INDICES
WEIGHT_INDEX = 29

KP = 60.0
KD = 1.5

OWNERSHIP_RAMP_SECONDS = 2.0
RELEASE_RAMP_SECONDS = 3.0

MAX_TRACKING_ERROR_RAD = 0.20
MAX_WAIST_DEVIATION_RAD = 0.10
MAX_RIGHT_ARM_DEVIATION_RAD = 0.12


def smoothstep5(value):
    """Quintic interpolation with zero endpoint velocity and acceleration."""
    value = max(0.0, min(1.0, float(value)))
    return value ** 3 * (10.0 + value * (-15.0 + 6.0 * value))


class FreshStateReader:
    """Read advancing low-state samples and reject a stalled DDS stream."""

    def __init__(self, subscriber):
        self.subscriber = subscriber
        self.last_tick = None

    def read(self, timeout=0.10):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            message = self.subscriber.Read()
            if message is None:
                time.sleep(0.002)
                continue

            tick = int(message.tick)
            if self.last_tick is None or tick != self.last_tick:
                self.last_tick = tick
                return message

            time.sleep(0.002)

        raise RuntimeError("rt/lowstate stopped advancing")


def validate_motion_state(state, expected_mode):
    """Validate live balance state without requiring stationary arms."""
    limits = PresentationLimits()

    if state.mode_machine != expected_mode:
        raise RuntimeError(
            "mode_machine changed from {} to {}".format(
                expected_mode, state.mode_machine
            )
        )

    if len(state.positions) <= max(RIGHT_ARM_INDICES):
        raise RuntimeError("low state does not contain 29 body joints")

    values = (
        tuple(state.positions[index] for index in UPPER_BODY_INDICES)
        + tuple(state.velocities[index] for index in UPPER_BODY_INDICES)
        + tuple(state.rpy[:2])
        + tuple(state.omega)
    )

    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("low state contains a non-finite value")

    if max(abs(value) for value in state.rpy[:2]) > limits.max_roll_pitch_rad:
        raise RuntimeError("torso roll or pitch exceeded 5 degrees")

    if max(abs(value) for value in state.omega) > \
            limits.max_angular_velocity_rad_s:
        raise RuntimeError("torso angular velocity exceeded safety limit")


def fill_upper_body_command(low_cmd, desired, weight):
    """Populate every waist and arm field before publishing."""
    if len(desired) != len(UPPER_BODY_INDICES):
        raise ValueError("expected 17 upper-body targets")

    for joint, position in zip(UPPER_BODY_INDICES, desired):
        command = low_cmd.motor_cmd[joint]
        command.tau = 0.0
        command.q = float(position)
        command.dq = 0.0
        command.kp = KP
        command.kd = KD

    low_cmd.motor_cmd[WEIGHT_INDEX].q = float(weight)


def publish_command(publisher, low_cmd, crc, desired, weight):
    fill_upper_body_command(low_cmd, desired, weight)
    low_cmd.crc = crc.Crc(low_cmd)
    publisher.Write(low_cmd)


def check_tracking(state, desired, captured_upper):
    """Abort on excessive tracking, waist, or right-arm deviation."""
    actual = tuple(state.positions[index] for index in UPPER_BODY_INDICES)

    tracking_error = max(
        abs(measured - commanded)
        for measured, commanded in zip(actual, desired)
    )
    if tracking_error > MAX_TRACKING_ERROR_RAD:
        raise RuntimeError(
            "upper-body tracking error {:.4f} rad exceeds {:.4f}".format(
                tracking_error, MAX_TRACKING_ERROR_RAD
            )
        )

    waist_error = max(
        abs(state.positions[index] - captured_upper[offset])
        for offset, index in enumerate(WAIST_INDICES)
    )
    if waist_error > MAX_WAIST_DEVIATION_RAD:
        raise RuntimeError(
            "waist deviation {:.4f} rad exceeds {:.4f}".format(
                waist_error, MAX_WAIST_DEVIATION_RAD
            )
        )

    right_offset = len(WAIST_INDICES) + len(LEFT_ARM_INDICES)
    right_error = max(
        abs(state.positions[index] - captured_upper[right_offset + offset])
        for offset, index in enumerate(RIGHT_ARM_INDICES)
    )
    if right_error > MAX_RIGHT_ARM_DEVIATION_RAD:
        raise RuntimeError(
            "right-arm deviation {:.4f} rad exceeds {:.4f}".format(
                right_error, MAX_RIGHT_ARM_DEVIATION_RAD
            )
        )

    return tracking_error, waist_error, right_error


def run_phase(
        reader,
        publisher,
        low_cmd,
        crc,
        start,
        target,
        duration,
        expected_mode,
        captured_upper,
        start_weight=1.0,
        target_weight=1.0,
        check_errors=True):
    """Execute one deterministic 50 Hz quintic phase."""
    sample_count = int(round(duration / CONTROL_DT))
    maxima = {
        "tracking_error_rad": 0.0,
        "waist_deviation_rad": 0.0,
        "right_arm_deviation_rad": 0.0,
    }

    phase_started = time.monotonic()

    for sample in range(sample_count + 1):
        message = reader.read()
        now = time.monotonic()
        state = observed_state(message, now)
        validate_motion_state(state, expected_mode)

        fraction = sample / float(sample_count)
        blend = smoothstep5(fraction)

        desired = tuple(
            initial + blend * (final - initial)
            for initial, final in zip(start, target)
        )
        weight = start_weight + blend * (target_weight - start_weight)

        if check_errors:
            tracking, waist, right = check_tracking(
                state, desired, captured_upper
            )
            maxima["tracking_error_rad"] = max(
                maxima["tracking_error_rad"], tracking
            )
            maxima["waist_deviation_rad"] = max(
                maxima["waist_deviation_rad"], waist
            )
            maxima["right_arm_deviation_rad"] = max(
                maxima["right_arm_deviation_rad"], right
            )

        publish_command(publisher, low_cmd, crc, desired, weight)

        deadline = phase_started + (sample + 1) * CONTROL_DT
        time.sleep(max(0.0, deadline - time.monotonic()))

    return maxima


def safe_release(
        reader, publisher, low_cmd, crc, weight, fixed_target=None):
    """Track the measured upper body while relinquishing SDK ownership."""
    samples = int(round(RELEASE_RAMP_SECONDS / CONTROL_DT))
    started = time.monotonic()
    last_measured = None

    for sample in range(samples + 1):
        try:
            message = reader.read(timeout=0.05)
            measured = tuple(
                float(message.motor_state[index].q)
                for index in UPPER_BODY_INDICES
            )
            if fixed_target is None:
                last_measured = measured
            else:
                last_measured = fixed_target
        except Exception:
            if last_measured is None:
                raise

        fraction = sample / float(samples)
        release_weight = weight * (1.0 - smoothstep5(fraction))
        publish_command(
            publisher, low_cmd, crc, last_measured, release_weight
        )

        deadline = started + (sample + 1) * CONTROL_DT
        time.sleep(max(0.0, deadline - time.monotonic()))

    publish_command(publisher, low_cmd, crc, last_measured, 0.0)


def run(interface, fraction, move_seconds, hold_seconds, confirmation):
    if confirmation != CONFIRMATION:
        raise ValueError(
            "physical control requires --confirm {}".format(CONFIRMATION)
        )
    if not 0.0 < fraction <= 1.00:
        raise ValueError("physical trial is limited to 100 percent")
    if not 5.0 <= move_seconds <= 10.0:
        raise ValueError("move duration must be between 5 and 10 seconds")
    if not 0.0 <= hold_seconds <= 2.0:
        raise ValueError("hold duration must be between 0 and 2 seconds")

    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, interface)

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()
    reader = FreshStateReader(subscriber)

    stable_states = []
    for _ in range(STABLE_SAMPLES):
        message = reader.read()
        now = time.monotonic()
        state = observed_state(message, now)
        validate_state(state, now)
        stable_states.append(state)
        time.sleep(CONTROL_DT)

    captured_state = stable_states[-1]
    if captured_state.mode_machine != 5:
        raise RuntimeError(
            "expected stable mode_machine 5, received {}".format(
                captured_state.mode_machine
            )
        )

    captured_arms = captured_arm_positions(captured_state)
    target_arms = presentation_target(captured_arms, fraction)

    captured_upper = tuple(
        captured_state.positions[index] for index in WAIST_INDICES
    ) + captured_arms

    target_upper = (
        captured_upper[:len(WAIST_INDICES)]
        + target_arms
    )

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    ownership_weight = 0.0
    phase_metrics = {}

    try:
        phase_metrics["ownership"] = run_phase(
            reader,
            publisher,
            low_cmd,
            crc,
            captured_upper,
            captured_upper,
            OWNERSHIP_RAMP_SECONDS,
            captured_state.mode_machine,
            captured_upper,
            start_weight=0.0,
            target_weight=1.0,
        )
        ownership_weight = 1.0

        phase_metrics["move"] = run_phase(
            reader,
            publisher,
            low_cmd,
            crc,
            captured_upper,
            target_upper,
            move_seconds,
            captured_state.mode_machine,
            captured_upper,
        )

        if hold_seconds > 0.0:
            phase_metrics["hold"] = run_phase(
                reader,
                publisher,
                low_cmd,
                crc,
                target_upper,
                target_upper,
                hold_seconds,
                captured_state.mode_machine,
                captured_upper,
            )

        phase_metrics["return"] = run_phase(
            reader,
            publisher,
            low_cmd,
            crc,
            target_upper,
            captured_upper,
            move_seconds,
            captured_state.mode_machine,
            captured_upper,
        )

    except BaseException:
        safe_release(
            reader, publisher, low_cmd, crc, ownership_weight
        )
        raise

    safe_release(
        reader,
        publisher,
        low_cmd,
        crc,
        ownership_weight,
        fixed_target=captured_upper,
    )

    return {
        "mode": "physical_guarded_trial",
        "published": True,
        "fraction": fraction,
        "move_s": move_seconds,
        "hold_s": hold_seconds,
        "captured_upper_body": captured_upper,
        "target_upper_body": target_upper,
        "metrics": phase_metrics,
        "released_weight": 0.0,
    }


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface")
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--move-seconds", type=float, required=True)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--confirm", required=True)
    parsed = parser.parse_args(args)

    result = run(
        parsed.interface,
        parsed.fraction,
        parsed.move_seconds,
        parsed.hold_seconds,
        parsed.confirm,
    )
    print(
        "PHYSICAL_TRIAL_RESULT=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
