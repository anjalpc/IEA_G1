"""BrainCo diagnostics, dry-run planning, and guarded curl-pulse testing."""

import argparse
import json
import statistics
import time


JOINT_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")


def plan_curl(amount):
    """Return a conservative symmetric curl without commanding hardware."""
    amount = max(0.0, min(float(amount), 0.25))
    # Keep thumb opposition smaller during the initial object-only tests.
    return [round(amount * 0.4, 4), round(amount * 0.25, 4)] + [amount] * 4


def read_state(interface, hand, timeout):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

    ChannelFactoryInitialize(0, interface)
    topic = f"rt/brainco/{hand}/state"
    subscriber = ChannelSubscriber(topic, MotorStates_)
    subscriber.Init()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = subscriber.Read()
        if message is not None:
            values = [float(message.states[index].q) for index in range(6)]
            return {"hand": hand, "topic": topic,
                    "joints": dict(zip(JOINT_NAMES, values))}
        time.sleep(0.01)
    raise TimeoutError(f"no BrainCo {hand} state received within {timeout}s")


def command_position(interface, hand, amount, ramp_seconds, timeout,
                     confirmation):
    """Ramp the manufacturer's six-motor DDS interface to a safe target."""
    if confirmation != "HANDSHAKE_OPERATOR_ENABLED":
        raise ValueError(
            "handshake control requires --confirm HANDSHAKE_OPERATOR_ENABLED")
    if hand != "right":
        raise ValueError("handshake control is restricted to the right hand")
    if not 0.0 <= amount <= 0.15:
        raise ValueError("handshake position must be between 0 and 0.15")
    if not 0.5 <= ramp_seconds <= 5.0:
        raise ValueError("ramp duration must be between 0.5 and 5.0 seconds")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

    ChannelFactoryInitialize(0, interface)
    subscriber = ChannelSubscriber(
        f"rt/brainco/{hand}/state", MotorStates_)
    subscriber.Init()
    publisher = ChannelPublisher(f"rt/brainco/{hand}/cmd", MotorCmds_)
    publisher.Init()

    deadline = time.monotonic() + timeout
    initial = None
    while time.monotonic() < deadline:
        state = subscriber.Read()
        if state is not None and len(state.states) >= 6:
            initial = [float(motor.q) for motor in state.states[:6]]
            break
        time.sleep(0.01)
    if initial is None:
        raise TimeoutError(f"no BrainCo {hand} state received within {timeout}s")
    if any(value < -0.05 or value > 1.05 for value in initial):
        raise RuntimeError(f"invalid initial hand state: {initial}")

    target = plan_curl(amount) if amount else [0.0] * 6
    message = MotorCmds_()
    message.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
    for motor in message.cmds:
        motor.dq = 1.0

    steps = max(2, int(ramp_seconds * 25.0))
    for step in range(1, steps + 1):
        fraction = step / steps
        values = [start + (end - start) * fraction
                  for start, end in zip(initial, target)]
        for index, value in enumerate(values):
            message.cmds[index].q = float(max(0.0, min(value, 1.0)))
        publisher.Write(message)
        time.sleep(1.0 / 25.0)
    for _ in range(5):
        publisher.Write(message)
        time.sleep(1.0 / 25.0)
    return {"hand": hand, "initial": dict(zip(JOINT_NAMES, initial)),
            "target": dict(zip(JOINT_NAMES, target)), "commanded": True}


def pulse_curl(interface, hand, amount, ramp_seconds, hold_seconds, timeout,
               confirmation, guarded_object=False, target_override=None):
    """Perform one conservative curl pulse and always command reopening."""
    required_confirmation = "SOFT_OBJECT" if guarded_object else "EMPTY_HAND"
    if confirmation != required_confirmation:
        raise ValueError(
            f"this hardware test requires --confirm {required_confirmation}")
    if hand != "right":
        raise ValueError("initial hardware testing is restricted to the right hand")
    if target_override is None and not 0.0 < amount <= 0.15:
        raise ValueError("initial hardware curl amount must be > 0 and <= 0.15")
    if target_override is not None:
        if len(target_override) != 6:
            raise ValueError("target override must contain six joint positions")
        if any(value < 0.0 or value > 1.0 for value in target_override):
            raise ValueError("target override positions must be between 0 and 1")
    if not 0.5 <= ramp_seconds <= 5.0:
        raise ValueError("ramp duration must be between 0.5 and 5.0 seconds")
    if not 0.0 <= hold_seconds <= 2.0:
        raise ValueError("hold duration must be between 0 and 2.0 seconds")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

    ChannelFactoryInitialize(0, interface)
    state_topic = f"rt/brainco/{hand}/state"
    command_topic = f"rt/brainco/{hand}/cmd"
    subscriber = ChannelSubscriber(state_topic, MotorStates_)
    subscriber.Init()
    publisher = ChannelPublisher(command_topic, MotorCmds_)
    publisher.Init()

    deadline = time.monotonic() + timeout
    initial = None
    while time.monotonic() < deadline:
        message = subscriber.Read()
        if message is not None and len(message.states) >= 6:
            initial = [float(message.states[index].q) for index in range(6)]
            break
        time.sleep(0.01)
    if initial is None:
        raise TimeoutError(f"no BrainCo {hand} state received within {timeout}s")
    if any(value < -0.05 or value > 1.05 for value in initial):
        raise RuntimeError(f"invalid initial hand state: {initial}")

    message = MotorCmds_()
    message.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(6)]
    for motor in message.cmds:
        motor.dq = 1.0

    def publish(values):
        for index, value in enumerate(values):
            message.cmds[index].q = float(max(0.0, min(value, 1.0)))
        publisher.Write(message)

    target = (list(target_override) if target_override is not None
              else plan_curl(amount))
    opened = [0.0] * 6
    rate_hz = 25.0
    current_command = initial[:]
    measured_peak = initial[:]
    peak_abs_velocity = [0.0] * 6
    peak_abs_torque = [0.0] * 6
    max_temperature = [0] * 6
    communication_lost = [0] * 6
    current_samples = {"closing": [[] for _ in range(6)],
                       "holding": [[] for _ in range(6)],
                       "opening": [[] for _ in range(6)]}
    tracking_error_samples = {"closing": [[] for _ in range(6)],
                              "holding": [[] for _ in range(6)],
                              "opening": [[] for _ in range(6)]}
    contact_counts = [0] * 6
    contact_detection = None

    # thumb_aux is excluded because its normal opening current is unusually high.
    contact_error_threshold = [0.010, None, 0.020, 0.020, 0.020, 0.020]
    contact_current_threshold = [0.040, None, 0.080, 0.080, 0.080, 0.080]
    holding_current_threshold = [0.040, None, 0.100, 0.080, 0.080, 0.080]
    contact_samples_required = 5

    def sample_state(phase=None):
        nonlocal measured_peak, peak_abs_velocity, peak_abs_torque
        nonlocal max_temperature, communication_lost
        state_message = subscriber.Read()
        if state_message is not None and len(state_message.states) >= 6:
            motors = state_message.states[:6]
            values = [float(motor.q) for motor in motors]
            measured_peak = [max(old, new)
                             for old, new in zip(measured_peak, values)]
            peak_abs_velocity = [max(old, abs(float(motor.dq)))
                                 for old, motor in zip(peak_abs_velocity, motors)]
            peak_abs_torque = [max(old, abs(float(motor.tau_est)))
                               for old, motor in zip(peak_abs_torque, motors)]
            if phase in current_samples:
                for index, motor in enumerate(motors):
                    current_samples[phase][index].append(float(motor.tau_est))
                    tracking_error_samples[phase][index].append(
                        abs(current_command[index] - values[index]))
            max_temperature = [max(old, int(motor.temperature))
                               for old, motor in zip(max_temperature, motors)]
            communication_lost = [max(old, int(motor.lost))
                                  for old, motor in zip(communication_lost, motors)]
            return motors, values
        return None

    def ramp(start, end, duration, phase):
        nonlocal current_command, contact_detection
        steps = max(2, int(duration * rate_hz))
        for step in range(1, steps + 1):
            fraction = step / steps
            current_command = [a + (b - a) * fraction
                               for a, b in zip(start, end)]
            publish(current_command)
            sample = sample_state(phase)
            if guarded_object and phase == "closing" and sample is not None:
                motors, values = sample
                for index, (motor, measured) in enumerate(zip(motors, values)):
                    error_limit = contact_error_threshold[index]
                    current_limit = contact_current_threshold[index]
                    if error_limit is None:
                        continue
                    error = abs(current_command[index] - measured)
                    current = abs(float(motor.tau_est))
                    if error >= error_limit and current >= current_limit:
                        contact_counts[index] += 1
                    else:
                        contact_counts[index] = 0
                    if contact_counts[index] >= contact_samples_required:
                        contact_detection = {
                            "phase": "closing",
                            "reason": "sustained tracking error and current",
                            "joint": JOINT_NAMES[index],
                            "consecutive_samples": contact_counts[index],
                            "tracking_error": error,
                            "current_A": current,
                            "commanded_position": current_command[index],
                            "measured_position": measured,
                        }
                        return False
            time.sleep(1.0 / rate_hz)
        return True

    try:
        completed = ramp(initial, target, ramp_seconds, "closing")
        if completed:
            contact_counts[:] = [0] * 6
            end_hold = time.monotonic() + hold_seconds
            while time.monotonic() < end_hold:
                publish(target)
                sample = sample_state("holding")
                if guarded_object and sample is not None:
                    motors, values = sample
                    for index, motor in enumerate(motors):
                        current_limit = holding_current_threshold[index]
                        if current_limit is None:
                            continue
                        current = abs(float(motor.tau_est))
                        if current >= current_limit:
                            contact_counts[index] += 1
                        else:
                            contact_counts[index] = 0
                        if contact_counts[index] >= contact_samples_required:
                            contact_detection = {
                                "phase": "holding",
                                "reason": "sustained holding current",
                                "joint": JOINT_NAMES[index],
                                "consecutive_samples": contact_counts[index],
                                "tracking_error": abs(
                                    target[index] - values[index]),
                                "current_A": current,
                                "commanded_position": target[index],
                                "measured_position": values[index],
                            }
                            break
                    if contact_detection is not None:
                        break
                time.sleep(1.0 / rate_hz)
    finally:
        # Reopening is attempted even after Ctrl-C or an exception.
        ramp(current_command[:], opened, ramp_seconds, "opening")
        for _ in range(5):
            publish(opened)
            sample_state("opening")
            time.sleep(1.0 / rate_hz)

    final_measured = None
    final_deadline = time.monotonic() + 0.5
    while time.monotonic() < final_deadline:
        state_message = subscriber.Read()
        if state_message is not None and len(state_message.states) >= 6:
            final_measured = [float(state_message.states[index].q)
                              for index in range(6)]
        time.sleep(0.01)

    def summarize_phase(samples):
        def percentile(values, fraction):
            if not values:
                return None
            ordered = sorted(values)
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        result = {}
        for name, values in zip(JOINT_NAMES, samples):
            absolute = [abs(value) for value in values]
            result[name] = {
                "samples": len(values),
                "median_abs_A": (statistics.median(absolute)
                                 if absolute else None),
                "p90_abs_A": percentile(absolute, 0.90),
                "p95_abs_A": percentile(absolute, 0.95),
                "max_abs_A": max(absolute) if absolute else None,
            }
        return result

    def summarize_tracking(samples):
        result = {}
        for name, values in zip(JOINT_NAMES, samples):
            ordered = sorted(values)
            p95_index = round((len(ordered) - 1) * 0.95) if ordered else None
            result[name] = {
                "samples": len(values),
                "median_abs_error": (statistics.median(values)
                                     if values else None),
                "p95_abs_error": (ordered[p95_index]
                                  if p95_index is not None else None),
                "max_abs_error": max(values) if values else None,
            }
        return result

    return {"hand": hand, "initial": dict(zip(JOINT_NAMES, initial)),
            "target": dict(zip(JOINT_NAMES, target)),
            "measured_peak": dict(zip(JOINT_NAMES, measured_peak)),
            "peak_abs_velocity": dict(zip(JOINT_NAMES, peak_abs_velocity)),
            "peak_abs_torque": dict(zip(JOINT_NAMES, peak_abs_torque)),
            "max_temperature": dict(zip(JOINT_NAMES, max_temperature)),
            "communication_lost": dict(zip(JOINT_NAMES, communication_lost)),
            "phase_current": {phase: summarize_phase(samples)
                              for phase, samples in current_samples.items()},
            "phase_tracking_error": {
                phase: summarize_tracking(samples)
                for phase, samples in tracking_error_samples.items()},
            "guarded_object_mode": guarded_object,
            "contact_detection": contact_detection,
            "final_measured": (dict(zip(JOINT_NAMES, final_measured))
                               if final_measured is not None else None),
            "reopen_commanded": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("state", "plan-curl", "pulse-curl",
                                            "guarded-grasp", "thumb-pulse",
                                            "command-position"))
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--amount", type=float, default=0.10)
    parser.add_argument("--ramp-seconds", type=float, default=1.5)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--thumb-position", type=float, default=0.10)
    parser.add_argument("--thumb-aux-position", type=float, default=0.008)
    args = parser.parse_args()

    if args.command == "state":
        result = read_state(args.interface, args.hand, args.timeout)
    elif args.command == "plan-curl":
        result = {
            "dry_run": True,
            "hand": args.hand,
            "maximum_allowed": 0.25,
            "target": dict(zip(JOINT_NAMES, plan_curl(args.amount))),
        }
    elif args.command == "command-position":
        result = command_position(
            args.interface, args.hand, args.amount, args.ramp_seconds,
            args.timeout, args.confirm)
    elif args.command == "thumb-pulse":
        if not 0.0 <= args.thumb_position <= 0.35:
            raise ValueError("thumb pulse is limited to 0.00 through 0.35")
        if not 0.0 <= args.thumb_aux_position <= 0.08:
            raise ValueError("thumb auxiliary pulse is limited to 0.00 through 0.08")
        result = pulse_curl(
            args.interface, args.hand, args.amount, args.ramp_seconds,
            args.hold_seconds, args.timeout, args.confirm,
            target_override=[args.thumb_position, args.thumb_aux_position,
                             0.0, 0.0, 0.0, 0.0])
    else:
        result = pulse_curl(args.interface, args.hand, args.amount,
                            args.ramp_seconds, args.hold_seconds, args.timeout,
                            args.confirm,
                            guarded_object=(args.command == "guarded-grasp"))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
