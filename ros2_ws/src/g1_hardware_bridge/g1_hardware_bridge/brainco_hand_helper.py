"""Minimal BrainCo state and position-command helper."""

import argparse
import json
import os
import sys
import time
import traceback


JOINT_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")
COMMAND_CONFIRMATION = "CONFIRM_PHYSICAL_HAND_COMMAND"


def read_state(interface, hand, timeout):
    """Read one six-joint BrainCo state sample."""
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
        if message is not None and len(message.states) >= 6:
            values = [float(message.states[index].q) for index in range(6)]
            return {"hand": hand, "topic": topic,
                    "joints": dict(zip(JOINT_NAMES, values))}
        time.sleep(0.01)
    raise TimeoutError(f"no BrainCo {hand} state received within {timeout}s")


def command_position(interface, hand, target, ramp_seconds, timeout,
                     confirmation):
    """Ramp the right hand from its measured state to an explicit target."""
    if confirmation != COMMAND_CONFIRMATION:
        raise ValueError(
            f"physical hand control requires --confirm {COMMAND_CONFIRMATION}")
    if hand != "right":
        raise ValueError("handshake control is restricted to the right hand")
    if target is None or len(target) != 6:
        raise ValueError("target must contain six positions")
    if any(value < 0.0 or value > 1.0 for value in target):
        raise ValueError("target positions must be between 0 and 1")
    if not 0.5 <= ramp_seconds <= 5.0:
        raise ValueError("ramp duration must be between 0.5 and 5.0 seconds")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

    ChannelFactoryInitialize(0, interface)
    subscriber = ChannelSubscriber(f"rt/brainco/{hand}/state", MotorStates_)
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
            message.cmds[index].q = float(value)
        publisher.Write(message)
        time.sleep(1.0 / 25.0)
    for _ in range(5):
        publisher.Write(message)
        time.sleep(1.0 / 25.0)
    return {"hand": hand, "initial": dict(zip(JOINT_NAMES, initial)),
            "target": dict(zip(JOINT_NAMES, target)), "commanded": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("state", "command-position"))
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.75)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--target", type=float, nargs=6)
    args = parser.parse_args()

    if args.command == "state":
        result = read_state(args.interface, args.hand, args.timeout)
    else:
        result = command_position(
            args.interface, args.hand, args.target, args.ramp_seconds,
            args.timeout, args.confirm)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # ensure SDK background threads cannot orphan helper
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
