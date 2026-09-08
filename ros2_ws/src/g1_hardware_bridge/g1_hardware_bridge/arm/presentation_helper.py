"""Persistent physical left-card presentation over Unitree rt/arm_sdk."""

import argparse
import json
import math
import os
import select
import signal
import sys
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


PRESENTED_PREFIX = "PRESENTED="
RELEASED_PREFIX = "RELEASED="


RIGHT_REST = [
    +0.239587,
    -0.270012,
    +0.079531,
    +0.802344,
    -0.004820,
    +0.007794,
    +0.003641,
]

RIGHT_ACTION27 = [
    -0.108721,
    -0.063594,
    +0.094819,
    +0.182400,
    +0.159618,
    -0.339249,
    -0.061755,
]

MIRROR_SIGN = [
    +1.0,   # shoulder pitch
    -1.0,   # shoulder roll
    -1.0,   # shoulder yaw
    +1.0,   # elbow
    -1.0,   # wrist roll
    +1.0,   # wrist pitch
    -1.0,   # wrist yaw
]

# Measured from Unitree Action 27 bilateral preparation/companion posture.
LEFT_COMPANION = [
    +0.286,
    +0.218,
    -0.009,
    +0.978,
    +0.144,
    +0.006,
    -0.004,
]

RIGHT_COMPANION = [
    +0.286,
    -0.218,
    +0.009,
    +0.978,
    -0.144,
    +0.006,
    +0.004,
]

LEFT_IDS = list(range(15, 22))
RIGHT_IDS = list(range(22, 29))
ARM_IDS = set(range(15, 29))
WRIST_IDS = {19, 20, 21, 26, 27, 28}
WEIGHT_INDEX = 29

DT = 0.02
STATE_TIMEOUT = 0.5

KP_BODY = 300.0
KD_BODY = 3.0
KP_ARM = 80.0
KD_ARM = 3.0
KP_WRIST = 40.0
KD_WRIST = 1.5

# Fast authority transition.
ACQUIRE_TIME = 0.30
SETTLE_TIME = 0.05

# Measured Unitree-style visible choreography:
# ~0.30 s bilateral preparation
# ~0.90 s performing-arm extension
PREPARE_TIME = 0.30
EXTEND_TIME = 0.90

# Fast return / hand-back to Unitree controller.
HOME_SETTLE_TIME = 0.05
RELEASE_TIME = 0.30

MAX_ROLL_CHANGE = 0.10
MAX_PITCH_CHANGE = 0.10
MAX_WAIST_CHANGE = {
    12: 0.10,
    13: 0.15,
    14: 0.15,
}


def smoothstep5(value):
    value = max(0.0, min(1.0, value))
    return (
        10.0 * value**3
        - 15.0 * value**4
        + 6.0 * value**5
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("interface")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--move-seconds",
        type=float,
        default=1.2,
    )
    return parser.parse_args()


class PresentationController:

    def __init__(self, interface, fraction, move_seconds):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be between 0 and 1")

        # Keep accepting the legacy backend argument for API compatibility.
        # Physical timing is now fixed to the measured Unitree choreography.
        if move_seconds <= 0.0:
            raise ValueError("move_seconds must be positive")

        self.interface = interface
        self.fraction = fraction

        # Do not derive physical timing from the legacy caller value anymore.
        self.prepare_seconds = PREPARE_TIME
        self.extend_seconds = EXTEND_TIME
        self.move_seconds = self.prepare_seconds + self.extend_seconds

        self.extension_started = False

        right_delta = [
            action - rest
            for action, rest in zip(
                RIGHT_ACTION27,
                RIGHT_REST,
            )
        ]

        self.left_delta = [
            sign * delta
            for sign, delta in zip(
                MIRROR_SIGN,
                right_delta,
            )
        ]

        self.release_requested = False
        self.state = None
        self.state_lock = threading.Lock()
        self.state_received_at = None
        self.state_tick = None
        self.weight = 0.0
        self.command_initialized = False
        self.input_buffer = b""

        ChannelFactoryInitialize(0, interface)

        self.subscriber = ChannelSubscriber(
            "rt/lowstate",
            LowState_,
        )
        self.subscriber.Init(self.receive_state)

        self.publisher = ChannelPublisher(
            "rt/arm_sdk",
            LowCmd_,
        )
        self.publisher.Init()

        self.crc = CRC()
        self.cmd = unitree_hg_msg_dds__LowCmd_()

        self.q_start = None
        self.baseline_roll = None
        self.baseline_pitch = None
        self.baseline_waist = None

    def request_release(self, *_args):
        self.release_requested = True

    def receive_state(self, message):
        """Keep the newest advancing telemetry sample, without a DDS backlog."""
        with self.state_lock:
            if message.tick != self.state_tick:
                self.state = message
                self.state_tick = message.tick
                self.state_received_at = time.monotonic()

    def latest(self):
        with self.state_lock:
            state = self.state
            received_at = self.state_received_at

        if (
            state is None
            or received_at is None
            or time.monotonic() - received_at > STATE_TIMEOUT
        ):
            raise RuntimeError("stale or unavailable rt/lowstate")

        if len(state.motor_state) < 29:
            raise RuntimeError("invalid rt/lowstate motor count")

        values = [
            motor.q
            for motor in state.motor_state[:29]
        ]
        values.extend(state.imu_state.rpy[:2])

        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("invalid rt/lowstate")

        return state

    def poll_release(self):
        """Accept RELEASE or parent EOF during every acquisition/motion phase."""
        readable, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            0.0,
        )

        if readable:
            chunk = os.read(
                sys.stdin.fileno(),
                4096,
            )

            if not chunk:
                self.release_requested = True

            else:
                self.input_buffer += chunk

                while b"\n" in self.input_buffer:
                    line, self.input_buffer = (
                        self.input_buffer.split(b"\n", 1)
                    )

                    if line.strip().upper() == b"RELEASE":
                        self.release_requested = True

                if len(self.input_buffer) > 4096:
                    raise RuntimeError(
                        "oversized helper command"
                    )

        return self.release_requested

    def wait_for_state(self):
        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:
            if self.poll_release():
                return None

            with self.state_lock:
                available = self.state is not None

            if available:
                return self.latest()

            time.sleep(0.01)

        raise TimeoutError(
            "rt/lowstate is unavailable"
        )

    def initialize_command(self, state):
        self.cmd.mode_pr = 0
        self.cmd.mode_machine = state.mode_machine

        for joint in range(29):
            measured = state.motor_state[joint].q
            motor = self.cmd.motor_cmd[joint]

            motor.mode = 1
            motor.q = measured
            motor.dq = 0.0
            motor.tau = 0.0

            if joint in WRIST_IDS:
                motor.kp = KP_WRIST
                motor.kd = KD_WRIST

            elif joint in ARM_IDS:
                motor.kp = KP_ARM
                motor.kd = KD_ARM

            else:
                motor.kp = KP_BODY
                motor.kd = KD_BODY

    def publish(self, weight):
        self.weight = float(weight)

        self.cmd.motor_cmd[WEIGHT_INDEX].q = (
            float(weight)
        )

        self.cmd.crc = self.crc.Crc(self.cmd)

        if self.publisher.Write(self.cmd) is False:
            raise RuntimeError(
                "rt/arm_sdk publication failed"
            )

    def safety_check(self):
        state = self.latest()

        roll = state.imu_state.rpy[0]
        pitch = state.imu_state.rpy[1]

        if (
            abs(
                roll
                - self.baseline_roll
            )
            > MAX_ROLL_CHANGE
        ):
            raise RuntimeError(
                "roll safety limit exceeded: "
                f"{roll - self.baseline_roll:+.3f} rad"
            )

        if (
            abs(
                pitch
                - self.baseline_pitch
            )
            > MAX_PITCH_CHANGE
        ):
            raise RuntimeError(
                "pitch safety limit exceeded: "
                f"{pitch - self.baseline_pitch:+.3f} rad"
            )

        for joint, limit in MAX_WAIST_CHANGE.items():
            deviation = (
                state.motor_state[joint].q
                - self.baseline_waist[joint]
            )

            if abs(deviation) > limit:
                raise RuntimeError(
                    f"waist joint {joint} safety limit exceeded: "
                    f"{deviation:+.3f} rad"
                )

    def acquire(self):
        state = self.wait_for_state()

        if state is None:
            return False

        self.initialize_command(state)
        self.command_initialized = True

        # Initial baseline protects the acquisition transition itself.
        self.capture_baseline(state)

        start = time.monotonic()

        while True:
            now = time.monotonic()
            elapsed = now - start

            if elapsed >= ACQUIRE_TIME:
                break

            if self.poll_release():
                return False

            self.safety_check()

            self.publish(
                smoothstep5(
                    elapsed / ACQUIRE_TIME
                )
            )

            time.sleep(DT)

        self.publish(1.0)

        # Short post-acquisition settle.
        start = time.monotonic()

        while (
            time.monotonic() - start
            < SETTLE_TIME
        ):
            if self.poll_release():
                return False

            self.safety_check()
            self.publish(1.0)
            time.sleep(DT)

        # IMPORTANT:
        # Capture the actual post-acquisition pose.
        # This becomes the trajectory origin and return pose.
        state = self.latest()
        self.capture_baseline(state)

        return True

    def capture_baseline(self, state):
        self.q_start = {
            joint: state.motor_state[joint].q
            for joint in range(29)
        }

        self.baseline_roll = (
            state.imu_state.rpy[0]
        )

        self.baseline_pitch = (
            state.imu_state.rpy[1]
        )

        self.baseline_waist = {
            joint: state.motor_state[joint].q
            for joint in (12, 13, 14)
        }

    def companion(self):
        """
        Return the bilateral preparation/companion posture.

        fraction scales the complete gesture for staged
        physical validation.
        """
        result = dict(self.q_start)

        for joints, pose in (
            (LEFT_IDS, LEFT_COMPANION),
            (RIGHT_IDS, RIGHT_COMPANION),
        ):
            for joint, value in zip(
                joints,
                pose,
            ):
                result[joint] += (
                    self.fraction
                    * (
                        value
                        - self.q_start[joint]
                    )
                )

        return result

    def target(self):
        """
        Return the presentation target.

        Right arm remains in the companion posture.
        Left arm continues toward the mirrored Action-27 card pose.
        """
        result = self.companion()

        for local_index, joint in enumerate(
            LEFT_IDS
        ):
            result[joint] = (
                self.q_start[joint]
                + self.fraction
                * self.left_delta[local_index]
            )

        return result

    def move_phase(
        self,
        origin,
        target,
        seconds,
        *,
        returning=False,
    ):
        """
        Smoothly interpolate the complete arm command.

        Only arm joints 15-28 move.
        All other joints retain the validated post-acquisition q_start.
        """
        if seconds <= 0.0:
            raise ValueError(
                "phase duration must be positive"
            )

        weight = self.weight
        start = time.monotonic()
        deadline = start + seconds

        while True:
            now = time.monotonic()
            elapsed = now - start

            if not returning:
                if self.poll_release():
                    return False

                self.safety_check()

            else:
                # Recovery must remain possible after a safety trip,
                # but never generate a return trajectory using stale
                # telemetry.
                self.latest()

            if now >= deadline:
                alpha = 1.0
            else:
                alpha = smoothstep5(
                    elapsed / seconds
                )

            for joint in range(29):
                if joint in ARM_IDS:
                    self.cmd.motor_cmd[joint].q = (
                        origin[joint]
                        + alpha
                        * (
                            target[joint]
                            - origin[joint]
                        )
                    )

                else:
                    self.cmd.motor_cmd[joint].q = (
                        self.q_start[joint]
                    )

            self.publish(weight)

            if now >= deadline:
                return True

            time.sleep(
                min(
                    DT,
                    deadline - now,
                )
            )

    def move_to_presentation(self):
        """
        Unitree-style choreography.

        Phase A:
            both arms -> bilateral companion pose (~0.30 s)

        Phase B:
            right arm holds companion
            left arm -> card presentation (~0.90 s)
        """
        companion = self.companion()

        if not self.move_phase(
            self.q_start,
            companion,
            self.prepare_seconds,
        ):
            return False

        self.extension_started = True

        return self.move_phase(
            companion,
            self.target(),
            self.extend_seconds,
        )

    def wait_presented(self):
        """Hold the presented posture until RELEASE."""
        while not self.poll_release():
            self.safety_check()
            self.publish(1.0)
            time.sleep(DT)

    def return_home(self):
        """
        Reverse the Unitree-style choreography.

        Phase C:
            left arm presentation -> bilateral companion (~0.90 s)

        Phase D:
            both arms companion -> captured Run/start pose (~0.30 s)
        """
        return_weight = self.weight

        state = self.latest()

        # Always begin recovery from actual measured arm state.
        q_return_start = {
            joint: state.motor_state[joint].q
            for joint in range(29)
        }

        if self.extension_started:
            companion = self.companion()

            self.move_phase(
                q_return_start,
                companion,
                self.extend_seconds,
                returning=True,
            )

            q_return_start = companion

        # If cancellation occurred during acquisition/preparation,
        # this safely returns directly to q_start without first
        # extending toward the card pose.
        self.move_phase(
            q_return_start,
            self.q_start,
            self.prepare_seconds,
            returning=True,
        )

        # Tiny stabilization dwell only.
        start = time.monotonic()

        while (
            time.monotonic() - start
            < HOME_SETTLE_TIME
        ):
            self.latest()
            self.publish(return_weight)
            time.sleep(DT)

    def release_authority(self):
        """
        Smoothly hand control back to Unitree.

        The complete LowCmd remains valid during the ramp.
        """
        initial_weight = self.weight
        errors = []

        start = time.monotonic()

        while True:
            now = time.monotonic()
            elapsed = now - start

            if elapsed >= RELEASE_TIME:
                break

            weight = (
                initial_weight
                * (
                    1.0
                    - smoothstep5(
                        elapsed / RELEASE_TIME
                    )
                )
            )

            try:
                self.publish(weight)

            except Exception as exc:
                errors.append(exc)

            time.sleep(DT)

        self.publish(0.0)

        if errors:
            raise RuntimeError(
                "publication failed during authority release"
            ) from errors[0]

    def safe_release(self):
        """
        Best-effort physical cleanup.

        If authority was acquired, always try to return the arms
        and then ramp authority to zero.
        """
        if (
            not self.command_initialized
            or self.weight <= 0.0
        ):
            return

        try:
            self.return_home()

        finally:
            self.release_authority()

    def run(self):
        try:
            acquired = self.acquire()

            if (
                acquired
                and self.move_to_presentation()
            ):
                state = self.latest()

                print(
                    PRESENTED_PREFIX
                    + json.dumps(
                        {
                            "mode": "live",
                            "fraction": self.fraction,
                            "move_seconds": self.move_seconds,
                            "prepare_seconds": self.prepare_seconds,
                            "extend_seconds": self.extend_seconds,
                            "left_q": [
                                state.motor_state[j].q
                                for j in LEFT_IDS
                            ],
                            "right_q": [
                                state.motor_state[j].q
                                for j in RIGHT_IDS
                            ],
                            "waist_q": [
                                state.motor_state[j].q
                                for j in (12, 13, 14)
                            ],
                            "roll": state.imu_state.rpy[0],
                            "pitch": state.imu_state.rpy[1],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

                self.wait_presented()

        finally:
            # Covers:
            # - normal release
            # - cancellation
            # - acquisition failure
            # - motion exception
            # - safety exception
            self.safe_release()

        print(
            RELEASED_PREFIX
            + json.dumps(
                {
                    "mode": "live",
                    "released": True,
                    "prepare_seconds": self.prepare_seconds,
                    "extend_seconds": self.extend_seconds,
                    "release_seconds": RELEASE_TIME,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main():
    args = parse_args()

    controller = PresentationController(
        args.interface,
        args.fraction,
        args.move_seconds,
    )

    signal.signal(
        signal.SIGINT,
        controller.request_release,
    )

    signal.signal(
        signal.SIGTERM,
        controller.request_release,
    )

    controller.run()


if __name__ == "__main__":
    main()
