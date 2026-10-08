"""Adapt acknowledged G1 ROS commands to the project MuJoCo backend."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from g1_robot_interfaces.bridge_server import BridgeServer
from g1_robot_interfaces.arm_motion import validate_arm_motion
from g1_robot_interfaces.locomotion import PlanarVelocityLimits
from g1_robot_interfaces.validation import (
    validate_hand_name,
    validate_hand_ramp_seconds,
    validate_hand_target,
)
from g1_simulation_bridge.simulation_state import (
    SimulationStateReader,
)
from g1_simulation_bridge.state_config import (
    load_lidar_publisher_config,
    load_simulation_locomotion_config,
    load_state_publisher_config,
)
from g1_simulation_bridge.simulation_timing import (
    LidarPublicationSchedule,
    SimulationPublicationSchedule,
)
from g1_simulation_bridge.state_publisher import (
    SimulationStatePublisher,
)
from g1_simulation_bridge.lidar_publisher import (
    SimulationLidarPublisher,
)
from g1_simulation_bridge.mujoco_raycast import (
    MuJoCoRayAdapter,
)
from g1_simulation_bridge.simulation_locomotion import (
    SimulationLocomotionController,
)


def resolve_simulation_root(explicit_path=None):
    """Find the checkout's simulation directory or validate an override."""
    candidates = []

    def add_candidate(path):
        candidate = Path(path)
        candidates.append(candidate)
        candidates.append(candidate / "mujoco")

    if explicit_path:
        add_candidate(explicit_path)
    if os.environ.get("IEA_G1_SIMULATION_ROOT"):
        add_candidate(os.environ["IEA_G1_SIMULATION_ROOT"])
    candidates.extend(
        parent / "simulation" / "mujoco"
        for parent in [Path.cwd(), *Path.cwd().parents]
    )
    candidates.extend(
        parent / "simulation" / "mujoco"
        for parent in Path(__file__).parents
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (
            resolved
            / "runtime"
            / "integrated_backend.py"
        ).is_file():
            return resolved
    raise FileNotFoundError(
        "simulation backend not found; run from the IEA_G1 checkout or set "
        "IEA_G1_SIMULATION_ROOT")


def load_backend(simulation_root):
    """Load the MuJoCo implementation behind the ROS adapter boundary."""
    sys.path.insert(0, str(simulation_root))
    from runtime.integrated_backend import IntegratedG1Backend, SimulationControls
    return SimulationControls, IntegratedG1Backend


class G1SimulationAdapter(Node):
    """Validate ROS requests and dispatch them to named simulation methods."""

    def __init__(self, controls):
        super().__init__("g1_simulation_bridge")
        self.controls = controls
        self.declare_parameter(
            "right_hand_neutral_target",
            [0.399, 0.399, 0.050, 0.050, 0.049, 0.050])
        self.declare_parameter(
            "left_hand_neutral_target",
            [0.400, 0.400, 0.049, 0.050, 0.049, 0.050])
        self.neutral_targets = {
            "right": list(self.get_parameter(
                "right_hand_neutral_target").value),
            "left": list(self.get_parameter(
                "left_hand_neutral_target").value),
        }
        for hand, target in self.neutral_targets.items():
            self.neutral_targets[hand] = validate_hand_target(
                target, f"{hand} hand neutral")
        self.server = BridgeServer(self, "simulation", {
            "arm_prepare": self._prepare,
            "arm_motion_prepare": self._prepare,
            "arm_execute": self._arm_execute,
            "arm_motion": self._arm_motion,
            "arm_motion_release": self._arm_motion_release,
            "arm_interrupt": self._arm_interrupt,
            "arm_release": self._arm_release,
            "hand_prepare": self._prepare,
            "hand_command": self._hand_command,
            "hand_interrupt": self._hand_interrupt,
            "hand_neutral": self._hand_neutral,
        })

    def _prepare(self, request):
        del request
        return None

    def _arm_execute(self, request):
        action_id = int(request["action_id"])
        if action_id != 27:
            raise ValueError("only arm action 27 is implemented")
        self.controls.set_arm(action_id, f"ros:arm:{action_id}")
        return {"code": 0}

    def _arm_interrupt(self, request):
        del request
        return None

    def _arm_motion(self, request):
        """Accept the restricted neutral arm motion contract."""
        motion = validate_arm_motion(request['motion'])
        execution_timeout_s = float(request['execution_timeout_s'])
        if not math.isfinite(execution_timeout_s) or execution_timeout_s <= 0.0:
            raise ValueError('execution_timeout_s must be finite and positive')
        self.controls.set_arm_motion(motion)
        return {'mode': 'simulation',
                'trajectory_duration': sum(motion['phase_seconds'])}

    def _arm_release(self, request):
        del request
        self.controls.set_arm(99, "ros:arm:99")
        return {"code": 0}

    def _arm_motion_release(self, request):
        del request
        self.controls.set_arm(99, "ros:card-arm:release")
        return {"mode": "simulation", "released": True}

    def _hand_command(self, request):
        hand = validate_hand_name(request["hand"])
        target = validate_hand_target(request["target"])
        ramp = validate_hand_ramp_seconds(request["ramp_s"])
        self.controls.set_hand(
            hand, target, f"ros:{hand}_hand:command", ramp)
        return None

    def _hand_interrupt(self, request):
        del request
        return None

    def _hand_neutral(self, request):
        hand = validate_hand_name(request["hand"])
        ramp = validate_hand_ramp_seconds(request["ramp_s"])
        self.controls.set_hand(
            hand, self.neutral_targets[hand], f"ros:{hand}_hand:neutral",
            ramp)
        return None

    def safe_reset(self):
        """Return application-controlled targets to passive defaults."""
        for hand, target in self.neutral_targets.items():
            self.controls.set_hand(
                hand, target, f"bridge:shutdown:{hand}_hand_neutral")
        self.controls.set_arm(99, "bridge:shutdown:arm_release")


def build_lidar_publisher(
    *,
    node,
    backend,
    config,
    geometry,
):
    """Build the enabled lidar runtime or preserve the disabled path."""
    if not config.enabled:
        return None

    ray_adapter = MuJoCoRayAdapter(
        model=backend.model,
        data=backend.data,
        torso_body_name=(
            config.torso_body_name
        ),
        robot_root_body_name=(
            config.robot_root_body_name
        ),
    )
    schedule = LidarPublicationSchedule(
        period_seconds=config.period_seconds,
    )
    return SimulationLidarPublisher(
        node=node,
        ray_adapter=ray_adapter,
        schedule=schedule,
        geometry=geometry,
        directions_sensor=(
            geometry.integration_ray_directions()
        ),
        frame_id=config.frame_id,
        range_min=config.range_min,
        range_max=config.range_max,
    )



def build_simulation_locomotion(
    *,
    node,
    backend,
    config,
    carrier_type,
):
    """Build the explicitly enabled planar carrier runtime."""
    if not config.enabled:
        return None

    carrier = carrier_type(
        mujoco=backend.mujoco,
        model=backend.model,
        data=backend.data,
        harness=backend.harness,
        command_timeout_seconds=(
            config.command_timeout_seconds
        ),
        max_forward_mps=config.max_forward_mps,
        max_lateral_mps=config.max_lateral_mps,
        max_yaw_rate_rps=config.max_yaw_rate_rps,
        allow_lateral=config.allow_lateral,
        joint_name=config.joint_name,
    )
    return SimulationLocomotionController(
        node=node,
        carrier=carrier,
        data=backend.data,
        velocity_limits=PlanarVelocityLimits(
            forward_mps=config.max_forward_mps,
            lateral_mps=config.max_lateral_mps,
            yaw_rate_rps=config.max_yaw_rate_rps,
        ),
        allow_lateral=config.allow_lateral,
    )

def advance_simulation(
    *,
    backend,
    state_publisher,
    substeps,
    lidar_publisher=None,
    locomotion_controller=None,
):
    """Advance physics and publish after every completed step."""

    if (
        isinstance(substeps, bool)
        or not isinstance(substeps, int)
        or substeps <= 0
    ):
        raise ValueError(
            "substeps must be a positive integer"
        )

    for _ in range(substeps):
        backend.step()
        simulation_time = float(
            backend.data.time
        )
        if locomotion_controller is not None:
            locomotion_controller.after_step()
        state_publisher.after_step(
            data=backend.data,
            simulation_time=simulation_time,
        )
        if lidar_publisher is not None:
            lidar_publisher.after_step(
                simulation_time=simulation_time,
            )


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--simulation-root", type=Path)
    return parser.parse_known_args(args)


def main(args=None):
    parsed, ros_args = parse_args(args)
    if parsed.render_fps <= 0.0:
        raise ValueError("render-fps must be positive")
    simulation_root = resolve_simulation_root(parsed.simulation_root)
    controls_type, backend_type = load_backend(simulation_root)
    controls = controls_type()
    backend = backend_type(controls)
    if parsed.validate_only:
        print("VALID: g1_simulation_bridge adapter loaded the MuJoCo backend")
        print("VALID: backend-neutral arm and hand operations")
        return 0

    import mujoco

    rclpy.init(args=ros_args)
    node = G1SimulationAdapter(controls)
    state_config = load_state_publisher_config(node)
    lidar_config = load_lidar_publisher_config(
        node
    )
    locomotion_config = (
        load_simulation_locomotion_config(node)
    )
    state_publisher = SimulationStatePublisher(
        node=node,
        state_reader=SimulationStateReader(
            mujoco=mujoco,
            model=backend.model,
        ),
        schedule=SimulationPublicationSchedule(
            state_period_seconds=(
                state_config.state_period_seconds
            ),
        ),
        imu_frame_id=state_config.imu_frame_id,
        odom_frame_id=state_config.odom_frame_id,
        child_frame_id=state_config.child_frame_id,
        covariance_variance=(
            state_config.covariance_variance
        ),
    )

    lidar_geometry = None
    if lidar_config.enabled:
        from navigation import lidar_geometry

    lidar_publisher = build_lidar_publisher(
        node=node,
        backend=backend,
        config=lidar_config,
        geometry=lidar_geometry,
    )

    carrier_type = None
    if locomotion_config.enabled:
        from navigation.mujoco_planar_carrier import (
            MuJoCoPlanarCarrier,
        )
        carrier_type = MuJoCoPlanarCarrier

    locomotion_controller = (
        build_simulation_locomotion(
            node=node,
            backend=backend,
            config=locomotion_config,
            carrier_type=carrier_type,
        )
    )
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous_sigint = signal.signal(signal.SIGINT, stop)
    frame_period = 1.0 / parsed.render_fps
    substeps = max(1, round(frame_period / backend.model.opt.timestep))

    def frame():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
        advance_simulation(
            backend=backend,
            state_publisher=state_publisher,
            substeps=substeps,
            lidar_publisher=lidar_publisher,
            locomotion_controller=locomotion_controller,
        )

    try:
        if parsed.headless:
            while not interrupted and rclpy.ok():
                started = time.monotonic()
                frame()
                remaining = frame_period - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(
                    backend.model, backend.data) as viewer:
                while (viewer.is_running() and not interrupted
                       and rclpy.ok()):
                    started = time.monotonic()
                    frame()
                    viewer.sync()
                    remaining = frame_period - (time.monotonic() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
    finally:
        if locomotion_controller is not None:
            locomotion_controller.stop()
        node.safe_reset()
        for _ in range(substeps):
            backend.step()
        signal.signal(signal.SIGINT, previous_sigint)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    # SIGINT from ros2 launch is an expected supervised shutdown.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
