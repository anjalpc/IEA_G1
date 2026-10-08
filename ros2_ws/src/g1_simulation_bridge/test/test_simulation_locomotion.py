"""Tests for trusted simulation locomotion ingress."""

from types import SimpleNamespace

from geometry_msgs.msg import Twist
from rclpy.qos import (
    DurabilityPolicy,
    ReliabilityPolicy,
)

from g1_robot_interfaces.locomotion import (
    PlanarVelocityLimits,
)
from g1_simulation_bridge.simulation_locomotion import (
    SimulationLocomotionController,
)


class FakeNode:
    """Capture one ROS subscription."""

    def __init__(self):
        self.subscription = None

    def create_subscription(
        self,
        message_type,
        topic,
        callback,
        qos,
    ):
        """Record subscription construction."""
        self.subscription = SimpleNamespace(
            message_type=message_type,
            topic_name=topic,
            callback=callback,
            qos=qos,
        )
        return self.subscription

    def get_logger(self):
        """Return a logger that accepts sanitized messages."""
        return SimpleNamespace(
            error=lambda _message: None,
        )


class FakeCarrier:
    """Record carrier commands and stops."""

    def __init__(self):
        self.commands = []
        self.stop_count = 0
        self.step_times = []

    def command(self, **kwargs):
        """Record one carrier command."""
        self.commands.append(kwargs)

    def stop(self):
        """Record one carrier stop."""
        self.stop_count += 1

    def after_step(self, *, simulation_time):
        """Record one completed simulation step."""
        self.step_times.append(simulation_time)


def controller():
    """Return a controller with deterministic fakes."""
    node = FakeNode()
    carrier = FakeCarrier()
    data = SimpleNamespace(time=12.5)
    instance = SimulationLocomotionController(
        node=node,
        carrier=carrier,
        data=data,
        velocity_limits=PlanarVelocityLimits(
            forward_mps=0.15,
            lateral_mps=0.01,
            yaw_rate_rps=0.40,
        ),
        allow_lateral=False,
    )
    return node, carrier, data, instance


def valid_twist():
    """Return one safe forward-only velocity."""
    message = Twist()
    message.linear.x = 0.10
    message.angular.z = -0.20
    return message


def test_subscribes_to_private_simulation_command_boundary():
    """Only the arbiter-facing simulation topic is consumed."""
    node, _carrier, _data, instance = controller()

    assert (
        node.subscription.topic_name
        == "/g1/simulation/locomotion_command"
    )
    assert instance.command_qos.depth == 1
    assert (
        instance.command_qos.reliability
        == ReliabilityPolicy.BEST_EFFORT
    )
    assert (
        instance.command_qos.durability
        == DurabilityPolicy.VOLATILE
    )


def test_valid_command_reaches_carrier_in_simulation_time():
    """Validated planar velocity is timestamped by MuJoCo."""
    _node, carrier, _data, instance = controller()

    instance.command_callback(valid_twist())

    assert carrier.commands == [{
        "forward_mps": 0.10,
        "lateral_mps": 0.0,
        "yaw_rate_rps": -0.20,
        "simulation_time": 12.5,
    }]
    assert carrier.stop_count == 0


def test_invalid_command_stops_without_reaching_carrier():
    """Unsupported velocity axes fail closed."""
    _node, carrier, _data, instance = controller()
    message = valid_twist()
    message.linear.z = 0.01

    instance.command_callback(message)

    assert carrier.commands == []
    assert carrier.stop_count == 1


def test_completed_physics_step_advances_carrier():
    """Carrier state is applied after each physics step."""
    _node, carrier, data, instance = controller()
    data.time = 13.25

    instance.after_step()

    assert carrier.step_times == [13.25]
