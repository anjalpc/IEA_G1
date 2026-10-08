"""Runtime coupling tests for MuJoCo state publication."""

from types import SimpleNamespace

from g1_simulation_bridge import adapter_node
from g1_simulation_bridge.adapter_node import (
    advance_simulation,
    build_lidar_publisher,
    build_simulation_locomotion,
)


class FakeBackend:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0)
        self.events = []

    def step(self):
        self.events.append(("step", self.data.time))
        self.data.time += 0.005


class LidarPublisherRecorder:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def after_step(self, *, simulation_time):
        self.calls.append(
            (
                simulation_time,
                tuple(self.backend.events),
            )
        )


class StatePublisherRecorder:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def after_step(self, *, data, simulation_time):
        self.calls.append(
            (
                data,
                simulation_time,
                tuple(self.backend.events),
            )
        )


def test_publishes_after_every_completed_physics_step():
    backend = FakeBackend()
    recorder = StatePublisherRecorder(backend)

    advance_simulation(
        backend=backend,
        state_publisher=recorder,
        substeps=4,
    )

    assert [
        call[1]
        for call in recorder.calls
    ] == [
        0.005,
        0.010,
        0.015,
        0.020,
    ]
    assert all(
        call[0] is backend.data
        for call in recorder.calls
    )

    for index, call in enumerate(
        recorder.calls,
        start=1,
    ):
        assert len(call[2]) == index


def test_rejects_nonpositive_substep_count():
    backend = FakeBackend()
    recorder = StatePublisherRecorder(backend)

    for substeps in (0, -1):
        try:
            advance_simulation(
                backend=backend,
                state_publisher=recorder,
                substeps=substeps,
            )
        except ValueError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError(
                "nonpositive substeps must be rejected"
            )


def test_lidar_runs_after_each_completed_physics_step():
    backend = FakeBackend()
    state_recorder = StatePublisherRecorder(
        backend
    )
    lidar_recorder = LidarPublisherRecorder(
        backend
    )

    advance_simulation(
        backend=backend,
        state_publisher=state_recorder,
        lidar_publisher=lidar_recorder,
        substeps=4,
    )

    assert [
        call[0]
        for call in lidar_recorder.calls
    ] == [
        0.005,
        0.010,
        0.015,
        0.020,
    ]

    for index, call in enumerate(
        lidar_recorder.calls,
        start=1,
    ):
        assert len(call[1]) == index

    assert [
        call[0]
        for call in lidar_recorder.calls
    ] == [
        call[1]
        for call in state_recorder.calls
    ]


def lidar_config(*, enabled):
    return SimpleNamespace(
        enabled=enabled,
        period_seconds=0.1,
        frame_id="mid360_link",
        range_min=0.25,
        range_max=40.0,
        torso_body_name="torso_link",
        robot_root_body_name="pelvis",
    )


def test_disabled_lidar_constructs_no_runtime():
    backend = SimpleNamespace(
        model=object(),
        data=object(),
    )

    result = build_lidar_publisher(
        node=object(),
        backend=backend,
        config=lidar_config(enabled=False),
        geometry=object(),
    )

    assert result is None


def test_enabled_lidar_binds_live_backend_and_contract(
    monkeypatch,
):
    backend = SimpleNamespace(
        model=object(),
        data=object(),
    )
    node = object()
    directions = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    class Geometry:
        @staticmethod
        def integration_ray_directions():
            return directions

    created = {}

    class FakeRayAdapter:
        def __init__(self, **kwargs):
            created["ray_adapter"] = kwargs

    class FakeSchedule:
        def __init__(self, **kwargs):
            created["schedule"] = kwargs

    class FakePublisher:
        def __init__(self, **kwargs):
            created["publisher"] = kwargs

    monkeypatch.setattr(
        adapter_node,
        "MuJoCoRayAdapter",
        FakeRayAdapter,
    )
    monkeypatch.setattr(
        adapter_node,
        "LidarPublicationSchedule",
        FakeSchedule,
    )
    monkeypatch.setattr(
        adapter_node,
        "SimulationLidarPublisher",
        FakePublisher,
    )

    result = build_lidar_publisher(
        node=node,
        backend=backend,
        config=lidar_config(enabled=True),
        geometry=Geometry,
    )

    assert isinstance(result, FakePublisher)
    assert created["ray_adapter"] == {
        "model": backend.model,
        "data": backend.data,
        "torso_body_name": "torso_link",
        "robot_root_body_name": "pelvis",
    }
    assert created["schedule"] == {
        "period_seconds": 0.1,
    }

    publisher = created["publisher"]
    assert publisher["node"] is node
    assert isinstance(
        publisher["ray_adapter"],
        FakeRayAdapter,
    )
    assert isinstance(
        publisher["schedule"],
        FakeSchedule,
    )
    assert publisher["geometry"] is Geometry
    assert publisher["directions_sensor"] == directions
    assert publisher["frame_id"] == "mid360_link"
    assert publisher["range_min"] == 0.25
    assert publisher["range_max"] == 40.0

class LocomotionRecorder:
    """Record carrier application in backend event order."""

    def __init__(self, backend):
        self.backend = backend
        self.calls = []

    def after_step(self):
        """Record one post-physics carrier application."""
        self.calls.append(self.backend.data.time)
        self.backend.events.append(
            ("locomotion", self.backend.data.time)
        )


def test_locomotion_runs_before_state_and_lidar_publication():
    """Published sensors observe the corrected carrier pose."""
    backend = FakeBackend()
    locomotion = LocomotionRecorder(backend)
    state = StatePublisherRecorder(backend)
    lidar = LidarPublisherRecorder(backend)

    advance_simulation(
        backend=backend,
        state_publisher=state,
        lidar_publisher=lidar,
        locomotion_controller=locomotion,
        substeps=2,
    )

    assert locomotion.calls == [0.005, 0.010]
    assert backend.events == [
        ("step", 0.0),
        ("locomotion", 0.005),
        ("step", 0.005),
        ("locomotion", 0.010),
    ]
    assert all(
        call[2][-1][0] == "locomotion"
        for call in state.calls
    )
    assert all(
        call[1][-1][0] == "locomotion"
        for call in lidar.calls
    )


def locomotion_config(*, enabled):
    """Return a complete planar-carrier configuration."""
    return SimpleNamespace(
        enabled=enabled,
        command_timeout_seconds=0.15,
        max_forward_mps=0.15,
        max_lateral_mps=0.01,
        max_yaw_rate_rps=0.40,
        allow_lateral=False,
        joint_name="floating_base_joint",
    )


def test_disabled_locomotion_constructs_no_runtime():
    """Default simulation remains unchanged."""
    result = build_simulation_locomotion(
        node=object(),
        backend=object(),
        config=locomotion_config(enabled=False),
        carrier_type=object(),
    )

    assert result is None


def test_enabled_locomotion_binds_backend_and_limits(
    monkeypatch,
):
    """Explicit enablement constructs carrier and controller."""
    backend = SimpleNamespace(
        mujoco=object(),
        model=object(),
        data=object(),
        harness=object(),
    )
    node = object()
    created = {}

    class FakeCarrier:
        def __init__(self, **kwargs):
            created["carrier"] = kwargs

    class FakeController:
        def __init__(self, **kwargs):
            created["controller"] = kwargs

    monkeypatch.setattr(
        adapter_node,
        "SimulationLocomotionController",
        FakeController,
    )

    result = build_simulation_locomotion(
        node=node,
        backend=backend,
        config=locomotion_config(enabled=True),
        carrier_type=FakeCarrier,
    )

    assert isinstance(result, FakeController)
    assert created["carrier"] == {
        "mujoco": backend.mujoco,
        "model": backend.model,
        "data": backend.data,
        "harness": backend.harness,
        "command_timeout_seconds": 0.15,
        "max_forward_mps": 0.15,
        "max_lateral_mps": 0.01,
        "max_yaw_rate_rps": 0.40,
        "allow_lateral": False,
        "joint_name": "floating_base_joint",
    }

    controller = created["controller"]
    assert controller["node"] is node
    assert controller["data"] is backend.data
    assert controller["carrier"].__class__ is FakeCarrier
    assert controller["allow_lateral"] is False
    assert controller[
        "velocity_limits"
    ].forward_mps == 0.15
    assert controller[
        "velocity_limits"
    ].lateral_mps == 0.01
    assert controller[
        "velocity_limits"
    ].yaw_rate_rps == 0.40
