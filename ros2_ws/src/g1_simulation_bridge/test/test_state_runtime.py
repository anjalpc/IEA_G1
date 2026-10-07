"""Runtime coupling tests for MuJoCo state publication."""

from types import SimpleNamespace

from g1_simulation_bridge import adapter_node
from g1_simulation_bridge.adapter_node import (
    advance_simulation,
    build_lidar_publisher,
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
