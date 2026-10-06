"""Runtime coupling tests for MuJoCo state publication."""

from types import SimpleNamespace

from g1_simulation_bridge.adapter_node import (
    advance_simulation,
)


class FakeBackend:
    def __init__(self):
        self.data = SimpleNamespace(time=0.0)
        self.events = []

    def step(self):
        self.events.append(("step", self.data.time))
        self.data.time += 0.005


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
