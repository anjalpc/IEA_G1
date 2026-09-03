import pytest

from g1_hardware_bridge.arm.ownership import ArmOwnership


def tracker(stale=2.0, stable=0.5):
    clock = [0.0]
    ownership = ArmOwnership(stale, stable, monotonic=lambda: clock[0])
    return ownership, clock


def test_healthy_startup_silence_is_idle():
    ownership, _ = tracker()
    state = ownership.snapshot()
    assert state["status"] == ArmOwnership.IDLE
    assert state["id"] is None
    assert ownership.require_available()["idle_stable"] is True


def test_monitor_exit_is_unavailable_even_without_a_sample():
    ownership, _ = tracker()
    ownership.mark_monitor_unavailable()
    assert ownership.snapshot()["status"] == ArmOwnership.UNAVAILABLE
    with pytest.raises(RuntimeError, match="unavailable"):
        ownership.require_available()


def test_monitor_restart_restores_startup_availability():
    ownership, _ = tracker()
    ownership.mark_monitor_unavailable()
    ownership.mark_monitor_available()
    assert ownership.snapshot()["status"] == ArmOwnership.IDLE


def test_idle_must_remain_stable_before_commands_are_allowed():
    ownership, clock = tracker()
    ownership.update(0)
    assert ownership.snapshot()["status"] == ArmOwnership.UNAVAILABLE
    clock[0] = 0.5
    assert ownership.snapshot()["status"] == ArmOwnership.IDLE
    assert ownership.require_available()["idle_stable"] is True


def test_expected_action_is_owned():
    ownership, _ = tracker()
    ownership.expect(27)
    ownership.update(27, "shake_hand", True)
    assert ownership.snapshot()["status"] == ArmOwnership.OWNED


@pytest.mark.parametrize("action_id,name", [
    (27, "shake_hand"),
    (99, "release_arm"),
    (552, "vq_shake_hand"),
    (18, "high_five"),
])
def test_unexpected_nonzero_action_is_external(action_id, name):
    ownership, _ = tracker()
    ownership.update(action_id, name, True)
    state = ownership.snapshot()
    assert state["status"] == ArmOwnership.EXTERNAL
    with pytest.raises(RuntimeError, match="external Unitree arm action"):
        ownership.require_available()


def test_different_action_overrides_local_expectation():
    ownership, _ = tracker()
    ownership.expect(27)
    ownership.update(552, "vq_shake_hand", True)
    assert ownership.snapshot()["status"] == ArmOwnership.EXTERNAL


def test_owned_action_remains_owned_while_release_feedback_transitions():
    ownership, _ = tracker()
    ownership.expect(27)
    ownership.update(27, "shake_hand", True)

    ownership.expect(99)
    state = ownership.snapshot()
    assert state["status"] == ArmOwnership.OWNED
    assert state["id"] == 27
    assert state["expected_id"] == 99

    ownership.update(99, "release_arm", False)
    assert ownership.snapshot()["status"] == ArmOwnership.OWNED


def test_external_action_during_owned_release_transition_is_external():
    ownership, _ = tracker()
    ownership.expect(27)
    ownership.update(27, "shake_hand", True)
    ownership.expect(99)
    ownership.update(552, "vq_shake_hand", True)

    assert ownership.snapshot()["status"] == ArmOwnership.EXTERNAL


def test_stale_active_action_fails_closed():
    # The native publisher only reports while an action is active or has
    # just changed; a nonzero action id going stale must still fail closed.
    ownership, clock = tracker(stale=1.0, stable=0.0)
    ownership.update(552, "vq_shake_hand", True)
    assert ownership.snapshot()["status"] == ArmOwnership.EXTERNAL
    clock[0] = 1.01
    assert ownership.snapshot()["status"] == ArmOwnership.UNAVAILABLE


def test_idle_does_not_expire_after_the_native_publisher_goes_silent():
    # The native `/arm/action/state` publisher stops sending once idle is
    # reached (observed on hardware: it does not re-publish id 0 as a
    # heartbeat). Idle must remain trusted indefinitely, or the guard would
    # permanently block every future command a few seconds after going idle.
    ownership, clock = tracker(stale=1.0, stable=0.0)
    ownership.update(0)
    assert ownership.snapshot()["status"] == ArmOwnership.IDLE
    clock[0] = 100.0
    assert ownership.snapshot()["status"] == ArmOwnership.IDLE


def test_idle_clears_previous_expectation():
    ownership, clock = tracker(stable=0.5)
    ownership.expect(99)
    ownership.update(99, "release_arm", False)
    ownership.update(0, timestamp=1.0)
    clock[0] = 1.5
    state = ownership.snapshot()
    assert state["status"] == ArmOwnership.IDLE
    assert state["expected_id"] is None
