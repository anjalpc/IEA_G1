"""Capability rejection before worker startup and baseline-equivalent poses."""

import copy
import subprocess

import pytest

from g1_hardware_bridge.arm.unitree_arm import UnitreeArm, validate_motion_envelope
from g1_robot_interfaces.arm_motion import validate_arm_motion
from motion_fixture import checkpoint_motion


@pytest.mark.parametrize('change', [
    lambda m: m.update(version=2),
    lambda m: m.update(gains=[80] * 14),
    lambda m: m.update(fraction=float('nan')),
    lambda m: m.update(phase_seconds=[0.3, float('inf')]),
    lambda m: m.update(phase_seconds=[0.3, 0.9, 0.1]),
    lambda m: m.update(recovery=['baseline']),
    lambda m: m['preparation'].update(waist_yaw_joint=0.0),
    lambda m: m['target_offsets'].update(left_elbow_joint=float('nan')),
])
def test_malformed_requests_rejected(change):
    motion = checkpoint_motion()
    change(motion)
    with pytest.raises(ValueError):
        validate_arm_motion(motion)


@pytest.mark.parametrize('change', [
    lambda m: m.update(phase_seconds=[0.3, 0.8]),
    lambda m: m['target_offsets'].update(right_elbow_joint=0.0),
    lambda m: m['target_offsets'].update(left_elbow_joint=1.0),
    lambda m: m['preparation'].update(left_elbow_joint=2.0),
    lambda m: m['preparation'].update(
        left_elbow_joint=-m['preparation']['left_elbow_joint']),
    lambda m: m['target_offsets'].update(
        left_elbow_joint=-m['target_offsets']['left_elbow_joint']),
])
def test_outside_envelope_never_starts_worker(change, monkeypatch):
    """Reject excess magnitude, added joints, timing drift, and sign flips."""
    motion = checkpoint_motion()
    change(motion)
    monkeypatch.setattr(subprocess, 'Popen', lambda *_a, **_k: pytest.fail('DDS worker started'))
    arm = UnitreeArm('lo', 0.1, 'live')
    arm.prepare_motion()
    with pytest.raises(ValueError):
        arm.execute_motion(motion, 1.2)


def test_validation_copies_complete_recovery_data():
    original = checkpoint_motion()
    expected = copy.deepcopy(original)
    accepted = validate_motion_envelope(original)
    original['preparation'].clear()
    original['target_offsets'].clear()
    original['recovery'].clear()
    assert accepted == expected


@pytest.mark.parametrize(
    'execution_timeout_s',
    [0.0, -1.0, float('nan'), float('inf'), float('-inf')],
)
def test_backend_rejects_invalid_compatibility_timeout_before_worker(
        execution_timeout_s, monkeypatch):
    """Defend the physical boundary even when a client bypasses ArmClient."""
    monkeypatch.setattr(
        subprocess,
        'Popen',
        lambda *_args, **_kwargs: pytest.fail('DDS worker started'),
    )
    arm = UnitreeArm('lo', 0.1, 'live')
    arm.prepare_motion()
    with pytest.raises(ValueError, match='finite and positive'):
        arm.execute_motion(checkpoint_motion(), execution_timeout_s)
