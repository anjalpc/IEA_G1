"""Restricted two-phase arm motion data; no vendor indices or SDK access."""

import math


ARM_JOINT_NAMES = tuple(
    f'{side}_{joint}_joint'
    for side in ('left', 'right')
    for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
                  'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('motion values must be numbers')
    if not math.isfinite(value):
        raise ValueError('motion values must be finite')
    return float(value)


def validate_arm_motion(value):
    """Return an independent, bounded-size canonical version-1 request.

    Preparation is an absolute pose blended from the acquired baseline.
    The second phase replaces selected joints with baseline-relative offsets.
    Recovery reverses through preparation, then the captured baseline.
    This is deliberately not an arbitrary trajectory language.
    """
    keys = {'version', 'preparation', 'target_offsets', 'phase_seconds',
            'recovery'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('invalid arm motion fields')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('unsupported arm motion version')
    preparation = value['preparation']
    offsets = value['target_offsets']
    if not isinstance(preparation, dict) or set(preparation) != set(ARM_JOINT_NAMES):
        raise ValueError('preparation must specify the fourteen arm joints')
    if (not isinstance(offsets, dict) or not offsets
            or not set(offsets).issubset(ARM_JOINT_NAMES)):
        raise ValueError('invalid target joints')
    durations = value['phase_seconds']
    if not isinstance(durations, list) or len(durations) != 2:
        raise ValueError('exactly two motion phases are required')
    durations = [_number(v) for v in durations]
    if any(v <= 0 for v in durations):
        raise ValueError('phase durations must be positive')
    if value['recovery'] != ['preparation', 'baseline']:
        raise ValueError('unsupported recovery path')
    return {
        'version': 1,
        'preparation': {name: _number(preparation[name]) for name in ARM_JOINT_NAMES},
        'target_offsets': {name: _number(offsets[name]) for name in ARM_JOINT_NAMES
                           if name in offsets},
        'phase_seconds': durations,
        'recovery': ['preparation', 'baseline'],
    }


def resolve_arm_motion(motion, baseline):
    """Resolve waypoints only after the executor captures its actual baseline."""
    preparation = dict(baseline)
    for name, value in motion['preparation'].items():
        preparation[name] = value
    target = dict(preparation)
    for name, offset in motion['target_offsets'].items():
        target[name] = baseline[name] + offset
    return preparation, target
