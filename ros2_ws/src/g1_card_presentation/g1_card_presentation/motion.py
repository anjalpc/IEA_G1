"""Translate the card-owned calibration recipe into a neutral arm request."""

import math
from pathlib import Path

import yaml

from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES, validate_arm_motion


def load_card_motion(path=None):
    """Load installed behavior data; explicit paths support offline tests."""
    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = Path(get_package_share_directory('g1_card_presentation')) / \
            'config' / 'card_arm_motion.yaml'
    return yaml.safe_load(Path(path).read_text())


def card_arm_motion(profile, side="left"):
    """Build the measured right or mirrored-left Action-27 card motion."""
    if side not in ("left", "right"):
        raise ValueError("card side must be 'left' or 'right'")
    vectors = {}
    for key in ('right_rest', 'right_action27', 'mirror_sign',
                'left_companion', 'right_companion'):
        values = profile[key]
        if not isinstance(values, list) or len(values) != 7:
            raise ValueError(f'{key} must contain seven positions')
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            raise ValueError(f'{key} must contain finite numbers')
        vectors[key] = values
    right_delta = [action - rest for action, rest in zip(
        vectors['right_action27'], vectors['right_rest'])]
    if side == 'left':
        delta = [sign * value for sign, value in zip(
            vectors['mirror_sign'], right_delta)]
        target_names = ARM_JOINT_NAMES[:7]
    else:
        delta = right_delta
        target_names = ARM_JOINT_NAMES[7:]
    return validate_arm_motion({
        'version': 1,
        'preparation': dict(zip(ARM_JOINT_NAMES,
                                vectors['left_companion'] + vectors['right_companion'])),
        'target_offsets': dict(zip(target_names, delta)),
        'phase_seconds': profile['phase_seconds'],
        'recovery': profile['recovery'],
    })
