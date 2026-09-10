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


def card_arm_motion(profile):
    """Preserve the original mirrored Action-27 delta and bilateral preparation."""
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
    delta = [sign * (action - rest) for sign, action, rest in zip(
        vectors['mirror_sign'], vectors['right_action27'], vectors['right_rest'])]
    return validate_arm_motion({
        'version': 1,
        'preparation': dict(zip(ARM_JOINT_NAMES,
                                vectors['left_companion'] + vectors['right_companion'])),
        'target_offsets': dict(zip(ARM_JOINT_NAMES[:7], delta)),
        'phase_seconds': profile['phase_seconds'],
        'recovery': profile['recovery'],
    })
