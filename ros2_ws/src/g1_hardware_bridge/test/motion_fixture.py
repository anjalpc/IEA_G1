"""Independent frozen recipe from ff44390, used only for regression tests."""

from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES


LEFT_COMPANION = [0.286, 0.218, -0.009, 0.978, 0.144, 0.006, -0.004]
RIGHT_COMPANION = [0.286, -0.218, 0.009, 0.978, -0.144, 0.006, 0.004]


def checkpoint_motion(fraction=1.0):
    """Keep the old arithmetic order so binary floating-point results match."""
    rest = [0.239587, -0.270012, 0.079531, 0.802344, -0.004820, 0.007794, 0.003641]
    action = [-0.108721, -0.063594, 0.094819, 0.182400, 0.159618, -0.339249, -0.061755]
    signs = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    delta = [sign * (end - start) for sign, end, start in zip(signs, action, rest)]
    return {
        'version': 1,
        'preparation': dict(zip(ARM_JOINT_NAMES, LEFT_COMPANION + RIGHT_COMPANION)),
        'target_offsets': dict(zip(ARM_JOINT_NAMES[:7], delta)),
        'phase_seconds': [0.3, 0.9],
        'recovery': ['preparation', 'baseline'],
        'fraction': fraction,
    }
