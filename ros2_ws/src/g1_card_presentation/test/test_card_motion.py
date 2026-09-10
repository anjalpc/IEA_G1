"""Recipe equivalence against the pre-extraction physical implementation."""

from pathlib import Path

from g1_card_presentation.card_backends import CardArm
from g1_card_presentation.motion import card_arm_motion, load_card_motion
from g1_robot_interfaces.arm_motion import ARM_JOINT_NAMES, resolve_arm_motion


def profile():
    return load_card_motion(Path(__file__).resolve().parents[1] /
                            'config' / 'card_arm_motion.yaml')


def test_recipe_matches_checkpoint_waypoints():
    # Frozen ff44390 values, independent of the YAML under test.
    left = [0.286, 0.218, -0.009, 0.978, 0.144, 0.006, -0.004]
    right = [0.286, -0.218, 0.009, 0.978, -0.144, 0.006, 0.004]
    rest = [0.239587, -0.270012, 0.079531, 0.802344, -0.004820, 0.007794, 0.003641]
    action = [-0.108721, -0.063594, 0.094819, 0.182400, 0.159618, -0.339249, -0.061755]
    signs = [1, -1, -1, 1, -1, 1, -1]
    motion = card_arm_motion(profile())
    baseline = {name: 0.01 * index
                for name, index in zip(ARM_JOINT_NAMES, range(15, 29))}
    preparation, target = resolve_arm_motion(motion, baseline)
    expected = dict(zip(ARM_JOINT_NAMES, left + right))
    assert preparation == expected
    for name, sign, end, start in zip(ARM_JOINT_NAMES, signs, action, rest):
        expected[name] = baseline[name] + sign * (end - start)
    assert target == expected
    assert motion['phase_seconds'] == [0.3, 0.9]
    assert motion['recovery'] == ['preparation', 'baseline']


def test_adapter_sends_data_without_card_name_and_preserves_timeout_input():
    class Client:
        def execute_motion(self, motion, execution_timeout_s):
            return motion, execution_timeout_s

    arm = CardArm(Client(), profile())
    motion, timeout_input = arm.present(8.0)
    assert motion == card_arm_motion(profile())
    assert timeout_input == 8.0
    assert 'card' not in str(motion)
