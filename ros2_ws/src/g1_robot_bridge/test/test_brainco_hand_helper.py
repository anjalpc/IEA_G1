import pytest

from g1_robot_bridge.brainco_hand_helper import command_position, plan_curl


def test_plan_curl_clamps_to_conservative_limit():
    assert plan_curl(1.0) == [0.1, 0.0625, 0.25, 0.25, 0.25, 0.25]
    assert plan_curl(-1.0) == [0.0] * 6


@pytest.mark.parametrize("amount", [-0.01, 0.151])
def test_command_position_rejects_unsafe_amount_before_loading_sdk(amount):
    with pytest.raises(ValueError, match="between 0 and 0.15"):
        command_position(
            "eth0", "right", amount, 1.5, 5.0,
            "HANDSHAKE_OPERATOR_ENABLED")


def test_command_position_requires_explicit_interlock():
    with pytest.raises(ValueError, match="requires --confirm"):
        command_position("eth0", "right", 0.1, 1.5, 5.0, "")
