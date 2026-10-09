"""Authority boundary for the read-only physical-state probe."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HELPER = (
    PACKAGE_ROOT
    / "g1_hardware_bridge"
    / "state_probe"
    / "helper.py"
)


def test_helper_has_no_command_or_publisher_authority():
    source = HELPER.read_text(encoding="utf-8")

    forbidden = (
        "ChannelPublisher",
        "LowCmd_",
        "MotorCmd",
        "SportClient",
        "Move(",
        "StopMove(",
    )

    assert not any(token in source for token in forbidden)


def test_helper_does_not_initialize_dds_at_import_time():
    source = HELPER.read_text(encoding="utf-8")

    assert "ChannelFactoryInitialize(" not in source
    assert "ChannelSubscriber(" not in source
