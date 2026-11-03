"""Contracts for isolated native Unitree DDS environments."""

from g1_hardware_bridge import native_runtime


def test_native_environment_does_not_inherit_application_dds_uri(
    monkeypatch,
):
    monkeypatch.setenv(
        "CYCLONEDDS_URI",
        "file:///project/application-domain-42.xml",
    )

    environment = native_runtime.native_dds_environment()

    assert environment["ROS_DOMAIN_ID"] == "0"
    assert "CYCLONEDDS_URI" not in environment
