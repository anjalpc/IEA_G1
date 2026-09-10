"""Shared native Unitree SDK runtime configuration."""

import os


DEFAULT_NETWORK_INTERFACE = "eth0"
DEFAULT_CYCLONEDDS_LIBRARY = (
    "/home/unitree/cyclonedds_ws/install/cyclonedds/lib"
)


def network_interface():
    """Return the configured physical robot network interface."""
    interface = os.environ.get(
        "G1_NETWORK_INTERFACE", DEFAULT_NETWORK_INTERFACE
    ).strip()
    if not interface:
        raise ValueError("G1_NETWORK_INTERFACE must not be empty")
    return interface


def native_dds_environment():
    """Return an environment using the robot's native CycloneDDS runtime."""
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = "0"
    library = env.get("UNITREE_CYCLONEDDS_LIB", DEFAULT_CYCLONEDDS_LIBRARY)
    if os.path.isdir(library):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{library}:{existing}" if existing else library
    return env
