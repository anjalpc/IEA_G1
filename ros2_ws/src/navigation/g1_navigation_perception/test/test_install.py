"""Installed-resource contract for navigation perception."""

from pathlib import Path

from ament_index_python.packages import (
    get_package_share_directory,
)


def test_installs_projection_launch_and_configuration():
    share = Path(
        get_package_share_directory(
            "g1_navigation_perception"
        )
    )

    required = (
        share / "config/lidar_projection.yaml",
        share / "launch/perception.launch.py",
    )

    for path in required:
        assert path.is_file(), (
            f"installed resource is missing: {path}"
        )
        assert path.stat().st_size > 0
