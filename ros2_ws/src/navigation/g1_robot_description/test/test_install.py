"""Installed-resource contract for g1_robot_description."""

from pathlib import Path

from ament_index_python.packages import (
    get_package_share_directory,
)


def test_installs_description_and_provenance():
    share = Path(
        get_package_share_directory(
            "g1_robot_description"
        )
    )

    required = (
        share / "urdf/g1_navigation.urdf",
        share / "vendor/UNITREE_SOURCE.md",
        share / "vendor/LICENSE",
    )

    for path in required:
        assert path.is_file(), (
            f"installed resource is missing: {path}"
        )
        assert path.stat().st_size > 0
