"""Runtime launch contract for G1 asynchronous mapping."""

import importlib.util
from pathlib import Path

from ament_index_python.packages import (
    get_package_share_directory,
)


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_PATH = (
    PACKAGE_ROOT
    / "launch"
    / "mapping.launch.py"
)


def load_launch_module():
    spec = importlib.util.spec_from_file_location(
        "g1_navigation_mapping_launch",
        LAUNCH_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cmake_registers_runtime_launch_test():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "test_runtime_launch" in cmake
    assert "test/test_runtime_launch.py" in cmake


def test_installs_mapping_launch_and_configuration():
    share = Path(
        get_package_share_directory(
            "g1_navigation_mapping"
        )
    )

    required = (
        share / "config/online_async.yaml",
        share / "launch/mapping.launch.py",
    )

    for path in required:
        assert path.is_file(), (
            f"installed resource is missing: {path}"
        )
        assert path.stat().st_size > 0


def test_constructs_one_upstream_async_mapping_node(
    tmp_path,
):
    share = tmp_path / "share/g1_navigation_mapping"
    configuration = share / "config/online_async.yaml"
    configuration.parent.mkdir(parents=True)
    configuration.write_text(
        "slam_toolbox:\n"
        "  ros__parameters:\n"
        "    mode: mapping\n",
        encoding="utf-8",
    )

    module = load_launch_module()
    calls = {
        "arguments": [],
        "nodes": [],
    }

    module.get_package_share_directory = (
        lambda package_name: (
            str(share)
            if package_name == "g1_navigation_mapping"
            else None
        )
    )
    module.LaunchConfiguration = (
        lambda name: ("launch_configuration", name)
    )

    def record_argument(name, **kwargs):
        call = {
            "name": name,
            **kwargs,
        }
        calls["arguments"].append(call)
        return ("argument", call)

    def record_node(**kwargs):
        calls["nodes"].append(kwargs)
        return ("node", kwargs)

    module.DeclareLaunchArgument = record_argument
    module.Node = record_node
    module.LaunchDescription = lambda entities: entities

    entities = module.generate_launch_description()

    assert len(entities) == 3
    assert calls["arguments"] == [
        {
            "name": "use_sim_time",
            "default_value": "false",
            "description": "Use the ROS simulation clock",
        },
        {
            "name": "slam_params_file",
            "default_value": str(configuration),
            "description": (
                "Full path to the SLAM Toolbox parameter file"
            ),
        },
    ]

    assert len(calls["nodes"]) == 1
    node = calls["nodes"][0]

    assert node["package"] == "slam_toolbox"
    assert node["executable"] == (
        "async_slam_toolbox_node"
    )
    assert node["name"] == "slam_toolbox"
    assert node["output"] == "screen"
    assert "remappings" not in node
    assert node["parameters"] == [
        (
            "launch_configuration",
            "slam_params_file",
        ),
        {
            "use_sim_time": (
                "launch_configuration",
                "use_sim_time",
            ),
        },
    ]
