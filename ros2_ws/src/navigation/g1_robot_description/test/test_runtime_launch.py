"""Launch contract for the G1 robot-state runtime."""

import importlib.util
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_PATH = (
    PACKAGE_ROOT
    / "launch"
    / "robot_state.launch.py"
)


def load_launch_module():
    spec = importlib.util.spec_from_file_location(
        "g1_robot_state_launch",
        LAUNCH_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launches_one_upstream_robot_state_publisher(
    tmp_path,
):
    share = tmp_path / "share/g1_robot_description"
    urdf = share / "urdf/g1_navigation.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text(
        "<robot name=\"contract_fixture\"/>\n",
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
            if package_name == "g1_robot_description"
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

    assert len(entities) == 2

    assert calls["arguments"] == [{
        "name": "use_sim_time",
        "default_value": "false",
        "description": (
            "Use the ROS simulation clock"
        ),
    }]

    assert len(calls["nodes"]) == 1
    node = calls["nodes"][0]

    assert node["package"] == "robot_state_publisher"
    assert node["executable"] == "robot_state_publisher"
    assert node["name"] == "g1_robot_state_publisher"
    assert node["output"] == "screen"
    assert "remappings" not in node

    assert node["parameters"] == [{
        "robot_description": (
            "<robot name=\"contract_fixture\"/>\n"
        ),
        "use_sim_time": (
            "launch_configuration",
            "use_sim_time",
        ),
        "publish_frequency": 50.0,
        "ignore_timestamp": False,
        "frame_prefix": "",
    }]
