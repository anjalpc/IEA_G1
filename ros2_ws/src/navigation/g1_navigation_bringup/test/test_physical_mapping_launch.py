"""Contract for read-only physical mapping bringup."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_FILE = (
    PACKAGE_ROOT / "launch/physical_mapping.launch.py"
)


def launch_constants():
    """Return string constants from physical mapping launch."""

    assert LAUNCH_FILE.is_file(), str(LAUNCH_FILE)
    tree = ast.parse(
        LAUNCH_FILE.read_text(encoding="utf-8")
    )
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    }


def test_composes_authority_perception_and_mapping():
    values = launch_constants()

    assert {
        "g1_navigation_bringup",
        "physical_odometry_authority.launch.py",
        "g1_navigation_perception",
        "perception.launch.py",
        "g1_navigation_mapping",
        "mapping.launch.py",
    } <= values


def test_uses_wall_clock_for_physical_mapping():
    values = launch_constants()

    assert "use_sim_time" in values
    assert "false" in values


def test_starts_no_localization_navigation_or_commands():
    values = launch_constants()

    forbidden = {
        "localization.launch.py",
        "navigation.launch.py",
        "navigation_control.launch.py",
        "amcl",
        "nav2_controller",
        "g1_locomotion_arbiter",
        "cmd_vel",
    }
    assert forbidden.isdisjoint(values)


def test_cmake_registers_physical_mapping_contract():
    cmake = (
        PACKAGE_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "test_physical_mapping_launch" in cmake
    assert "test/test_physical_mapping_launch.py" in cmake


def test_wifi_mapping_composes_one_relay_and_existing_mapping():
    """Wi-Fi mapping selects tested profiles without command nodes."""
    import importlib.util
    path = PACKAGE_ROOT / "launch/physical_wifi_mapping.launch.py"
    assert path.is_file(), str(path)
    spec = importlib.util.spec_from_file_location("wifi_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {"nodes": [], "includes": [], "environment": {}}
    module.get_package_share_directory = lambda name: str(PACKAGE_ROOT)
    module.Node = lambda **kw: calls["nodes"].append(kw)
    module.IncludeLaunchDescription = lambda source, **kw: calls["includes"].append((source, kw))
    module.PythonLaunchDescriptionSource = lambda path: path
    module.SetEnvironmentVariable = lambda name, value: calls["environment"].update({name: value})
    module.LaunchDescription = lambda entities: entities
    module.generate_launch_description()
    assert calls["nodes"] == [{
        "package": "topic_tools", "executable": "relay",
        "name": "g1_wifi_cloud_relay", "output": "screen",
        "arguments": ["/g1/transport/lidar/points_wifi", "/g1/sensors/lidar/points"]
    }]
    assert len(calls["includes"]) == 1
    source, options = calls["includes"][0]
    assert source == str(PACKAGE_ROOT / "launch/physical_mapping.launch.py")
    assert dict(options["launch_arguments"])["glim_configuration"] == str(
        PACKAGE_ROOT / "config/physical/glim_mid360_wifi"
    )
    assert calls["environment"]["ROS_DOMAIN_ID"] == "42"
    assert calls["environment"]["CYCLONEDDS_URI"] == (
        "file://" + str(PACKAGE_ROOT / "config/physical/cyclonedds_laptop_wifi.xml")
    )
