"""Installation and ownership contract for the physical D435i driver."""

from pathlib import Path


PACKAGE = Path(__file__).parents[1]
SOURCE_ROOT = PACKAGE.parent
CONFIG = PACKAGE / "config" / "d435i.yaml"
CAMERA_CONFIG = PACKAGE / "config" / "camera.yaml"
SETUP = PACKAGE / "setup.py"
MANIFEST = PACKAGE / "package.xml"
START_SCRIPT = SOURCE_ROOT.parents[1] / "scripts" / "start.sh"

BRINGUP = (
    SOURCE_ROOT
    / "g1_bringup"
    / "launch"
    / "event_interaction.launch.py"
)


def compact(path):
    """Return file text without insignificant whitespace."""
    return "".join(path.read_text(encoding="utf-8").split())


def test_installs_owned_d435i_configuration():
    assert CONFIG.is_file()
    setup = compact(SETUP)

    assert '"config/d435i.yaml"' in setup
    assert "<exec_depend>realsense2_camera</exec_depend>" in compact(MANIFEST)


def test_d435i_configuration_pins_physical_stream_contract():
    configuration = compact(CONFIG)

    for contract in (
        "serial_no:'254322073626'",
        "device_type:d435i",
        "enable_depth:true",
        "enable_color:true",
        "enable_infra1:true",
        "enable_infra2:true",
        "enable_gyro:true",
        "enable_accel:true",
        "unite_imu_method:2",
        "pointcloud.enable:false",
        "enable_sync:true",
        "depth_module.profile:848x480x30",
        "rgb_camera.profile:640x480x30",
    ):
        assert contract in configuration


def test_camera_bridge_consumes_driver_color_and_depth():
    configuration = compact(CAMERA_CONFIG)

    assert "rgb_input_topic:/d435/color/image_raw" in configuration
    assert "rgb_stale_after_s:0.25" in configuration
    assert "depth_input_topic:/d435/depth/image_rect_raw" in configuration
    assert "depth_stale_after_s:0.5" in configuration


def test_startup_does_not_override_pinned_realsense_runtime():
    source = START_SCRIPT.read_text(encoding="utf-8")

    assert "REALSENSE_PYTHON_PATH" not in source
    assert "REALSENSE_LIBRARY_PATH" not in source


def test_startup_pins_pc2_wired_dds_configuration():
    source = START_SCRIPT.read_text(encoding="utf-8")

    assert (
        'PC2_DDS_CONFIGURATION="'
        '${G1_PC2_DDS_CONFIGURATION:-'
        '${PROJECT_ROOT}/livox_ws/config/'
        'cyclonedds_pc2_wired.xml}"'
    ) in source
    assert 'export CYCLONEDDS_URI="file://${PC2_DDS_CONFIGURATION}"' in source
    assert 'export ROS_LOCALHOST_ONLY="0"' in source


def test_event_bringup_supervises_the_single_d435i_owner():
    launch = compact(BRINGUP)

    assert 'package="realsense2_camera"' in launch
    assert 'executable="realsense2_camera_node"' in launch
    assert 'namespace="d435"' in launch
    assert 'name="d435"' in launch
    assert '(d435_driver,"D435idriverexited")' in launch
    assert "rgb_helper" not in launch
