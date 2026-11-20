"""The isolated marking preview cannot acquire command authority."""

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_preview_starts_only_costmap_and_lifecycle_manager():
    tree = ast.parse((ROOT/"launch/depth_costmap_preview.launch.py").read_text())
    packages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "IncludeLaunchDescription"
            if node.func.id == "Node":
                package = next(k.value for k in node.keywords if k.arg == "package")
                assert isinstance(package, ast.Constant)
                packages.append(package.value)
    assert sorted(packages) == ["nav2_costmap_2d", "nav2_lifecycle_manager"]


def test_preview_marks_body_relative_obstacles_without_clearing_rays():
    data = yaml.safe_load((ROOT/"config/depth_costmap_preview.yaml").read_text())
    params = data["/costmap/costmap"]["ros__parameters"]
    assert params["global_frame"] == params["robot_base_frame"] == "pelvis"
    layer = params["depth_obstacles"]
    source = layer["depth"]
    assert source["topic"] == "/g1/diagnostics/depth/costmap_input"
    assert source["marking"] is True
    assert source["clearing"] is False
    assert layer["footprint_clearing_enabled"] is False
    assert source["min_obstacle_height"] < -0.8


def test_voxel_preview_separates_marks_from_observed_clearing_rays():
    data = yaml.safe_load((ROOT/"config/depth_voxel_preview.yaml").read_text())
    layer = data["/costmap/costmap"]["ros__parameters"]["depth_obstacles"]
    assert layer["plugin"] == "nav2_costmap_2d::VoxelLayer"
    marks, rays = layer["depth"], layer["depth_rays"]
    assert marks["marking"] is True and marks["clearing"] is False
    assert rays["marking"] is False and rays["clearing"] is True
    assert rays["topic"] == "/g1/diagnostics/depth/costmap_clear"
    assert marks["sensor_frame"] == rays["sensor_frame"] == "depth_preview_camera"
    top = layer["origin_z"] + layer["z_resolution"] * layer["z_voxels"]
    assert layer["origin_z"] < 0.0 < 1.3 < top
    params = data["/costmap/costmap"]["ros__parameters"]
    assert params["global_frame"] == params["robot_base_frame"] == "depth_preview_ground"
    assert layer["z_voxels"] <= 16
