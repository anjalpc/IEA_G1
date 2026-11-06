"""Launch recording-only physical controller validation."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


RAW_COMMAND_TOPIC = "/g1/navigation/cmd_vel_raw"


def _launch_file(package_name, launch_name):
    """Resolve one package-owned launch file."""
    return PythonLaunchDescriptionSource(
        join(
            get_package_share_directory(package_name),
            "launch",
            launch_name,
        )
    )


def generate_launch_description():
    """Compose physical planning and recorded control."""
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    control_params_file = LaunchConfiguration(
        "control_params_file"
    )

    bringup_share = get_package_share_directory(
        "g1_navigation_bringup"
    )
    default_params = join(
        bringup_share,
        "config",
        "navigation.yaml",
    )
    default_control_params = join(
        bringup_share,
        "config",
        "navigation_control.yaml",
    )

    planning = IncludeLaunchDescription(
        _launch_file(
            "g1_navigation_bringup",
            "physical_planning.launch.py",
        ),
        launch_arguments={
            "map": map_file,
            "params_file": params_file,
        }.items(),
    )

    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": False},
        ],
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
            ("cmd_vel", "cmd_vel_nav"),
        ],
    )

    velocity_smoother = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": False},
        ],
        remappings=[
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
            ("cmd_vel", "cmd_vel_nav"),
            ("cmd_vel_smoothed", RAW_COMMAND_TOPIC),
        ],
    )

    control_boundary = IncludeLaunchDescription(
        _launch_file(
            "g1_navigation_bringup",
            "navigation_control.launch.py",
        ),
        launch_arguments={
            "use_sim_time": "false",
            "params_file": control_params_file,
        }.items(),
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_control_dry_run",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "autostart": True,
                "node_names": [
                    "controller_server",
                    "velocity_smoother",
                ],
            }
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            description=(
                "Full path to the saved physical occupancy-map YAML"
            ),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Full path to Nav2 parameters",
        ),
        DeclareLaunchArgument(
            "control_params_file",
            default_value=default_control_params,
            description=(
                "Recording-only navigation control parameters"
            ),
        ),
        planning,
        controller,
        velocity_smoother,
        control_boundary,
        lifecycle_manager,
    ])
