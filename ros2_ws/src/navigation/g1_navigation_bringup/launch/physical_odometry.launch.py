"""Launch guarded laptop-side MID-360 odometry."""

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


def generate_launch_description():
    """Launch GLIM behind the project health boundary."""
    glim_configuration = LaunchConfiguration(
        "glim_configuration"
    )
    health_parameters = LaunchConfiguration(
        "health_parameters"
    )
    body_parameters = LaunchConfiguration(
        "body_parameters"
    )

    bringup_share = get_package_share_directory(
        "g1_navigation_bringup"
    )
    state_estimation_share = get_package_share_directory(
        "g1_state_estimation"
    )
    robot_description_share = get_package_share_directory(
        "g1_robot_description"
    )

    default_glim_configuration = join(
        bringup_share,
        "config",
        "physical",
        "glim_mid360",
    )
    default_health_parameters = join(
        state_estimation_share,
        "config",
        "glim_health.yaml",
    )
    default_body_parameters = join(
        state_estimation_share,
        "config",
        "physical_body.yaml",
    )

    robot_state = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            join(
                robot_description_share,
                "launch",
                "physical_robot_state.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": "false",
        }.items(),
    )
    glim = Node(
        package="glim_ros",
        executable="glim_rosnode",
        output="screen",
        parameters=[
            {
                "config_path": glim_configuration,
            }
        ],
    )
    health_boundary = Node(
        package="g1_state_estimation",
        executable="glim_health_node",
        name="g1_glim_health",
        output="screen",
        parameters=[health_parameters],
    )
    planar_staging = Node(
        package="g1_state_estimation",
        executable="glim_planar_node",
        name="g1_glim_planar",
        output="screen",
    )
    body_staging = Node(
        package="g1_state_estimation",
        executable="physical_body_node",
        name="g1_physical_body",
        output="screen",
        parameters=[body_parameters],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "glim_configuration",
            default_value=default_glim_configuration,
            description=(
                "Project-owned MID-360 GLIM configuration directory"
            ),
        ),
        DeclareLaunchArgument(
            "health_parameters",
            default_value=default_health_parameters,
            description="GLIM health-boundary parameters",
        ),
        DeclareLaunchArgument(
            "body_parameters",
            default_value=default_body_parameters,
            description="Physical body-staging parameters",
        ),
        robot_state,
        glim,
        health_boundary,
        planar_staging,
        body_staging,
    ])
