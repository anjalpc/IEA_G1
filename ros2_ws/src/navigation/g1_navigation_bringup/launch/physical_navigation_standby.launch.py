"""Bring up laptop sensing, odometry, localization, and navigation candidates."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    OpaqueFunction,
    SetLaunchConfiguration,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include(path, **arguments):
    return GroupAction(actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path)),
        launch_arguments=arguments.items(),
    )])


def generate_launch_description():
    bringup = Path(get_package_share_directory('g1_navigation_bringup'))
    perception = Path(get_package_share_directory('g1_navigation_perception'))
    localization = Path(get_package_share_directory('g1_navigation_localization'))
    physical = bringup / 'config' / 'physical'
    base = physical / 'candidate'
    depth_executable = perception.parent.parent / 'lib' / perception.name / 'depth_obstacle_preview'

    def start(context):
        transport = LaunchConfiguration('transport').perform(context)
        if transport not in ('wifi', 'wired'):
            raise ValueError('transport must be wifi or wired')
        glim_profile = 'glim_mid360_wifi' if transport == 'wifi' else 'glim_mid360'
        controller = LaunchConfiguration('controller').perform(context)
        profiles = {'dwb': 'g1_gait_yaw_030_navigation.yaml',
                    'mppi': 'g1_mppi_navigation.yaml'}
        if controller not in profiles:
            raise ValueError('controller must be dwb or mppi')
        relay = []
        if transport == 'wifi':
            relay.append(Node(
                package='topic_tools', executable='relay', name='g1_wifi_cloud_relay',
                arguments=['/g1/transport/lidar/points_wifi', '/g1/sensors/lidar/points'],
                output='screen',
            ))
        return [
            SetEnvironmentVariable('ROS_DOMAIN_ID', '42'),
            SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '0'),
            SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
            SetEnvironmentVariable('CYCLONEDDS_URI', (physical / f'cyclonedds_laptop_{transport}.xml').as_uri()),
            *relay,
            include(bringup / 'launch' / 'physical_odometry_authority.launch.py',
                    glim_configuration=str(physical / glim_profile)),
            include(perception / 'launch' / 'depth_preview.launch.py'),
            ExecuteProcess(
                cmd=[
                    '/opt/iea-depth-gpu/bin/python', str(depth_executable),
                    '--ros-args', '--params-file',
                    str(perception / 'config' / 'depth_hand_self_filter_turning_candidate.yaml'),
                    '-p', 'compute_backend:=cuda',
                ],
                additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                output='screen',
            ),
            include(perception / 'launch' / 'perception.launch.py'),
            ExecuteProcess(
                cmd=['python3', str(base / 'g1_world_depth_costmap_bridge.py')],
                additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                output='screen',
            ),
            include(bringup / 'launch' / 'depth_costmap_preview.launch.py',
                    params_file=str(base / 'g1_filtered_combined_costmap_preview.yaml')),
            SetLaunchConfiguration(
                'localization_params', str(base / 'amcl_gantry_start.yaml'),
                condition=IfCondition(LaunchConfiguration('gantry_start')),
            ),
            include(localization / 'launch' / 'localization.launch.py',
                    map=LaunchConfiguration('map'),
                    params_file=LaunchConfiguration('localization_params'),
                    use_sim_time='false', autostart='true'),
            include(base / 'g1_gait_yaw_030_physical_nav.launch.py',
                    start_localization='false', navigation_params=str(base / profiles[controller])),
            include(base / 'g1_navigate_to_pose_preview.launch.py'),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'map', default_value=str(localization / 'maps' / 'physical' / 'building_map_verified_cellfix.yaml'),
        ),
        DeclareLaunchArgument(
            'localization_params',
            default_value=str(localization / 'config' / 'amcl.yaml'),
            description='Custom AMCL profile path.',
        ),
        DeclareLaunchArgument(
            'gantry_start', default_value='false',
            description='Use the verified AMCL seed for the physical gantry reference.',
        ),
        DeclareLaunchArgument('transport', default_value='wifi', description='wifi or wired'),
        DeclareLaunchArgument('controller', default_value='mppi', description='dwb or mppi'),
        OpaqueFunction(function=start),
    ])
