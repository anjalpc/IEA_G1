"""Supervised physical candidate generation; forwarding requires a separate bounded helper."""
from os.path import join
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

BASE = str(Path(__file__).absolute().parent)
NAV=join(BASE,'g1_gait_yaw_030_navigation.yaml')
CONTROL=join(BASE,'g1_yaw_probe_030_adapter.yaml')

def include(package, filename, arguments=None):
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(join(
        get_package_share_directory(package),'launch',filename)),
        launch_arguments=(arguments or {}).items())

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=['python3',join(BASE,'g1_controller_velocity_preview.py'),'--window-s','0.8'],output='screen'),
        DeclareLaunchArgument('map',default_value=join(get_package_share_directory('g1_navigation_localization'), 'maps', 'physical', 'building_map_verified_cellfix.yaml')),
        DeclareLaunchArgument('start_localization', default_value='true'),
        GroupAction(
            condition=IfCondition(LaunchConfiguration('start_localization')),
            actions=[include('g1_navigation_localization', 'localization.launch.py', {
                'map': LaunchConfiguration('map'), 'use_sim_time': 'false', 'autostart': 'true',
            })],
        ),
        Node(package='nav2_planner',executable='planner_server',name='planner_server',output='screen',parameters=[NAV]),
        Node(package='nav2_controller',executable='controller_server',name='controller_server',output='screen',parameters=[NAV],remappings=[('cmd_vel','cmd_vel_nav')]),
        Node(package='nav2_velocity_smoother',executable='velocity_smoother',name='velocity_smoother',output='screen',parameters=[NAV],remappings=[('cmd_vel','cmd_vel_nav'),('cmd_vel_smoothed','/g1/diagnostics/navigation/cmd_vel_candidate')]),
        Node(package='g1_navigation_control',executable='navigation_command_adapter',name='g1_navigation_command_adapter',output='screen',parameters=[CONTROL]),
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='lifecycle_manager_depth_recording',output='screen',parameters=[{'use_sim_time':False,'autostart':True,'node_names':['planner_server','controller_server','velocity_smoother']}]),
    ])
