"""Navigator-only add-on; existing controller output remains candidate-only."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

BASE = Path(__file__).absolute().parent

def generate_launch_description():
    tree = Path(get_package_share_directory('nav2_bt_navigator')) / 'behavior_trees/navigate_w_replanning_time.xml'
    if not tree.is_file():
        raise RuntimeError('Installed Humble planning/following behavior tree is missing')
    return LaunchDescription([
        Node(package='nav2_bt_navigator', executable='bt_navigator',
             name='bt_navigator', output='screen', parameters=[
                 str(BASE / 'g1_gait_yaw_030_navigation.yaml'),
                 {'use_sim_time': False,
                  'global_frame': 'map', 'robot_base_frame': 'base_footprint',
                  'odom_topic': '/g1/navigation/controller/odometry',
                  'default_nav_to_pose_bt_xml': str(tree),
                  'default_nav_through_poses_bt_xml': str(BASE / 'g1_through_poses_no_recovery.xml')}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_g1_navigator_preview', output='screen',
             parameters=[{'use_sim_time': False, 'autostart': True,
                          'node_names': ['bt_navigator']}]),
    ])
