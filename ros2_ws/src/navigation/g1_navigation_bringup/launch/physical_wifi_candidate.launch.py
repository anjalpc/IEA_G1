'''Compatibility entry point. Prefer physical_navigation_standby.launch.py.'''
from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    path = Path(__file__).absolute().with_name("physical_navigation_standby.launch.py")
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path))),
    ])
