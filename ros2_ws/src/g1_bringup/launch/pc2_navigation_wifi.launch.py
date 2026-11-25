'''Compatibility entry point. Prefer pc2_navigation_hardware.launch.py.'''
from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    path = Path(__file__).absolute().with_name("pc2_navigation_hardware.launch.py")
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path))),
    ])
