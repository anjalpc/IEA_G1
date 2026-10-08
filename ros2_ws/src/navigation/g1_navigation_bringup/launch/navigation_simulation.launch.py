"""Launch the complete headless G1 navigation simulation."""

from os.path import join

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)


USE_SIM_TIME = "true"


def package_launch(package_name, launch_name, arguments=None):
    """Include one package-owned launch description."""
    source = PythonLaunchDescriptionSource(
        join(
            get_package_share_directory(package_name),
            "launch",
            launch_name,
        )
    )
    include = IncludeLaunchDescription(
        source,
        launch_arguments=(arguments or {}).items(),
    )
    return GroupAction(
        actions=[include],
        scoped=True,
    )


def generate_launch_description():
    """Compose the validated navigation simulation stack."""
    simulation_share = get_package_share_directory(
        "g1_simulation_bridge"
    )
    bringup_share = get_package_share_directory(
        "g1_navigation_bringup"
    )

    simulation_config = join(
        simulation_share,
        "config",
        "navigation_simulation.yaml",
    )
    control_config = join(
        bringup_share,
        "config",
        "navigation_simulation_control.yaml",
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            "G1_SIMULATION_ARGUMENTS",
            "--headless",
        ),
        package_launch(
            "g1_simulation_bridge",
            "simulation.launch.py",
            {"params_file": simulation_config},
        ),
        package_launch(
            "g1_state_estimation",
            "state_estimation.launch.py",
            {"use_sim_time": USE_SIM_TIME},
        ),
        package_launch(
            "g1_robot_description",
            "robot_state.launch.py",
            {"use_sim_time": USE_SIM_TIME},
        ),
        package_launch(
            "g1_navigation_perception",
            "perception.launch.py",
            {"use_sim_time": USE_SIM_TIME},
        ),
        package_launch(
            "g1_navigation_mapping",
            "mapping.launch.py",
            {"use_sim_time": USE_SIM_TIME},
        ),
        package_launch(
            "g1_navigation_bringup",
            "navigation.launch.py",
            {
                "use_sim_time": USE_SIM_TIME,
                "autostart": "true",
            },
        ),
        package_launch(
            "g1_navigation_bringup",
            "navigation_control.launch.py",
            {
                "use_sim_time": USE_SIM_TIME,
                "params_file": control_config,
            },
        ),
    ])
