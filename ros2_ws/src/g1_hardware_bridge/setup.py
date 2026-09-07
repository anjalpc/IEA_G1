from setuptools import find_packages, setup


package_name = "g1_hardware_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            ["config/hardware.yaml", "config/camera.yaml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email="anjal-mohammed.pothuvachola@telefonica.com",
    description="Guarded Unitree and BrainCo bridge for the physical G1.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "hardware_bridge = g1_hardware_bridge.hardware_bridge_node:main",
            (
                "camera_bridge = "
                "g1_hardware_bridge.camera.node:main"
            ),
            (
                "rgb_helper = "
                "g1_hardware_bridge.camera.rgb_helper:main"
            ),
        ],
    },
)
