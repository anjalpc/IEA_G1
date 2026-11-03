from glob import glob

from setuptools import find_packages, setup


package_name = "g1_state_estimation"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml"),
        ),
        (
            "share/" + package_name + "/launch",
            glob("launch/*.launch.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email="anjal-mohammed.pothuvachola@telefonica.com",
    description=(
        "Hardware-independent state validation and navigation readiness."
    ),
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "state_estimation_node = "
            "g1_state_estimation.node:main",
            "glim_health_node = "
            "g1_state_estimation.glim_node:main",
            "glim_planar_node = "
            "g1_state_estimation.glim_planar_node:main",
            "physical_body_node = "
            "g1_state_estimation.physical_body_node:main",
            "physical_authority_node = "
            "g1_state_estimation.physical_authority_node:main",
        ],
    },
)
