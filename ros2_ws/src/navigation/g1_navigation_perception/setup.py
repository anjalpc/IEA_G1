from glob import glob

from setuptools import find_packages, setup


package_name = "g1_navigation_perception"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
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
    maintainer_email=(
        "anjal-mohammed.pothuvachola@telefonica.com"
    ),
    description=(
        "Workstation navigation perception launch and configuration."
    ),
    license="Proprietary",
    extras_require={"test": ["pytest"]},
)
