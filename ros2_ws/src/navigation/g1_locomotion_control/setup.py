from glob import glob

from setuptools import find_packages, setup


package_name = "g1_locomotion_control"


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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email="anjal-mohammed.pothuvachola@telefonica.com",
    description=(
        "Hardware-independent locomotion ownership and safety control."
    ),
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            (
                "g1_locomotion_arbiter = "
                "g1_locomotion_control.node:main"
            ),
            (
                "g1_physical_locomotion_arbiter = "
                "g1_locomotion_control.node:physical_main"
            ),
            (
                "g1_sport_helper = "
                "g1_locomotion_control.sport_helper:main"
            ),
        ],
    },
)
