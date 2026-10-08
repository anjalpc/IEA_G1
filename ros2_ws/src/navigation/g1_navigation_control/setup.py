from setuptools import find_packages, setup


package_name = "g1_navigation_control"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email=(
        "anjal-mohammed.pothuvachola@telefonica.com"
    ),
    description=(
        "Lease-protected Nav2 command adapter for the G1."
    ),
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigation_command_adapter=g1_navigation_control.node:main",
        ],
    },
)
