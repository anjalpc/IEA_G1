from setuptools import find_packages, setup


package_name = "g1_robot_interfaces"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email="anjal-mohammed.pothuvachola@telefonica.com",
    description="Backend-neutral ROS command transport for G1 applications.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
)
