from setuptools import find_packages, setup


package_name = "g1_handshake"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/handshake.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email="anjal-mohammed.pothuvachola@telefonica.com",
    description="G1 handshake application using reusable robot interfaces.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "handshake_node = g1_handshake.handshake_node:main",
        ],
    },
)
