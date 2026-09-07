from setuptools import find_packages, setup


package_name = "g1_handshake"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(
        exclude=["test"]
    ),
    package_data={
        "g1_handshake": [
            "models/*.task",
        ],
    },
    data_files=[
        (
            "share/ament_index/"
            "resource_index/packages",
            [
                "resource/"
                + package_name
            ],
        ),
        (
            "share/"
            + package_name,
            [
                "package.xml"
            ],
        ),
        (
            "share/"
            + package_name
            + "/config",
            [
                "config/handshake.yaml",
                "config/hand_presence.yaml",
            ],
        ),
    ],
    install_requires=[
        "setuptools"
    ],
    zip_safe=True,
    maintainer=(
        "Anjal Mohammed Pothuvachola"
    ),
    maintainer_email=(
        "anjal-mohammed.pothuvachola"
        "@telefonica.com"
    ),
    description=(
        "G1 handshake application "
        "using reusable robot interfaces."
    ),
    license="Proprietary",
    extras_require={
        "test": [
            "pytest"
        ]
    },
    entry_points={
        "console_scripts": [
            (
                "handshake_node = "
                "g1_handshake.handshake_node:main"
            ),
            (
                "hand_presence_node = "
                "g1_handshake.hand_presence_node:main"
            ),
            (
                "unitree_video_helper = "
                "g1_handshake.video."
                "unitree_video_helper:main"
            ),
        ],
    },
)
