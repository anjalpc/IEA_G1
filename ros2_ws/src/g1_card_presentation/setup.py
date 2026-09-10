from setuptools import find_packages, setup


package_name = "g1_card_presentation"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config",
         ["config/card_presentation.yaml", "config/card_arm_motion.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjal Mohammed Pothuvachola",
    maintainer_email="anjal-mohammed.pothuvachola@telefonica.com",
    description="Guarded left-hand card-presentation application.",
    license="Proprietary",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "card_presentation = g1_card_presentation.card_node:main",
            "card_handover_detector = "
            "g1_card_presentation.handover_detector:main",
        ],
    },
)
