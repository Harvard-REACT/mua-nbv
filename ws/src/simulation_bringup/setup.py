from setuptools import setup
import os
from glob import glob

package_name = "simulation_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        (os.path.join("share", package_name, "models/target"), glob("models/target/*.sdf")),
        (os.path.join("share", package_name, "models/target/meshes"), glob("models/target/meshes/*")),
        (os.path.join("share", package_name, "models/pursuer"), glob("models/pursuer/*.sdf")),
        (os.path.join("share", package_name, "models/pursuer/meshes"), glob("models/pursuer/meshes/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="karen",
    maintainer_email="karenli@g.harvard.edu",
    description="Gazebo simulation bringup for MUA-NBV.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_pose_tf_bridge = simulation_bringup.sim_pose_tf_bridge:main",
            "sim_static_tf = simulation_bringup.sim_static_tf:main",
            "target_stepper = simulation_bringup.target_stepper:main",
            "trajectory_predictor = simulation_bringup.trajectory_predictor:main",
            "cloud_capturer = simulation_bringup.cloud_capturer:main",
            "pursuer_spawner = simulation_bringup.pursuer_spawner:main",
            "experiment_coordinator = simulation_bringup.experiment_coordinator:main",
        ],
    },
)
