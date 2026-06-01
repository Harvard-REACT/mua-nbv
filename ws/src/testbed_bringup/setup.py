from setuptools import setup
from glob import glob
import os

package_name = "testbed_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="karen",
    maintainer_email="karenli@g.harvard.edu",
    description="Physical testbed bringup for MUA-NBV.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "testbed_static_tf = testbed_bringup.static_tf_broadcaster:main",
            "vrpn_tf_bridge = testbed_bringup.vrpn_tf_bridge:main",
            "target_stepper = testbed_bringup.target_stepper:main",
            "cloud_capturer = testbed_bringup.cloud_capturer:main",
            "pursuer_teleporter = testbed_bringup.pursuer_teleporter:main",
            "pursuer_mover = testbed_bringup.pursuer_mover:main",
            "trajectory_predictor = testbed_bringup.trajectory_predictor:main",
            "trajectory_follower = testbed_bringup.trajectory_follower:main",
            "coordinated_trajectory_follower = testbed_bringup.coordinated_trajectory_follower:main",
            "experiment_coordinator = testbed_bringup.experiment_coordinator:main",
            "closest_candidate_coordinator = testbed_bringup.closest_candidate_coordinator:main",
            "rgb_capturer = testbed_bringup.rgb_capturer:main",
        ],
    },
)
