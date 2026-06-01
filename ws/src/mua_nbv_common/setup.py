from setuptools import setup

package_name = "mua_nbv_common"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="karen",
    maintainer_email="karenli@g.harvard.edu",
    description="Shared ROS 2 utilities for MUA-NBV nodes.",
    license="Apache-2.0",
)
