from setuptools import setup

package_name = "mua_nbv_prediction"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "jax", "jaxlib"],
    zip_safe=True,
    maintainer="karen",
    maintainer_email="karenli@g.harvard.edu",
    description="TGPR target trajectory predictor with CV prior for MUA-NBV.",
    license="Apache-2.0",
)
