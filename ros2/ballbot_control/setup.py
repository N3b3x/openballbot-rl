"""ament_python package: core (numpy only) + ROS 2 nodes. The exported bundle under config/bundle/ is installed into share/."""
import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ballbot_control"


def bundle_files():
    """(install dir, [files]) for every exported bundle directory."""
    out = []
    for d in sorted(glob("config/bundle/*")):
        if os.path.isdir(d):
            out.append((os.path.join("share", package_name, d), sorted(glob(os.path.join(d, "*")))))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ] + bundle_files(),
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Nebiyu Tadesse",
    description="Ballbot tracking controller and state estimator (ROS 2 bridge)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_node = ballbot_control.nodes.controller_node:main",
            "estimator_node = ballbot_control.nodes.estimator_node:main",
        ],
    },
)
