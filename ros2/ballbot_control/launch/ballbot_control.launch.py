"""Launch the ballbot tracking controller and state estimator.

  ros2 launch ballbot_control ballbot_control.launch.py [bundle:=/path/to/bundle.json] [params:=/path/to/params.yaml]
      [imu_topic:=/imu] [joint_states_topic:=/joint_states] [cmd_vel_topic:=/cmd_vel] [torque_topic:=/wheel_torque_cmd]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("ballbot_control")
    # recommended controller: g39 (goal logic on the ball centre); the g34 bundle is installed alongside:
    #   bundle:=$(ros2 pkg prefix ballbot_control)/share/ballbot_control/config/bundle/g34_mref_rsad_db_s0/bundle.json
    default_bundle = os.path.join(share, "config", "bundle", "g39_goalcentre_s0", "bundle.json")
    default_params = os.path.join(share, "config", "ballbot_control.yaml")
    args = [
        DeclareLaunchArgument("bundle", default_value=default_bundle),
        DeclareLaunchArgument("params", default_value=default_params),
        DeclareLaunchArgument("imu_topic", default_value="/imu"),
        DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("torque_topic", default_value="/wheel_torque_cmd"),
    ]
    L = LaunchConfiguration
    sensors = [("imu", L("imu_topic")), ("joint_states", L("joint_states_topic"))]
    controller = Node(package="ballbot_control", executable="controller_node", name="ballbot_controller", output="screen",
                      parameters=[L("params"), {"bundle_path": L("bundle")}],
                      remappings=sensors + [("cmd_vel", L("cmd_vel_topic")), ("wheel_torque_cmd", L("torque_topic"))])
    estimator = Node(package="ballbot_control", executable="estimator_node", name="ballbot_estimator", output="screen",
                     parameters=[L("params"), {"bundle_path": L("bundle")}], remappings=sensors)
    return LaunchDescription(args + [controller, estimator])
