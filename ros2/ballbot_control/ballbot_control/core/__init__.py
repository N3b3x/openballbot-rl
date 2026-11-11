"""Pure-Python ballbot control core (numpy only; no ROS imports). The ROS 2 nodes in ballbot_control.nodes are thin wrappers over it.

  bundle       load the controller bundle written by experiments/warp_ppo/export_policy.py
  state        RobotState: what the controller measures each tick
  estimator    IMU + wheel-encoder state estimator (attitude filter, gyro bias, Kalman filter) -> RobotState and odometry
  motion_ref   motion-reference layer and goal rule
  observation  policy observation and actor vector (history stacking, design vector)
  policy       actor forward pass (numpy, or onnxruntime)
  lqr          LQR base controller
  authority    adaptive residual authority
  controller   TrackingController: one 100 Hz tick, torques out
  safety       SafetySupervisor (HOLD / ACTIVE / FAULT) and CommandGate (command timeout, stop and hold)
"""
from .bundle import Bundle, load_bundle
from .controller import StepResult, TrackingController
from .estimator import BallbotOdometry, StateEstimator
from .safety import ACTIVE, FAULT, HOLD, CommandGate, SafetySupervisor
from .state import RobotState

__all__ = ["Bundle", "load_bundle", "TrackingController", "StepResult", "BallbotOdometry", "StateEstimator", "RobotState",
           "SafetySupervisor", "CommandGate", "HOLD", "ACTIVE", "FAULT"]
