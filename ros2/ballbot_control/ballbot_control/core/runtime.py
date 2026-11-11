"""ControlLoop: everything one 100 Hz control tick does, without ROS. The controller node only converts messages and calls tick().

  sensors (latest IMU sample, latest wheel rates) -> StateEstimator -> RobotState
  SafetySupervisor.update (tilt, state age)       -> HOLD / ACTIVE / FAULT; a fresh enable resets the controller (new episode)
  CommandGate.get (velocity / goal / timeout)     -> desired velocity, yaw rate, goal
  TrackingController.step                         -> wheel torques
  SafetySupervisor.gate_torque                    -> zero unless ACTIVE, clamp, non-finite check
"""
from dataclasses import dataclass, field

import numpy as np

from .controller import TrackingController
from .estimator import StateEstimator
from .math3d import tilt_deg, yaw_of
from .safety import ACTIVE, CommandGate, SafetySupervisor


@dataclass
class LoopConfig:
    design: str = None
    policy_backend: str = "numpy"
    residual: bool = True
    cmd_timeout: float = 0.5
    hold_on_timeout: bool = True
    cmd_vel_frame: str = "base_link"       # Nav2 controllers publish /cmd_vel in base_link; "odom" = already world-frame
    max_speed: float = 0.6                 # m/s cap on the desired speed (training range is up to 1.2 f; the study evaluated 0.3 to 0.9)
    max_yaw_rate: float = 1.0
    tilt_cutoff_deg: float = 30.0
    engage_tilt_deg: float = 3.0
    state_timeout: float = 0.05
    sensor_timeout: float = 0.05
    torque_limit: float = 0.0              # 0 = the design's actuator limit tau only
    track_point: str = None                # None = the bundle's goal point (ball_centre for goal_centre runs, else ball_body_origin)


@dataclass
class TickResult:
    torque: np.ndarray
    mode: str
    reason: str
    state: object = None
    step: object = None
    odom: dict = field(default_factory=dict)


class ControlLoop:
    def __init__(self, bundle, cfg=None, controller=None):
        self.cfg = cfg or LoopConfig()
        self.bundle = bundle
        self.ctrl = controller or TrackingController(bundle, design=self.cfg.design, policy_backend=self.cfg.policy_backend, residual=self.cfg.residual)
        d = bundle.design(self.ctrl.design_name)
        est_cfg = dict(bundle["estimator"], track_point=self.cfg.track_point or bundle.track_point)
        self.est = StateEstimator(est_cfg, d["kinematics"], bundle.dt)
        self.sup = SafetySupervisor(self.cfg.tilt_cutoff_deg, self.cfg.engage_tilt_deg, self.cfg.state_timeout, self.cfg.torque_limit)
        self.gate = CommandGate(self.cfg.cmd_timeout, self.cfg.hold_on_timeout, float(bundle["motion_ref"]["a_brk"]))
        self.state, self.t_sensor, self.last_cmd2 = None, None, np.zeros(2)

    # ---- inputs (call from subscriber callbacks)
    def on_cmd_vel(self, now, vx, vy, wz):
        v = np.array([vx, vy], dtype=np.float64)
        if self.cfg.cmd_vel_frame == "base_link" and self.state is not None:
            yaw = yaw_of(self.state.quat)
            c, s = np.cos(yaw), np.sin(yaw)
            v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        n = np.linalg.norm(v)
        if n > self.cfg.max_speed:
            v *= self.cfg.max_speed / n
        self.gate.set_velocity(now, v, float(np.clip(wz, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)))

    def on_goal(self, goal_xy):
        self.gate.set_goal(goal_xy)

    def enable(self):
        self.sup.request_enable()

    def disable(self):
        self.sup.request_disable()

    # ---- the tick
    def tick(self, now, gyro, acc, wheel_rate, t_sensor):
        """now: current time (s); gyro, acc: latest IMU sample (body frame); wheel_rate: latest encoder rates; t_sensor: time of the
        older of the two latest sensor samples (None if none yet). Returns TickResult with the torques to send."""
        odom = {}
        if t_sensor is not None and (now - t_sensor) <= self.cfg.sensor_timeout:
            self.state, odom = self.est.step(wheel_rate, gyro, acc)
            self.t_sensor = t_sensor
        age = None if self.t_sensor is None else now - self.t_sensor
        tilt = tilt_deg(self.state.quat) if self.state is not None else None
        mode, just_enabled = self.sup.update(tilt, age)
        if just_enabled:
            self.ctrl.reset(state=self.state)
            self.gate.hold_goal = None      # a stale hold point from before; a fresh command or goal stays valid
            self.last_cmd2 = np.zeros(2)
        if mode != ACTIVE:
            return TickResult(np.zeros(3), mode, self.sup.reason, self.state, None, odom)
        v_des, wz, goal = self.gate.get(now, self.state.ball_pos, self.last_cmd2, self.ctrl.f)
        r = self.ctrl.step(self.state, v_des, wz, goal)
        self.last_cmd2 = r.cmd[:2]
        torque = self.sup.gate_torque(r.torque)
        return TickResult(torque, self.sup.mode, self.sup.reason, self.state, r, odom)
