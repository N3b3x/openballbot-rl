"""Tracking controller: one 100 Hz tick of motion layer -> observation -> learned residual + LQR base -> wheel torques.

Order within a tick is the simulator's (GpuBallbot.step and the end of the previous step):
  1. motion layer      desired velocity (or goal rule) -> shaped command cmd (vx, vy, wz)
  2. observation       25-D observation with cmd, history shift, actor vector (ObservationBuilder)
  3. policy            residual action a in [-1, 1]
  4. authority         rs from the command rate, near-zero command and tilt (AdaptiveAuthority)
  5. LQR               base action u from the state and cmd (LQRController)
  6. combine           applied = clip(u + rs a, -1, 1); wheel torque = -clip(applied tau, -tau, tau)   (MuJoCo actuator ctrl, N m)
The torque is held by the motor driver until the next tick (the simulator holds it for 5 physics substeps of 2 ms).
"""
from dataclasses import dataclass

import numpy as np

from .authority import AdaptiveAuthority
from .lqr import LQRController
from .math3d import yaw_of
from .motion_ref import MotionRef, goal_velocity
from .observation import ObservationBuilder
from .policy import make_policy


@dataclass
class StepResult:
    torque: np.ndarray      # (3,) wheel motor torques, N m, sign convention of the MuJoCo actuators (joint axes in the bundle)
    applied: np.ndarray     # (3,) normalised applied action clip(u + rs a)
    action: np.ndarray      # (3,) policy residual a
    lqr: np.ndarray         # (3,) LQR base action u
    rs: float               # residual authority
    cmd: np.ndarray         # (3,) shaped command fed to the observation and the LQR
    obs: np.ndarray         # (25,) observation
    x: np.ndarray           # actor input vector
    goal_on: bool


class TrackingController:
    def __init__(self, bundle, design=None, policy=None, policy_backend="numpy", residual=True, snap_heading_at_rest=True):
        self.bundle = bundle
        self.design_name = design or bundle["designs"]["nominal"]
        d = bundle.design(self.design_name)
        self.tau, self.f = float(d["tau"]), float(d["f"])
        self.dt = bundle.dt
        lim = bundle["command_limits"]
        self.v_max, self.w_max = float(lim["cmd_vmax"]) * self.f, float(lim["cmd_wmax"])
        self.goal_gain, self.goal_d_zero = float(bundle["goal_rule"]["gain"]), float(bundle["goal_rule"]["d_zero"])
        self.obs = ObservationBuilder(bundle["observation"], self.tau, d["dvec"], self.dt)
        self.policy = policy or make_policy(bundle, policy_backend)
        if self.policy.input_dim != self.obs.actor_dim:
            raise ValueError(f"policy expects {self.policy.input_dim} inputs, observation builder makes {self.obs.actor_dim}")
        self.lqr = LQRController(bundle["lqr"], d)
        ra = bundle["action"]["rs_adapt"]
        self.auth = AdaptiveAuthority(ra, self.f) if ra is not None else None
        self.rs_fixed = float(bundle["action"]["rs"])
        self.mref = MotionRef(bundle["motion_ref"], self.f) if bundle["motion_ref"]["enabled"] else None
        self.residual = residual                    # False: LQR only (residual zeroed), a diagnostic and fallback mode
        self.snap = snap_heading_at_rest
        self.reset()

    def reset(self, heading=None, state=None):
        """Start a new 'episode' (call while the robot is held still). heading: initial motion-layer heading (rad, world); default the
        robot's yaw from state, else 0."""
        if heading is None:
            heading = yaw_of(state.quat) if state is not None else 0.0
        self.obs.reset(); self.lqr.reset()
        if self.auth is not None:
            self.auth.reset()
        if self.mref is not None:
            self.mref.reset(heading=heading)
        self._was_idle = True

    def shape(self, s, v_des, wz_des=0.0, goal=None):
        """Motion layer only: returns (cmd (3,), goal_on)."""
        goal_on = goal is not None
        if goal_on:
            v_des, wz_des = goal_velocity(goal, s.ball_pos, self.f, self.v_max / self.f, self.goal_gain, self.goal_d_zero), 0.0
        v_des = np.asarray(v_des, dtype=np.float64)
        n = float(np.linalg.norm(v_des))
        if n > self.v_max:
            v_des = v_des * (self.v_max / n)
        wz = float(np.clip(wz_des, -self.w_max, self.w_max))
        if self.mref is None:
            return np.array([v_des[0], v_des[1], wz]), goal_on
        if self.snap and self._was_idle and self.mref.s == 0.0 and not self.mref.hold and n > 1e-4:
            # starting from rest: face the desired direction at once (a holonomic robot need not turn first); training episodes
            # started the same way (motion layer reset to the heading of the first command)
            self.mref.th = float(np.arctan2(v_des[1], v_des[0]))
        self._was_idle = n <= 1e-4
        c2 = self.mref.step(v_des, s.ball_pos, goal, goal_on=goal_on)
        return np.array([c2[0], c2[1], wz]), goal_on

    def step(self, s, v_des=(0.0, 0.0), wz_des=0.0, goal=None):
        """One control tick. s: RobotState; v_des (2,) desired world-frame velocity; wz_des yaw rate; goal (2,) world xy or None."""
        cmd, goal_on = self.shape(s, v_des, wz_des, goal)
        return self.control(s, cmd, goal, goal_on)

    def control(self, s, cmd, goal=None, goal_on=False):
        """Balance layer for an already shaped command (the parity test drives this with the simulator's motion-layer output)."""
        o, x = self.obs.build(s, cmd, goal, goal_on)
        a = self.policy(x) if self.residual else np.zeros(3)
        a = np.clip(a, -1.0, 1.0)
        self.obs.commit_action(a)
        rs = self.auth(s, cmd) if self.auth is not None else self.rs_fixed
        u = self.lqr.act(s, cmd)
        applied = np.clip(u + rs * a, -1.0, 1.0)
        torque = -np.clip(applied * self.tau, -self.tau, self.tau)
        return StepResult(torque=torque, applied=applied, action=a, lqr=u, rs=float(rs), cmd=cmd, obs=o, x=x, goal_on=goal_on)
