"""Safety supervisor and command gate (pure Python; the ROS node only feeds them timestamps and messages).

Modes
  HOLD    startup and after disable: zero torque, controller not stepped. The robot must be held or on its stand.
  ACTIVE  balancing. Entered on an enable request only while the robot is nearly upright (tilt < engage_tilt_deg) and the state is
          fresh; the controller is reset at that moment (a fresh episode, as in training).
  FAULT   latched on tilt > tilt_cutoff_deg, a non-finite state or torque, or a stale state (no estimator update within
          state_timeout). Zero torque until clear_fault() and a new enable request.
Command gate
  Velocity commands (Nav2 /cmd_vel) expire after cmd_timeout. On expiry the desired velocity becomes zero; with hold_on_timeout the
  robot is then held at the point where it will come to rest (current position + v |v| / (2 a_brk), the motion layer's braking
  distance), using the trained goal-hold behaviour. A goal (if used) does not expire; a new velocity command cancels the hold.
"""
import math

import numpy as np

HOLD, ACTIVE, FAULT = "HOLD", "ACTIVE", "FAULT"


class SafetySupervisor:
    def __init__(self, tilt_cutoff_deg=30.0, engage_tilt_deg=3.0, state_timeout=0.05, torque_limit=None):
        self.tilt_cutoff, self.engage_tilt, self.state_timeout = float(tilt_cutoff_deg), float(engage_tilt_deg), float(state_timeout)
        self.torque_limit = None if torque_limit in (None, 0, 0.0) else float(torque_limit)
        self.mode, self.reason = HOLD, "startup"
        self._enable_req = False

    def request_enable(self):
        self._enable_req = True

    def request_disable(self, reason="disabled"):
        self._enable_req = False
        if self.mode == ACTIVE:
            self.mode, self.reason = HOLD, reason

    def clear_fault(self):
        if self.mode == FAULT:
            self.mode, self.reason = HOLD, "fault cleared"

    def update(self, tilt_deg, state_age):
        """Advance the mode machine. Returns (mode, just_enabled)."""
        fresh = state_age is not None and state_age <= self.state_timeout
        finite = tilt_deg is not None and math.isfinite(tilt_deg)
        if self.mode == ACTIVE:
            if not finite:
                self._fault("non-finite state")
            elif tilt_deg > self.tilt_cutoff:
                self._fault(f"tilt {tilt_deg:.1f} deg > cutoff {self.tilt_cutoff:.1f}")
            elif not fresh:
                self._fault("state estimate stale")
            return self.mode, False
        if self.mode == HOLD and self._enable_req:
            if fresh and finite and tilt_deg < self.engage_tilt:
                self._enable_req = False
                self.mode, self.reason = ACTIVE, "enabled"
                return self.mode, True
            self.reason = "enable pending: needs a fresh state and tilt below %.1f deg" % self.engage_tilt
        return self.mode, False

    def _fault(self, why):
        self.mode, self.reason = FAULT, why
        self._enable_req = False

    def gate_torque(self, torque):
        """Zero outside ACTIVE; clamp to the hardware limit; fault on non-finite values."""
        t = np.asarray(torque, dtype=np.float64)
        if self.mode != ACTIVE:
            return np.zeros(3)
        if not np.all(np.isfinite(t)):
            self._fault("non-finite torque")
            return np.zeros(3)
        if self.torque_limit is not None:
            t = np.clip(t, -self.torque_limit, self.torque_limit)
        return t


class CommandGate:
    def __init__(self, cmd_timeout=0.5, hold_on_timeout=True, a_brk=0.5):
        self.timeout, self.hold_on_timeout, self.a_brk = float(cmd_timeout), bool(hold_on_timeout), float(a_brk)
        self.v, self.wz, self.t_cmd = np.zeros(2), 0.0, None
        self.goal = None
        self.hold_goal = None

    def set_velocity(self, now, v_world, wz):
        self.v, self.wz, self.t_cmd = np.asarray(v_world, dtype=np.float64)[:2].copy(), float(wz), float(now)
        self.hold_goal = None
        self.goal = None           # a velocity command overrides a goal

    def set_goal(self, goal_xy):
        self.goal = None if goal_xy is None else np.asarray(goal_xy, dtype=np.float64)[:2].copy()
        self.hold_goal = None

    def reset(self):
        self.v, self.wz, self.t_cmd, self.goal, self.hold_goal = np.zeros(2), 0.0, None, None, None

    def timed_out(self, now):
        return self.t_cmd is None or (now - self.t_cmd) > self.timeout

    def get(self, now, pos, cmd_now, f=1.0):
        """(v_des, wz_des, goal) for this tick. pos: tracked position (2,); cmd_now: current shaped command (2,) for the stop point."""
        if self.goal is not None:
            return np.zeros(2), 0.0, self.goal
        if not self.timed_out(now):
            return self.v, self.wz, None
        if not self.hold_on_timeout:
            return np.zeros(2), 0.0, None
        if self.hold_goal is None:
            c = np.asarray(cmd_now, dtype=np.float64)[:2]
            self.hold_goal = np.asarray(pos, dtype=np.float64) + c * np.linalg.norm(c) / (2 * self.a_brk * f)
        return np.zeros(2), 0.0, self.hold_goal
