"""Motion-reference layer (single robot): experiments/warp_ppo/motion_ref.py MotionRef, plus the goal rule of GpuBallbot.update_goal_cmd.

Turns a desired velocity (from Nav2 /cmd_vel, a teleop, or the goal rule) into a command the balance layer can follow:
slew-limited speed (a_acc, a_brk), turn rate limited by lateral acceleration (min(w_max, a_lat / speed)), braking before reversing,
and, with a goal, a braking-curve speed cap and a gentle position hold at arrival. Speeds and accelerations scale with the design's
Froude factor f. The yaw-rate command passes through unchanged (the layer shapes only the xy velocity).
"""
import numpy as np

from .math3d import wrap_angle


def goal_velocity(goal, pos, f, cmd_vmax, gain=1.5, d_zero=0.05):
    """Training goal rule: v = unit(goal - pos) * min(gain d, cmd_vmax f), zero within d_zero of the goal."""
    e = np.asarray(goal, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    d = float(np.linalg.norm(e))
    if d < d_zero:
        return np.zeros(2)
    return e / max(d, 1e-6) * min(gain * d, cmd_vmax * f)


class MotionRef:
    def __init__(self, cfg, f):
        self.dt = float(cfg["dt"])
        self.a_acc, self.a_brk, self.a_lat, self.w_max = (float(cfg[k]) for k in ("a_acc", "a_brk", "a_lat", "w_max"))
        self.d_stop, self.d_hold, self.d_exit = (float(cfg[k]) for k in ("d_stop", "d_hold", "d_exit"))
        self.kp, self.v_hold, self.d_dead = (float(cfg[k]) for k in ("kp", "v_hold", "d_dead"))
        self.f = float(f)
        self.reset()

    def reset(self, heading=0.0, speed=0.0):
        self.s, self.th, self.hold = float(speed), float(heading), False

    def step(self, v_des, pos=None, goal=None, goal_on=False):
        """v_des (2,) desired world velocity; with goal_on, goal (2,) and pos (2,) enable arrival and hold. Returns the command (2,)."""
        dt, f = self.dt, self.f
        v_des = np.asarray(v_des, dtype=np.float64)
        sd = float(np.linalg.norm(v_des))
        hd = float(np.arctan2(v_des[1], v_des[0])) if sd > 1e-4 else self.th
        use_goal = goal is not None and goal_on
        if goal is not None:
            e = np.asarray(goal, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
            d = float(np.linalg.norm(e))
            if use_goal:
                sd = min(sd, float(np.sqrt(2 * self.a_brk * f * max(d - self.d_stop * f, 0.0))))
            enter = use_goal and d < self.d_hold * f and self.s < 0.1 * f
            leave = (not use_goal) or d > self.d_exit * f
            self.hold = (self.hold or enter) and not leave
        else:
            self.hold = False      # no goal at all: same as the simulator's goal_on False (leave)
        dh = float(wrap_angle(hd - self.th))
        sd = sd * max(np.cos(dh), 0.0)                                    # brake before turning more than 90 degrees
        wl = min(self.w_max, self.a_lat * f / max(self.s, 0.1 * f))
        self.th = float(wrap_angle(self.th + max(min(dh, wl * dt), -wl * dt)))
        self.s = max(self.s + float(np.clip(sd - self.s, -self.a_brk * f * dt, self.a_acc * f * dt)), 0.0)
        cmd = np.array([self.s * np.cos(self.th), self.s * np.sin(self.th)])
        if goal is not None and self.hold:
            ph = self.kp * e
            pn = float(np.linalg.norm(ph))
            ph = ph * min(self.v_hold * f / max(pn, 1e-9), 1.0)
            if d < self.d_dead * f:
                ph = np.zeros(2)
            cmd = ph
            self.s = float(np.linalg.norm(ph))
            if pn > 1e-6:
                self.th = float(np.arctan2(ph[1], ph[0]))
        return cmd
