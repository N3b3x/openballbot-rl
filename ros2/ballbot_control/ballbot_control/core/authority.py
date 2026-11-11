"""Adaptive residual authority (study S111/S112; the rs_adapt block of GpuBallbot.step).

rs = lo + (hi - lo) * w, where w rises at once to the current need and otherwise decays by a factor 0.98 per tick (0.5 s time
constant at 100 Hz). need = max of
  command rate     |d cmd_xy / dt| / 0.3 m/s^2, clipped to [0, 1]
  near-zero cmd    1 - |cmd_xy| / (0.1 f), clipped to [0, 1], counted only while the tracked ball point moves faster than v_db f
  tilt             (tilt - 4 deg) / 6 deg, clipped to [0, 1]
The applied action is clip(lqr + rs * a, -1, 1).
"""
import numpy as np

from .math3d import tilt_deg


class AdaptiveAuthority:
    def __init__(self, cfg, f):
        self.lo, self.hi, self.v_db = float(cfg["lo"]), float(cfg["hi"]), float(cfg["v_db"])
        self.dc_scale, self.near0_frac = float(cfg["dc_scale"]), float(cfg["near0_frac"])
        self.tilt0, self.tilt_span, self.decay = float(cfg["tilt0_deg"]), float(cfg["tilt_span_deg"]), float(cfg["decay"])
        self.w_init, self.dt, self.f = float(cfg["w_init"]), float(cfg["dt"]), float(f)
        self.w = self.w_init
        self.prev_cmd = None
        # The simulator never resets w or the previous command between episodes; reset() is provided for a controller (re)start and
        # starts at full authority (w_init = 1, the simulator's initial value).

    def reset(self):
        self.w = self.w_init
        self.prev_cmd = None

    def __call__(self, s, cmd):
        c2 = np.asarray(cmd[:2], dtype=np.float64)
        dc = 0.0 if self.prev_cmd is None else float(np.linalg.norm(c2 - self.prev_cmd)) / self.dt
        self.prev_cmd = c2.copy()
        tilt = tilt_deg(s.quat)
        near0 = float(np.clip(1.0 - np.linalg.norm(c2) / (self.near0_frac * self.f), 0.0, 1.0))
        if self.v_db > 0:
            near0 *= float(np.linalg.norm(np.asarray(s.ball_vel, dtype=np.float64)) > self.v_db * self.f)
        need = max(float(np.clip(dc / self.dc_scale, 0.0, 1.0)), near0, float(np.clip((tilt - self.tilt0) / self.tilt_span, 0.0, 1.0)))
        self.w = max(need, self.w * self.decay)
        return self.lo + (self.hi - self.lo) * self.w
