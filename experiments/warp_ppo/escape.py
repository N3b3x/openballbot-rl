"""Contact-based escape for a blind (or partly blind) ballbot: find obstacles by getting stuck on them, remember them, back off, re-plan.
Simulation only; batched over worlds on the GPU. It is a perception source plus a recovery behaviour, not a new controller: the
planner (nav.WavePlanner) and the balance controller are unchanged, which is the same split a ROS 2 Nav2 stack uses (a costmap layer
fed by bump events plus BackUp / Spin recovery behaviours in the behaviour tree).

State machine per robot:
  DRIVE    follow the planner. Stall check every step:
             no progress  speed along the command below `frac` of the commanded speed for `t_stall` s while commanded above `v_min`, or
             no approach  the distance to the goal has not dropped by `d_prog` in `t_prog` s, or
             tilt spike   tilt above `tilt_max` deg while driving (leaning hard into something)
           On a stall: mark the obstacle (a segment of points across the heading, `ahead` m in front of the ball, `width` m wide; the width
           grows by `widen` after each failure so repeated failures search wider), count the failure, go to BACKUP.
  BACKUP   reverse along the heading at `v_back` for `t_back` s (a smaller ball needs less; scaled by the Froude factor), then DRIVE; the
           planner sees the new obstacle points and routes around them, alternating sides naturally because the marked segment
           blocks the direct line.
  GIVE UP  after `max_fail` failures the robot stops (reported as stuck), instead of grinding forever.
Everything uses quantities a real robot has on board: commanded velocity, estimated velocity (here the simulator's, the odometry
study measures how close an on-board estimate is), tilt from the IMU, and its own pose estimate.
"""
import math
import torch


class ContactEscape:
    def __init__(self, N, dev="cuda", v_min=0.08, frac=0.3, t_stall=1.0, d_prog=0.1, t_prog=4.0, tilt_max=15.0,
                 ahead=0.2, width=0.3, widen=0.2, v_back=0.15, t_back=1.0, max_fail=6, pts=7, dt=0.01):
        self.N, self.dev, self.dt = N, dev, dt
        self.v_min, self.frac, self.t_stall, self.d_prog, self.t_prog, self.tilt_max = v_min, frac, t_stall, d_prog, t_prog, tilt_max
        self.ahead, self.width, self.widen, self.v_back, self.t_back, self.max_fail, self.pts = ahead, width, widen, v_back, t_back, max_fail, pts
        z = lambda: torch.zeros(N, device=dev)
        self.slow_t, self.back_t, self.prog_t = z(), z(), z()
        self.best_d = torch.full((N,), 1e9, device=dev)
        self.fails = torch.zeros(N, dtype=torch.long, device=dev)
        self.backing = torch.zeros(N, dtype=torch.bool, device=dev)
        self.heading = z()
        self.events = 0

    def gave_up(self):
        return self.fails >= self.max_fail

    def update(self, pos, vel, cmd, tilt_deg, goal, f_w, active):
        """pos, vel, cmd (N,2); tilt_deg, f_w (N,); goal (N,2); active (N,) bool (not finished).
        Returns (cmd_out (N,2), new_points (N, pts, 2) or None, new_mask (N,) bool)."""
        dt = self.dt
        cn = cmd.norm(dim=1)
        u = cmd / cn.clamp_min(1e-6)[:, None]
        along = (vel * u).sum(1)
        driving = active & ~self.backing & ~self.gave_up() & (cn > self.v_min * f_w)
        self.heading = torch.where(driving, torch.atan2(u[:, 1], u[:, 0]), self.heading)
        self.slow_t = torch.where(driving & (along < self.frac * cn), self.slow_t + dt, torch.zeros_like(self.slow_t))
        d = (goal - pos).norm(dim=1)
        improved = d < self.best_d - self.d_prog
        self.best_d = torch.where(improved | ~active, d, self.best_d)
        self.prog_t = torch.where(improved | ~driving, torch.zeros_like(self.prog_t), self.prog_t + dt)
        stall = driving & ((self.slow_t > self.t_stall) | (self.prog_t > self.t_prog) | (tilt_deg > self.tilt_max))
        new_pts = None
        if bool(stall.any()):
            self.events += int(stall.sum())
            hx, hy = torch.cos(self.heading), torch.sin(self.heading)
            wdt = (self.width + self.widen * self.fails.float()) * f_w
            s = torch.linspace(-0.5, 0.5, self.pts, device=self.dev)[None] * wdt[:, None]          # (N, pts)
            cx = pos[:, 0] + self.ahead * f_w * hx; cy = pos[:, 1] + self.ahead * f_w * hy
            new_pts = torch.stack([cx[:, None] - s * hy[:, None], cy[:, None] + s * hx[:, None]], -1)   # (N, pts, 2)
            self.fails = self.fails + stall.long()
            self.backing = self.backing | stall
            self.back_t = torch.where(stall, torch.zeros_like(self.back_t), self.back_t)
            self.slow_t = torch.where(stall, torch.zeros_like(self.slow_t), self.slow_t)
            self.prog_t = torch.where(stall, torch.zeros_like(self.prog_t), self.prog_t)
            self.best_d = torch.where(stall, torch.full_like(self.best_d, 1e9), self.best_d)
        self.back_t = torch.where(self.backing, self.back_t + dt, self.back_t)
        done_back = self.backing & (self.back_t > self.t_back)
        self.backing = self.backing & ~done_back
        back_cmd = -self.v_back * f_w[:, None] * torch.stack([torch.cos(self.heading), torch.sin(self.heading)], 1)
        out = torch.where(self.backing[:, None], back_cmd, cmd)
        out = torch.where(self.gave_up()[:, None], torch.zeros_like(out), out)
        return out, new_pts, stall
