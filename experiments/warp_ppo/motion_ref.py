"""Motion-reference layer between the planner and the balance controller (simulation only; batched over worlds on the GPU).

The balance layer (LQR + learned residual) tracks a velocity command. Feeding it raw planner or goal commands made the robot sway and
hunt (log S105): the commanded heading can turn faster than a ballbot can lean into, a goal overshoot flips the command by 180 degrees,
and nothing holds position at the goal. This layer turns a desired velocity (and optional goal) into a command a ballbot can follow:
  speed     slew-limited: accelerate at most a_acc, brake at most a_brk (m/s^2)
  heading   turn rate limited by lateral acceleration: at most min(w_max, a_lat / speed) rad/s
  reversal  a desired heading more than 90 degrees away is approached by braking first (speed scaled by cos of the heading error)
  arrival   with a goal, speed is capped by the braking curve sqrt(2 a_brk (d - d_stop)) so the robot arrives without overshoot
  hold      within d_hold and slow, a gentle position loop (gain kp, cap v_hold, dead band d_dead) holds the goal; leaves hold beyond d_exit
All speeds and accelerations scale with the design's Froude factor f_w (speeds) and are given at the nominal size.
"""
import math
import torch


def _wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class MotionRef:
    def __init__(self, N, dev="cuda", dt=0.01, a_acc=0.4, a_brk=0.5, a_lat=0.35, w_max=1.5, d_stop=0.03,
                 d_hold=0.06, d_exit=0.20, kp=0.8, v_hold=0.06, d_dead=0.015):
        self.N, self.dev, self.dt = N, dev, dt
        self.a_acc, self.a_brk, self.a_lat, self.w_max = a_acc, a_brk, a_lat, w_max
        self.d_stop, self.d_hold, self.d_exit, self.kp, self.v_hold, self.d_dead = d_stop, d_hold, d_exit, kp, v_hold, d_dead
        self.s = torch.zeros(N, device=dev)
        self.th = torch.zeros(N, device=dev)
        self.hold = torch.zeros(N, dtype=torch.bool, device=dev)

    def reset(self, mask, heading=None):
        self.s = torch.where(mask, torch.zeros_like(self.s), self.s)
        self.hold = self.hold & ~mask
        if heading is not None:
            self.th = torch.where(mask, heading, self.th)

    def step(self, v_des, f_w, pos=None, goal=None, goal_on=None):
        """v_des (N,2) desired velocity; f_w (N,) Froude speed factor; pos, goal (N,2) and goal_on (N,) bool for arrival and hold.
        Returns the command (N,2)."""
        dt, f = self.dt, f_w
        sd = v_des.norm(dim=1)
        hd = torch.atan2(v_des[:, 1], v_des[:, 0])
        hd = torch.where(sd > 1e-4, hd, self.th)
        if goal is not None:
            e = goal - pos
            d = e.norm(dim=1)
            on = goal_on if goal_on is not None else torch.ones_like(d, dtype=torch.bool)
            cap = torch.sqrt(2 * self.a_brk * f * (d - self.d_stop * f).clamp_min(0))
            sd = torch.where(on, torch.minimum(sd, cap), sd)
            enter = on & (d < self.d_hold * f) & (self.s < 0.1 * f)
            leave = ~on | (d > self.d_exit * f)
            self.hold = (self.hold | enter) & ~leave
        dh = _wrap(hd - self.th)
        sd = sd * torch.cos(dh).clamp_min(0)                                  # brake before turning more than 90 degrees
        wl = torch.minimum(torch.full_like(self.s, self.w_max), self.a_lat * f / self.s.clamp_min(0.1 * f))
        self.th = _wrap(self.th + torch.maximum(torch.minimum(dh, wl * dt), -wl * dt))
        self.s = (self.s + (sd - self.s).clamp(-self.a_brk * f * dt, self.a_acc * f * dt)).clamp_min(0)
        cmd = torch.stack([self.s * torch.cos(self.th), self.s * torch.sin(self.th)], 1)
        if goal is not None:
            ph = (self.kp * e)
            pn = ph.norm(dim=1, keepdim=True)
            ph = ph * torch.clamp(self.v_hold * f[:, None] / pn.clamp_min(1e-9), max=1.0)
            ph = torch.where((d < self.d_dead * f)[:, None], torch.zeros_like(ph), ph)
            cmd = torch.where(self.hold[:, None], ph, cmd)
            self.s = torch.where(self.hold, ph.norm(dim=1), self.s)
            self.th = torch.where(self.hold & (pn[:, 0] > 1e-6), torch.atan2(ph[:, 1], ph[:, 0]), self.th)
        return cmd
