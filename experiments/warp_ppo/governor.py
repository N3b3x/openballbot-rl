"""Terrain-aware speed layer ("governor") for the ballbot, simulation only.

The balance controller is unchanged (LQR + learned residual, ball arrestors). The governor sits above it and only rescales the
commanded planar velocity: it reads a height profile of the ground ahead along the command direction, finds the nearest terrain
discontinuity, and caps the speed so the robot reaches it slowly. Motivation (tracking_study_log S86): with the same controller,
obstacle falls at 0.1 m/s are 3 to 5 times lower than at 0.3 m/s, while flat ground and ramps are speed independent.

Height profile h_k (m), k = 0..6 at SCAN_FWD = 0.1 .. 1.3 m ahead, five lateral samples each, relative to the ground under the ball.
It comes either from the privileged map (oracle, upper bound) or from a sensor through a learned estimator (encoders in percep/).
Hazard at row k: second difference |h_k - 2 h_{k-1} + h_{k-2}| above `thr` in any of the three central lateral samples (h_{-1} = h_{-2} = 0,
the ground under the ball). A second difference flags steps, bumps, drops and the start and end of ramps, but not a constant slope.
Speed cap: v_lim = v_slow + gain * max(0, d - d_near) at the nearest hazard distance d (no hazard: no cap), and v_slow is held for
`hold` seconds after a hazard came within d_near (the hazard is then under or behind the ball, outside the forward profile).
All speeds are Froude-scaled per design (times f_w), like the commands themselves.
"""
import torch

FWD = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3)


class Governor:
    def __init__(self, env, v_slow=0.1, gain=0.6, d_near=0.25, thr=0.01, hold=1.0, dt=0.01):
        self.env, self.v_slow, self.gain, self.d_near, self.thr, self.hold_n = env, v_slow, gain, d_near, thr, int(hold / dt)
        self.hold = torch.zeros(env.N, dtype=torch.long, device="cuda")
        self.fwd = torch.tensor(FWD, device="cuda")
        self.scale_log = []

    def reset(self, mask):
        self.hold[mask] = 0

    def hazard_distance(self, scan):
        """scan: (N, 35) in units of 6 cm (env.scan_feat layout, row-major forward x lateral). Returns nearest hazard distance (inf if none)."""
        h = scan.reshape(-1, 7, 5)[:, :, 1:4] * 0.06
        z = torch.zeros_like(h[:, :1])
        hp = torch.cat([z, z, h], 1)                                   # rows -2, -1 (under the ball), 0..6
        dd = (hp[:, 2:] - 2 * hp[:, 1:-1] + hp[:, :-2]).abs().amax(2)  # (N, 7)
        haz = dd > self.thr
        d = torch.where(haz, self.fwd[None].expand_as(dd), torch.full_like(dd, float("inf")))
        return d.amin(1)

    def refuse(self, scan, rise_max=0.024, drop_max=0.08, look=0.5):
        """Safety mode: stop before a feature the controller cannot cross (S87: up-steps and bumps above about 2.4 cm, ledges above
        about 8 cm), drive at the commanded speed otherwise. A feature counts when its edge is within `look` metres ahead.
        Returns the stop mask; stopped worlds keep zero planar command (they stand and balance)."""
        e = self.env
        h = scan.reshape(-1, 7, 5)[:, :, 1:4] * 0.06
        hp = torch.cat([torch.zeros_like(h[:, :1]), h], 1)
        dh = hp[:, 1:] - hp[:, :-1]                                    # (N, 7, 3) rise between consecutive rows
        bad = ((dh > rise_max) | (dh < -drop_max)).any(2) & (self.fwd[None] <= look)
        stop = bad.any(1)
        if not hasattr(self, "stopped"):
            self.stopped = torch.zeros(e.N, dtype=torch.bool, device="cuda")
        self.stopped |= stop      # latch: once refused, the world stays stopped (a real robot would wait for a new command)
        e.cmd = torch.where(self.stopped[:, None], torch.cat([torch.zeros_like(e.cmd[:, :2]), e.cmd[:, 2:]], 1), e.cmd)
        self.scale_log.append(float((~self.stopped).float().mean()))
        return self.stopped

    def apply(self, scan):
        e = self.env
        d = self.hazard_distance(scan)
        near = d <= self.d_near
        self.hold = torch.where(near, torch.full_like(self.hold, self.hold_n), (self.hold - 1).clamp_min(0))
        fw = e.f_w
        vlim = (self.v_slow + self.gain * (d - self.d_near).clamp_min(0)) * fw
        vlim = torch.where(self.hold > 0, self.v_slow * fw, vlim)
        c = e.cmd[:, :2]; cn = c.norm(dim=1)
        s = torch.where(torch.isfinite(vlim) & (cn > vlim), vlim / cn.clamp_min(1e-6), torch.ones_like(cn))
        e.cmd = torch.cat([c * s[:, None], e.cmd[:, 2:]], 1)
        self.scale_log.append(float(s.mean()))
        return s
