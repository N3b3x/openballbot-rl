"""Navigation layer for the ballbot (simulation only): obstacle map -> wavefront distance field -> smooth velocity command.

Used by eval_nav.py (evaluation) and train_dev.py --nav-frac (training with the planner in the loop). The balance controller never sees
the map; it only receives the planar velocity command this layer produces.

Map: obstacle = ground more than `thr` above the tile ground, from the privileged height map (oracle) or from sensor hit points kept in
a memory (eval only). Obstacles are inflated by `half_w` (robot half width plus margin) on a G x G grid of `cs` cells around each
world's tile; a distance-to-goal field is propagated over free cells (8-connected) every `replan` steps.

Smooth following (what makes motion to goals smooth rather than zig-zag):
  carrot   the target is the point reached by descending the distance field `carrot` metres from the robot (several cells ahead),
           not the best neighbour, so small map changes do not flip the heading
  heading  the commanded heading turns toward the carrot at most `w_max` rad/s
  speed    v = min(v_max, k_goal * goal distance, v_turn) with v_turn lowered when the heading error is large, and the speed itself
           changes at most `a_max` m/s^2; all speeds Froude-scaled per design (times f_w)
"""
import numpy as np, torch
import torch.nn.functional as Fn


class WavePlanner:
    def __init__(self, env, worlds, centre, goal, hgt=None, G=64, cs=0.05, half_w=0.22, thr=0.03, replan=20, iters=110,
                 carrot=0.35, w_max=2.0, a_max=0.6, v_max=0.3, k_goal=1.5, dt=0.01, smooth=True):
        self.env, self.w, self.G, self.cs, self.half_w, self.thr, self.replan, self.iters = env, worlds, G, cs, half_w, thr, replan, iters
        self.carrot, self.w_max, self.a_max, self.v_max, self.k_goal, self.dt, self.smooth = carrot, w_max, a_max, v_max, k_goal, dt, smooth
        self.dev = centre.device
        self.M = len(worlds)
        self.c, self.goal = centre.clone(), goal.clone()
        gi = (torch.arange(G, device=self.dev) - G / 2 + 0.5) * cs
        self.gi = gi
        self.d = None; self.t = 0
        self.head = torch.zeros(self.M, device=self.dev); self.speed = torch.zeros(self.M, device=self.dev)
        self.inited = torch.zeros(self.M, dtype=torch.bool, device=self.dev)
        self.mem_occ = None

    def set_goal(self, mask, goal, centre=None):
        self.goal = torch.where(mask[:, None], goal, self.goal)
        if centre is not None:
            self.c = torch.where(mask[:, None], centre, self.c)
        self.inited = self.inited & ~mask
        self.speed = torch.where(mask, torch.zeros_like(self.speed), self.speed)
        self.d = None      # force a replan

    def occupancy_oracle(self):
        e, G = self.env, self.G
        GX = (self.c[:, 0, None, None] + self.gi[None, None, :]).expand(self.M, G, G)
        GY = (self.c[:, 1, None, None] + self.gi[None, :, None]).expand(self.M, G, G)
        h = e.hgt(GX.reshape(self.M, -1), GY.reshape(self.M, -1)).reshape(self.M, G, G)
        return (h - e.hgt(self.c[:, 0], self.c[:, 1])[:, None, None]) > self.thr

    def occupancy_points(self, pts):
        """pts (M, P, 2) world xy of obstacle hits (1e6 = empty slot)."""
        G, cs = self.G, self.cs
        ix = ((pts[..., 0] - self.c[:, 0, None]) / cs + G / 2).floor().long(); iy = ((pts[..., 1] - self.c[:, 1, None]) / cs + G / 2).floor().long()
        ok = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
        flat = torch.where(ok, iy * G + ix, torch.zeros_like(ix))
        occ = torch.zeros(self.M, G * G, device=self.dev); occ.scatter_add_(1, flat, ok.float())
        return (occ > 0).reshape(self.M, G, G)

    def field(self, occ):
        G, cs, ar = self.G, self.cs, torch.arange(self.M, device=self.dev)
        k = int(round(self.half_w / cs)) * 2 + 1
        infl = Fn.max_pool2d(occ.float()[:, None], k, stride=1, padding=k // 2)[:, 0] > 0
        gx = ((self.goal[:, 0] - self.c[:, 0]) / cs + G / 2).floor().long().clamp(0, G - 1)
        gy = ((self.goal[:, 1] - self.c[:, 1]) / cs + G / 2).floor().long().clamp(0, G - 1)
        d = torch.full((self.M, G, G), 1e4, device=self.dev)
        d[ar, gy, gx] = 0; infl[ar, gy, gx] = False
        for _ in range(self.iters):
            nb = -Fn.max_pool2d(-d[:, None], 3, stride=1, padding=1)[:, 0] + 1.0
            d = torch.where(infl, torch.full_like(d, 1e4), torch.minimum(d, nb))
        self.d, self.infl = d, infl

    def build_static(self, centres, goals):
        """Static map (privileged oracle, training): precompute one distance field per (tile, goal) pair, K of them. Worlds then
        point at a pair with set_key; no per-step field computation. Same field as `field`, computed once."""
        M0, c0, g0 = self.M, self.c, self.goal
        self.M, self.c, self.goal = len(centres), centres, goals
        self.field(self.occupancy_oracle())
        self.cache = self.d.clone()
        self.M, self.c, self.goal = M0, c0, g0
        self.key = torch.zeros(self.M, dtype=torch.long, device=self.dev)
        self.cache_c, self.cache_g = centres.clone(), goals.clone()

    def set_key(self, mask, key):
        self.key = torch.where(mask, key, self.key)
        self.c = torch.where(mask[:, None], self.cache_c[key], self.c); self.goal = torch.where(mask[:, None], self.cache_g[key], self.goal)
        self.inited = self.inited & ~mask
        self.speed = torch.where(mask, torch.zeros_like(self.speed), self.speed)

    def sample(self, x, y):
        G, cs = self.G, self.cs
        ix = ((x - self.c[:, 0, None]) / cs + G / 2).floor().long().clamp(0, G - 1); iy = ((y - self.c[:, 1, None]) / cs + G / 2).floor().long().clamp(0, G - 1)
        if getattr(self, "cache", None) is not None:
            return self.cache[self.key[:, None], iy, ix]
        return self.d[torch.arange(self.M, device=self.dev)[:, None], iy, ix]

    def command(self, pos, f_w, occ_fn, obs_pts=None, prox=0.0):
        """pos (M, 2) ball xy; f_w (M,) Froude scale; occ_fn() -> (M, G, G) bool. Returns (M, 2) planar velocity command."""
        if getattr(self, "cache", None) is None and (self.d is None or self.t % self.replan == 0):
            self.field(occ_fn())
        self.t += 1
        eg = self.goal - pos; dg = eg.norm(dim=1)
        nh = 24
        ang = torch.arange(nh, device=self.dev) * (2 * np.pi / nh)
        cu = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
        # carrot: greedy descent of the field in steps of one cell, `carrot` metres long
        p = pos.clone()
        # escape: if the robot stands inside the inflated margin (field = 1e4 there), start the descent from the best point on rings
        # 0.1 to 0.3 m around it, otherwise the carrot never leaves the robot and the command stays zero
        inside = self.sample(pos[:, 0:1], pos[:, 1:2])[:, 0] >= 1e4
        if inside.any():
            rings = torch.tensor([0.1, 0.2, 0.3], device=self.dev)
            ring_pts = pos[:, None, None, :] + rings[None, :, None, None] * cu[None, None]          # (M, 3, nh, 2)
            rv = self.sample(ring_pts[..., 0].reshape(self.M, -1), ring_pts[..., 1].reshape(self.M, -1))
            bi = rv.argmin(1)
            p = torch.where(inside[:, None], ring_pts.reshape(self.M, -1, 2)[torch.arange(self.M, device=self.dev), bi], p)
        nstep = max(1, int(self.carrot / self.cs)) if self.smooth else 3
        for _ in range(nstep):
            cand = p[:, None, :] + self.cs * cu[None]                         # (M, nh, 2)
            v = self.sample(cand[..., 0], cand[..., 1])
            here = self.sample(p[:, 0:1], p[:, 1:2])[:, 0]
            b = v.argmin(1); better = v.gather(1, b[:, None])[:, 0] < here
            p = torch.where(better[:, None], cand[torch.arange(self.M, device=self.dev), b], p)
        to = p - pos
        reach = self.sample(p[:, 0:1], p[:, 1:2])[:, 0] < 1e4      # the carrot is on a reachable cell (the robot itself may sit inside the inflated margin and must be able to leave it)
        near_goal = dg < self.carrot
        to = torch.where(near_goal[:, None], eg, to)
        h_t = torch.atan2(to[:, 1], to[:, 0])
        self.h_target = h_t
        wrap = lambda x: torch.atan2(torch.sin(x), torch.cos(x))
        scale = torch.ones_like(dg)
        if obs_pts is not None and prox > 0:      # proximity rule (sensor maps fill in late): near a seen obstacle, slow down and allow faster turns
            dmin = (obs_pts - pos[:, None]).norm(dim=2).amin(1)
            scale = ((dmin - 0.2) / prox).clamp(0.3, 1.0)
        if self.smooth:
            h0 = torch.where(self.inited, self.head, h_t)
            wl = self.w_max * (2.0 - scale) * self.dt
            dh = wrap(h_t - h0).clamp(-wl, wl)
            self.head = wrap(h0 + dh)
            err = wrap(h_t - self.head).abs()
            v_turn = self.v_max * torch.clamp(1.0 - err / 1.2, 0.2, 1.0)
            v_t = torch.minimum(torch.minimum(self.k_goal * dg, torch.full_like(dg, self.v_max)), v_turn) * scale
            v_t = torch.where(reach & (dg > 0.05), v_t, torch.zeros_like(v_t))
            self.speed = self.speed + (v_t - self.speed).clamp(-self.a_max * self.dt, self.a_max * self.dt)
        else:
            self.head = h_t
            self.speed = torch.where(reach & (dg > 0.05), torch.minimum(self.k_goal * dg, torch.full_like(dg, self.v_max)), torch.zeros_like(dg))
        self.inited[:] = True
        sp = self.speed * f_w
        return torch.stack([sp * torch.cos(self.head), sp * torch.sin(self.head)], 1)
