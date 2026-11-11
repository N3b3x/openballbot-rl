"""'What the robot sees': CPU ground-truth casts of the head-mounted ToF sensor for the render scripts (simulation only).

Geometry and noise follow train_dev.py exactly:
  rays      18 x 14 zones over SENSOR_FOV (55 x 42 deg), pitched SENSOR_PITCH (-25 deg), from the base origin + SENSOR_UP (0.15 m) in z,
            rotated by the sensor yaw only (head mount on a gimbal, tilt is not applied); zone index = row * 18 + col,
            row 0 = top of the view, col 0 = rightmost azimuth (negative az)
  yaw       env.sensor_yaw(): a yaw override if given (eval_nav.py --aim target), else the command direction when |cmd| > 0.05,
            else the body yaw
  range     no return or beyond 4 m reads 4 m (SENSOR_RANGE)
  noise     measured = true + N(0, 1) * depth_noise * true (depth_noise 0.005), then 2 % zone dropout which reads as max range
Casting uses mujoco.mj_multiRay on the CPU model; hits on the robot's own geoms are skipped by re-casting past them.
The 7 x 5 height grid is env.scan_feat(): SCAN_FWD x SCAN_LAT around the ball, in the sensor-yaw frame, heights relative to the
ground under the ball, divided by 6 cm. Here it is reported in cm.
"""
import numpy as np, mujoco
from PIL import Image, ImageDraw

import train_dev as T

ERR_OK = 0.01          # |measured - true| below this (m) counts as a good reading (green), above as noisy (amber)
COL = dict(ok=(0.25, 0.95, 0.45, 1.0), noisy=(1.0, 0.62, 0.08, 1.0), drop=(1.0, 0.13, 0.13, 1.0), frustum=(0.55, 0.85, 1.0, 0.85),
           ray=(0.85, 0.92, 1.0, 0.45), head=(0.55, 0.85, 1.0, 1.0))


def body_yaw(qpos):
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def sensor_yaw(qpos, cmd, override=None):
    """Same rule as GpuBallbot.sensor_yaw for one world."""
    if override is not None:
        return float(override)
    c = np.asarray(cmd[:2], float)
    return float(np.arctan2(c[1], c[0])) if np.linalg.norm(c) > 0.05 else body_yaw(qpos)


class SensorView:
    def __init__(self, m, res=(18, 14), fov=None, pitch=None, up=None, rng=None, noise=0.005, drop_p=0.02, seed=0, hmap=None):
        self.m = m
        self.w, self.h = res
        self.fov = fov if fov is not None else T.SENSOR_FOV
        self.pitch = pitch if pitch is not None else T.SENSOR_PITCH
        self.up = up if up is not None else T.SENSOR_UP
        self.rng_lim = rng if rng is not None else T.SENSOR_RANGE
        self.noise, self.drop_p = noise, drop_p
        self.rs = np.random.default_rng(seed)
        az, el = np.meshgrid((np.arange(self.w) + 0.5) / self.w * self.fov[0] - self.fov[0] / 2,
                             self.pitch + self.fov[1] / 2 - (np.arange(self.h) + 0.5) / self.h * self.fov[1])
        self.dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
        ca, ce = np.array([-1, 1, 1, -1]) * self.fov[0] / 2, self.pitch + np.array([1, 1, -1, -1]) * self.fov[1] / 2      # frustum corners
        self.corners = np.stack([np.cos(ce) * np.cos(ca), np.cos(ce) * np.sin(ca), np.sin(ce)], -1)
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")
        self.robot_geom = m.body_rootid[m.geom_bodyid] != 0
        self.hmap = None if hmap is None else np.asarray(hmap)
        import tiles as TL
        self.HALF, self.NCELL = TL.HALF, TL.NCELL

    @staticmethod
    def rot(v, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c, v[..., 2]], -1)

    def _cast(self, md, p, V):
        R = V.shape[0]
        gid, dist = np.zeros(R, np.int32), np.zeros(R)
        mujoco.mj_multiRay(self.m, md, p, V.ravel().copy(), None, 1, self.base, gid, dist, None, R, 50.0)
        for i in np.nonzero((gid >= 0) & self.robot_geom[np.maximum(gid, 0)])[0]:      # passed through a robot part: continue past it
            tot, q = dist[i], p.copy()
            for _ in range(6):
                q = p + V[i] * (tot + 1e-4)
                g1 = np.zeros(1, np.int32)
                d1 = mujoco.mj_ray(self.m, md, q, V[i].copy(), None, 1, self.base, g1)
                if g1[0] < 0:
                    tot, gid[i] = -1.0, -1; break
                tot += d1 + 1e-4; gid[i] = g1[0]
                if not self.robot_geom[g1[0]]:
                    break
            dist[i] = tot
        return dist, gid

    def read(self, md, qpos, yaw):
        """md must be mj_forward-ed at qpos. Returns a dict of per-zone arrays (R = w * h)."""
        o = np.array(qpos[0:3], float); o[2] += self.up
        V = self.rot(self.dirs, yaw)
        dist, _ = self._cast(md, o, V)
        ret = (dist >= 0) & (dist <= self.rng_lim[1])                               # a surface within range
        true = np.where(ret, dist, self.rng_lim[1]).clip(*self.rng_lim)
        meas = true + self.rs.standard_normal(true.shape) * self.noise * true
        drop = self.rs.random(true.shape) < self.drop_p
        meas = np.where(drop, self.rng_lim[1], meas.clip(*self.rng_lim))
        err = np.abs(meas - true)
        cls = np.where(~ret, 3, np.where(drop, 2, np.where(err < ERR_OK, 0, 1)))      # 0 ok, 1 noisy, 2 dropout, 3 no return
        cd, _ = self._cast(md, o, self.rot(self.corners, yaw))
        return dict(w=self.w, origin=o, yaw=yaw, dirs=V, true=true, meas=meas, drop=drop, ret=ret, err=err, cls=cls,
                    hit_true=o + V * true[:, None], hit_meas=o + V * meas[:, None], corner_dirs=self.rot(self.corners, yaw),
                    corner_d=np.where((cd >= 0) & (cd <= self.rng_lim[1]), cd, -1.0))

    # ---- 7 x 5 height grids (cm, relative to the ground under the ball) ----
    def _hgt(self, x, y):
        ix = np.clip(np.round((x + self.HALF) / (2 * self.HALF) * (self.NCELL - 1)).astype(int), 0, self.NCELL - 1)
        iy = np.clip(np.round((y + self.HALF) / (2 * self.HALF) * (self.NCELL - 1)).astype(int), 0, self.NCELL - 1)
        return self.hmap[iy, ix]

    def true_grid(self, qpos, yaw):
        f = np.array(T.SCAN_FWD)[:, None]; l = np.array(T.SCAN_LAT)[None]
        bx, by = qpos[10], qpos[11]; c, s = np.cos(yaw), np.sin(yaw)
        return (self._hgt(bx + f * c - l * s, by + f * s + l * c) - self._hgt(np.array(bx), np.array(by))) * 100.0

    def seen_grid(self, md, rd, qpos):
        """The robot's own grid: measured hit points (dropouts and no-returns removed) binned into the scan cells; median height
        above the ground at the ball's contact point (the robot stands on it, so this reference is known without a map; taken as the
        surface straight below the ball centre, which avoids the few mm of soft-contact sink of 'ball centre minus radius'). NaN = no point."""
        bc = md.geom_xpos[self.ball].copy()
        dz, _ = self._cast(md, bc, np.array([[0.0, 0.0, -1.0]]))
        g0 = bc[2] - dz[0] if dz[0] >= 0 else bc[2] - self.m.geom_size[self.ball][0]
        ok = rd["ret"] & ~rd["drop"]
        P = rd["hit_meas"][ok]
        rel = P[:, :2] - np.array([qpos[10], qpos[11]]); c, s = np.cos(rd["yaw"]), np.sin(rd["yaw"])
        fa, la = rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c
        F, L = np.array(T.SCAN_FWD), np.array(T.SCAN_LAT)
        out, cnt = np.full((len(F), len(L)), np.nan), np.zeros((len(F), len(L)), int)
        df, dl = (F[1] - F[0]) / 2, (L[1] - L[0]) / 2
        for i, fv in enumerate(F):
            for j, lv in enumerate(L):
                k = (np.abs(fa - fv) <= df) & (np.abs(la - lv) <= dl)
                cnt[i, j] = k.sum()
                if k.any():
                    out[i, j] = np.median(P[k, 2] - g0) * 100.0
        return out, cnt


# ---- 3-D scene overlay ----
def draw_3d(add, line, rd, ray_step=(4, 4), cap=1.6):
    """Frustum outline, a few sight lines (subsampled), and one dot per zone at the measured point (dropouts at the true surface).
    Overlay geoms are drawn emissive so their colours read the same in light and shadow."""
    def lit(g, em=0.6):
        if g is not None:
            g.emission = em
    o = rd["origin"]
    lit(add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], o, COL["head"]))
    ends = []
    for v, d in zip(rd["corner_dirs"], rd["corner_d"]):
        e = o + v * (d if d > 0 else cap)
        ends.append(e); lit(line(o, e, COL["frustum"], 0.0028))
    for k in range(4):
        lit(line(ends[k], ends[(k + 1) % 4], COL["frustum"], 0.0028))
    W = rd["w"]; H = len(rd["true"]) // W
    for r in range(ray_step[1] // 2, H, ray_step[1]):
        for c in range(ray_step[0] // 2, W, ray_step[0]):
            i = r * W + c
            if rd["ret"][i]:
                lit(line(o, rd["hit_meas"][i] if not rd["drop"][i] else rd["hit_true"][i], COL["ray"], 0.0012), 0.3)
    for i, k in enumerate(rd["cls"]):
        if k == 3:
            continue
        p = rd["hit_true"][i] if k == 2 else rd["hit_meas"][i]
        lit(add(mujoco.mjtGeom.mjGEOM_SPHERE, [(0.009, 0.0115, 0.0135)[k], 0, 0], p, COL[("ok", "noisy", "drop")[k]]))


# ---- 2-D insets ----
def _turbo(x):
    x = np.clip(x, 0, 1)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def _diverge(v, vmax):
    """Blue (below the ground under the ball) - grey - red (above)."""
    t = np.clip(v / vmax, -1, 1)
    mid, lo, hi = np.array([92, 98, 110.0]), np.array([40, 120, 255.0]), np.array([255, 70, 50.0])
    c = np.where(t[..., None] >= 0, mid + (hi - mid) * t[..., None], mid + (lo - mid) * (-t[..., None]))
    return c.astype(np.uint8)


def range_inset(rd, font, cell=16):
    """Measured range image (18 x 14 zones) as the sensor would report it, flipped so image-left is robot-left."""
    w = rd["w"]; h = len(rd["meas"]) // w
    meas, cls = rd["meas"].reshape(h, w)[:, ::-1], rd["cls"].reshape(h, w)[:, ::-1]
    lo, hi = 0.0, 4.0
    pad, top = 10, 30
    Wd, Hd = w * cell + 2 * pad, h * cell + top + 66
    im = Image.new("RGBA", (Wd, Hd), (10, 14, 22, 215)); d = ImageDraw.Draw(im)
    d.text((pad, 7), f"ToF range image (measured, {w} x {h})", font=font(14, "bold"), fill=(236, 241, 248))
    rgb = _turbo((meas - lo) / (hi - lo))
    for r in range(h):
        for c in range(w):
            x0, y0 = pad + c * cell, top + r * cell
            if cls[r, c] == 3:
                d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(38, 42, 52))
                d.line([x0 + 3, y0 + cell - 4, x0 + cell - 4, y0 + 3], fill=(80, 86, 100), width=1)
            else:
                d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=tuple(int(v) for v in rgb[r, c]))
            if cls[r, c] == 2:
                d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(255, 40, 40), width=2)
                d.line([x0 + 3, y0 + 3, x0 + cell - 4, y0 + cell - 4], fill=(255, 40, 40), width=2)
                d.line([x0 + 3, y0 + cell - 4, x0 + cell - 4, y0 + 3], fill=(255, 40, 40), width=2)
    y = top + h * cell + 8
    bar = _turbo(np.linspace(0, 1, w * cell))
    for i in range(w * cell):
        d.line([pad + i, y, pad + i, y + 9], fill=tuple(int(v) for v in bar[i]))
    for v in (0, 1, 2, 3, 4):
        x = pad + int(v / 4 * (w * cell - 1))
        d.text((x, y + 11), f"{v} m", font=font(11), fill=(150, 163, 182), anchor="ma")
    y += 30
    d.rectangle([pad, y + 2, pad + 10, y + 12], fill=(38, 42, 52)); d.text((pad + 15, y), "no return (> 4 m)", font=font(12), fill=(150, 163, 182))
    d.rectangle([pad + 140, y + 2, pad + 150, y + 12], outline=(255, 40, 40), width=2); d.text((pad + 155, y), "dropout (reads 4 m)", font=font(12), fill=(150, 163, 182))
    return im


def grids_inset(grids, font, vmax=6.0, cw=34, ch=25):
    """grids: list of (label, (7, 5) cm array with NaN for unseen). Forward is up, robot-left is image-left; the ball sits below row 0."""
    F, L = T.SCAN_FWD, T.SCAN_LAT
    nr, nc = len(F), len(L)
    pad, top, lab, gap = 10, 52, 38, 14
    Wd = lab + len(grids) * (nc * cw) + (len(grids) - 1) * gap + 2 * pad
    Hd = top + nr * ch + 62
    im = Image.new("RGBA", (Wd, Hd), (10, 14, 22, 215)); d = ImageDraw.Draw(im)
    d.text((pad, 7), "Ground height ahead (cm, vs ground under the ball)", font=font(14, "bold"), fill=(236, 241, 248))
    for r in range(nr):
        y = top + (nr - 1 - r) * ch
        d.text((pad + lab - 6, y + ch // 2), f"{F[r]:.1f}", font=font(11), fill=(150, 163, 182), anchor="rm")
    d.text((pad, top - 14), "m ahead", font=font(11), fill=(150, 163, 182))
    for g, (name, G) in enumerate(grids):
        x0 = pad + lab + g * (nc * cw + gap)
        d.text((x0 + nc * cw // 2, top - 6), name, font=font(12, "bold"), fill=(236, 241, 248), anchor="mb")
        for r in range(nr):
            for c in range(nc):
                v = G[r, nc - 1 - c]      # column order: +lat (robot-left) on the image left
                xa, ya = x0 + c * cw, top + (nr - 1 - r) * ch
                if np.isnan(v):
                    d.rectangle([xa, ya, xa + cw - 2, ya + ch - 2], fill=(26, 30, 40))
                    d.text((xa + cw // 2, ya + ch // 2), "?", font=font(12), fill=(90, 98, 115), anchor="mm")
                else:
                    col = tuple(int(u) for u in _diverge(np.array(v), vmax))
                    d.rectangle([xa, ya, xa + cw - 2, ya + ch - 2], fill=col)
                    d.text((xa + cw // 2, ya + ch // 2), f"{v:+.1f}" if abs(v) < 9.95 else f"{v:+.0f}", font=font(11, "mono"), fill=(255, 255, 255), anchor="mm")
        yb = top + nr * ch + 3
        d.polygon([(x0 + nc * cw // 2 - 6, yb + 10), (x0 + nc * cw // 2 + 6, yb + 10), (x0 + nc * cw // 2, yb + 2)], fill=(236, 241, 248))
    y = top + nr * ch + 18
    bar = _diverge(np.linspace(-vmax, vmax, 160), vmax)
    for i in range(160):
        d.line([pad + lab + i, y, pad + lab + i, y + 9], fill=tuple(int(v) for v in bar[i]))
    d.text((pad + lab, y + 11), f"-{vmax:.0f}", font=font(11), fill=(150, 163, 182))
    d.text((pad + lab + 80, y + 11), "0", font=font(11), fill=(150, 163, 182), anchor="ma")
    d.text((pad + lab + 160, y + 11), f"+{vmax:.0f} cm", font=font(11), fill=(150, 163, 182), anchor="ra")
    d.text((pad + lab + 178, y - 2), "?  no sensor point in the cell", font=font(11), fill=(150, 163, 182))
    d.text((pad + lab + 178, y + 12), "triangle = ball, up = sensor heading", font=font(11), fill=(150, 163, 182))
    return im


def legend_inset(font, sv, rd):
    """Colour key for the 3-D dots with this frame's zone counts, and the sensor model in one place."""
    n = np.bincount(rd["cls"], minlength=4)
    im = Image.new("RGBA", (300, 250), (10, 14, 22, 215)); d = ImageDraw.Draw(im)
    d.text((10, 7), "3-D view: one dot per zone", font=font(14, "bold"), fill=(236, 241, 248))
    rows = [((64, 242, 115), f"within 1 cm of the true surface", n[0]), ((255, 158, 20), "noisy: off by 1 cm or more", n[1]),
            ((255, 33, 33), "dropout (dot = surface it missed)", n[2]), ((60, 66, 80), "no return within 4 m (no dot)", n[3])]
    for k, (c, t, cnt) in enumerate(rows):
        y = 34 + k * 21
        d.ellipse([10, y + 2, 22, y + 14], fill=c); d.text((30, y), t, font=font(13), fill=(200, 208, 220))
        d.text((290, y), f"{cnt:3d}", font=font(13, "mono"), fill=(236, 241, 248), anchor="ra")
    y = 34 + 4 * 21 + 4
    d.line([10, y + 8, 22, y + 8], fill=(140, 217, 255), width=2); d.text((30, y), "field of view outline", font=font(13), fill=(200, 208, 220))
    y += 30
    for t in (f"ToF {sv.w} x {sv.h} zones, {np.degrees(sv.fov[0]):.0f} x {np.degrees(sv.fov[1]):.0f} deg,",
              f"pitched {np.degrees(sv.pitch):.0f} deg, {sv.up:.2f} m above the base",
              f"noise: sd {sv.noise * 100:.1f} % of range, {sv.drop_p * 100:.0f} % dropouts",
              "dots sit where the robot thinks the surface is"):
        d.text((10, y), t, font=font(12), fill=(150, 163, 182)); y += 17
    return im


class Estimator:
    """percep/tof18_gov.pt: noisy 252 zone ranges + 25-D proprioception -> 7 x 5 grid (units of 6 cm), as in eval_governor.py."""
    def __init__(self, path):
        import torch, torch.nn as nn
        self.torch = torch
        e = torch.load(path, map_location="cpu")
        self.net = nn.Sequential(nn.Linear(e["nin"], e["width"]), nn.GELU(), nn.Linear(e["width"], e["width"]), nn.GELU(), nn.Linear(e["width"], 35))
        self.net.load_state_dict(e["net"]); self.net.eval()
        self.dm, self.ds, self.pm, self.ps = e["dmean"].float(), e["dstd"].float(), e["pmean"].float(), e["pstd"].float()
        self.res = tuple(e["res"])

    def __call__(self, meas, prop):
        t = self.torch
        dpt, pr = t.tensor(meas, dtype=t.float32)[None], t.tensor(prop, dtype=t.float32)[None]
        x = t.cat([((dpt - self.dm) / self.ds).clamp(-10, 10), ((pr - self.pm) / self.ps).clamp(-6, 6)], 1)
        with t.no_grad():
            return (self.net(x)[0].numpy().reshape(7, 5) * 6.0)      # cm
