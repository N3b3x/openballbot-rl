"""Cinematic, information-rich renderer for the ballbot study (simulation only).

Pipeline: simulate once at 100 Hz and log everything (qpos, command, velocity, tracking error, pushes, falls), then replay
the log through the MuJoCo offscreen renderer (1280x720, 30 fps) with an eased chase/raking camera and compose a HUD with PIL.
The floor texture is repainted at runtime from the height field: height tint, hill shading, 1 cm contour lines (5 cm
lines emphasised) and an amber tint on cliffs, so flat / curved / bumpy / stepped ground is readable at a glance.
The model XML is copied to a temporary file with a 2048 px floor texture, a raking sun with shadows and a 1280x720 offscreen
buffer; ballbot.xml itself is never modified.

  python scripts/study/render_cinematic.py --job perlin8_tour_res_c
  python scripts/study/render_cinematic.py --job climb3cm_robs_c --sim-limit 5 --dump 2,4 --no-video
  python scripts/study/render_cinematic.py --compare perlin8_tour_lqr perlin8_tour_res_c
  python scripts/study/render_cinematic.py --list
Outputs go to outputs/study/cine/: cine_<scenario>_<controller>.mp4, .jpg poster, .json sidecar.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import subprocess
import sys
import time as _time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))
import render_control as RC  # noqa: E402  (scenario tables SEQS / WAYPOINTS; torch + env imports)
from policy_io import load_controller  # noqa: E402
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
OUT = ROOT / "outputs" / "study" / "cine"
W, H, FPS = 1280, 720, 30

# ------------------------------------------------------------------ palette (site/DESIGN.md semantics, dark theme)
C = dict(model=(74, 168, 255), learned=(167, 139, 250), hybrid=(46, 230, 196),
         warn=(255, 180, 84), bad=(255, 107, 107), good=(94, 224, 138),
         ink=(236, 241, 248), dim=(150, 163, 182), card=(10, 14, 22, 178), line=(255, 255, 255, 30))
ERR_MAX_CM = 60.0
HUD_BOXES = [(24, 22, 456, 130), (24, 446, 464, 638), (930, 22, 1256, 420), (930, 544, 1256, 638), (16, 652, 1264, 706)]
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_fc = {}


def font(sz, kind="reg"):
    from PIL import ImageFont
    key = (sz, kind)
    if key not in _fc:
        name = {"reg": "segoeui.ttf", "bold": "segoeuib.ttf", "semi": "seguisb.ttf", "mono": "consola.ttf",
                "monob": "consolab.ttf"}[kind]
        for cand in (FONT_DIR / name, FONT_DIR / "arial.ttf"):
            try:
                _fc[key] = ImageFont.truetype(str(cand), sz)
                break
            except OSError:
                continue
        else:
            _fc[key] = ImageFont.load_default()
    return _fc[key]


def family(run):
    r = run.split("@")[0]
    return "model" if r == "lqr" else "hybrid" if r.startswith("res") else "learned"


def ctrl_label(run):
    r = run.split("@")[0]
    if r == "lqr":
        return "LQR", "model-based, no learning"
    if r.startswith("res"):
        return r.rsplit("_s", 1)[0] + " (hybrid)", "LQR + learned residual (PPO)"
    sub = "PPO + outer velocity integral" if "@ki" in run else "PPO"
    return r.rsplit("_s", 1)[0], sub


def smooth(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ jobs
JOBS = {
    "tour_flat_robs_c": dict(scn="tour_flat", title="Velocity tour on a flat floor", seq="tour", run="robs_c_s0@ki=1.0", cfg="robs_c",
                             cam="chase", terrain="flat floor, no relief",
                             desc="Chase camera follows a scripted tour of velocity commands (forward, sideways, diagonal, turning) on a flat floor."),
    "push_robs_b": dict(scn="push", title="Pushes during a square loop", seq="push", run="robs_b_s0@ki=1.0", cfg="robs_b", cam="chase",
                        terrain="flat floor, no relief", pushes="3:30,7:40,11:40,15:40,19:40,23:40",
                        desc="The robot loops a 1.5 m square at 0.5 m/s while pushed in random horizontal directions (30 to 40 N, 0.05 s)."),
    "perlin8_tour_res_c": dict(scn="perlin8_tour", title="Tour on 8 cm Perlin terrain", seq="tour", run="res_c_s0", cfg="res_c", perlin=0.08,
                               cam="rake", terrain="Perlin floor, 8 cm amplitude",
                               desc="Low raking camera on a bumpy Perlin floor (8 cm amplitude, no terrain training)."),
    "perlin12_tour_res_c": dict(scn="perlin12_tour", title="Tour on 12 cm Perlin terrain", seq="tour", run="res_c_s0", cfg="res_c", perlin=0.12,
                                cam="rake", terrain="Perlin floor, 12 cm amplitude",
                                desc="The same controller on a rougher Perlin floor (12 cm amplitude)."),
    "perlin8_tour_lqr": dict(scn="perlin8_tour", title="Tour on 8 cm Perlin terrain", seq="tour", run="lqr", cfg="res_c", perlin=0.08,
                             cam="rake", terrain="Perlin floor, 8 cm amplitude",
                             desc="Plain LQR on the same 8 cm Perlin floor and command tour."),
    "perlin8_nav_robs_c": dict(scn="perlin8_nav", title="Waypoint navigation on 8 cm Perlin terrain", seq="nav", run="robs_c_s0@ki=1.0",
                               cfg="robs_c", perlin=0.08, cam="nav", terrain="Perlin floor, 8 cm amplitude", vnav=0.5,
                               desc="The robot is steered through numbered waypoints at 0.5 m/s on the 8 cm Perlin floor."),
    "climb2cm_robs_c": dict(scn="climb2cm", title="Crossing a 2 cm step", seq="climb", run="robs_c_s0@ki=1.0", cfg="robs_c", step=0.02,
                            cam="climb", slow=0.3, terrain="2 cm step at y = 1 m",
                            desc="Side view: the robot drives at 0.7 m/s onto a 2 cm raised floor (shown at 0.5x slow motion)."),
    "climb3cm_robs_c": dict(scn="climb3cm", title="Crossing a 3 cm step", seq="climb", run="robs_c_s0@ki=1.0", cfg="robs_c", step=0.03,
                            cam="climb", slow=0.3, terrain="3 cm step at y = 1 m",
                            desc="Side view: the same approach at 3 cm; the robot loses the ball at the step (shown at 0.5x slow motion)."),
    "square_lqr": dict(scn="square", title="Square course", seq="square", run="lqr", cfg="res_c", cam="chase",
                       terrain="flat floor, no relief", desc="Model-based LQR drives the square course (0.5 m/s legs, one fast leg, one turning leg)."),
    "square_robs_b": dict(scn="square", title="Square course", seq="square", run="robs_b_s0@ki=1.0", cfg="robs_b", cam="chase",
                          terrain="flat floor, no relief", desc="PPO (robs_b + integral) on the same square course."),
}


# ------------------------------------------------------------------ model xml with a big repaintable floor texture
def make_xml():
    src = ROOT / "ballbot_gym" / "models" / "ballbot.xml"
    s = src.read_text()
    mdir = (src.parent / "meshes").as_posix()
    tdir = (src.parent / "textures").as_posix()
    s = s.replace('meshdir="meshes"', f'meshdir="{mdir}"').replace('texturedir="textures"', f'texturedir="{tdir}"')
    s = s.replace('<global offwidth="640" offheight="640"/>',
                  '<global offwidth="1280" offheight="720"/><quality shadowsize="4096"/>'
                  '<headlight ambient="0.42 0.42 0.45" diffuse="0.30 0.30 0.30" specular="0 0 0"/>')
    import re
    s = re.sub(r'<texture name="floor_chk"[^>]*/>',
               '<texture name="floor_chk" type="2d" builtin="flat" rgb1=".5 .5 .5" width="2048" height="2048"/>', s)
    s = re.sub(r'<material name="floor_chk"[^>]*/>',
               '<material name="floor_chk" texture="floor_chk" texrepeat="1 1" reflectance="0" specular="0.1" shininess="0.1"/>', s)
    s = s.replace('<light name="top" pos="0 0 10.0"/>',
                  '<light name="sun" directional="true" pos="0 0 5" dir="-0.7 -0.45 -0.42" diffuse="0.9 0.86 0.8" '
                  'specular="0.1 0.1 0.1" castshadow="true"/>')
    s = s.replace('<hfield name="terrain"', '<texture name="cine_sky" type="skybox" builtin="gradient" rgb1="0.34 0.46 0.62" '
                  'rgb2="0.03 0.04 0.07" width="512" height="3072"/>\n    <hfield name="terrain"')
    s = s.replace('material="floor_chk"/>', 'material="floor_chk"/>\n    <geom name="cine_ground" type="plane" size="40 40 0.1" pos="0 0 -0.02" '
                  'contype="0" conaffinity="0" rgba="0.12 0.17 0.25 1"/>', 1)
    assert "sun" in s and 'width="2048"' in s and "cine_ground" in s and "cine_sky" in s
    p = OUT / "_tmp" / "ballbot_cine.xml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s)
    return p


# ------------------------------------------------------------------ terrain: sampling + colours
RAMP = [(0.0, (50, 76, 108)), (0.30, (62, 124, 142)), (0.60, (148, 176, 122)), (1.0, (240, 214, 150))]


def ramp(t):
    t = np.clip(t, 0, 1)
    xs = [r[0] for r in RAMP]
    return np.stack([np.interp(t, xs, [r[1][c] for r in RAMP]) for c in range(3)], -1)


class Terrain:
    def __init__(self, model):
        n = int(model.hfield_nrow[0])
        self.n, self.half = n, float(model.hfield_size[0, 0])
        self.h = model.hfield_data.reshape(n, n).astype(float) * float(model.hfield_size[0, 2])
        self.hmin, self.hmax = float(self.h.min()), float(self.h.max())
        self.rng = max(self.hmax - self.hmin, 0.02)
        self.flat = (self.hmax - self.hmin) < 2e-3

    def z(self, x, y):
        fx = (np.asarray(x, float) + self.half) / (2 * self.half) * (self.n - 1)
        fy = (np.asarray(y, float) + self.half) / (2 * self.half) * (self.n - 1)
        fx, fy = np.clip(fx, 0, self.n - 1.001), np.clip(fy, 0, self.n - 1.001)
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ax, ay = fx - x0, fy - y0
        h = self.h
        return (h[y0, x0] * (1 - ax) * (1 - ay) + h[y0, x0 + 1] * ax * (1 - ay)
                + h[y0 + 1, x0] * (1 - ax) * ay + h[y0 + 1, x0 + 1] * ax * ay)


def terrain_rgb(Hr, px_m, rng, sun=(-0.7, -0.45), lw=1.3):
    """Height tint + hill shading + 1 cm contours (5 cm emphasised) + amber cliffs. Hr: height above lowest point [m]."""
    rgb = ramp(Hr / rng)
    gy, gx = np.gradient(Hr, px_m)
    gm = np.hypot(gx, gy)
    rgb = rgb * np.clip(1 - 5.0 * (gx * sun[0] + gy * sun[1]), 0.72, 1.28)[..., None]
    for step, width, col, a in ((0.01, lw, (14, 20, 30), 0.55), (0.05, lw * 1.7, (250, 250, 255), 0.85)):
        ph = Hr / step
        pxd = np.abs(ph - np.round(ph)) * step / np.maximum(gm * px_m, 1e-9)
        al = np.clip(width - pxd, 0, 1) * (gm > 2e-3) * a
        rgb = rgb * (1 - al[..., None]) + np.array(col, float) * al[..., None]
    cl = np.clip((gm - 0.3) / 0.3, 0, 1) * 0.8
    rgb = rgb * (1 - cl[..., None]) + np.array([255, 170, 70.0]) * cl[..., None]
    return np.clip(rgb, 0, 255)


def upsample(h, T):
    n = h.shape[0]
    g = np.linspace(0, n - 1, T)
    i0 = np.clip(np.floor(g).astype(int), 0, n - 2)
    f = g - i0
    A = h[i0] * (1 - f)[:, None] + h[i0 + 1] * f[:, None]
    return A[:, i0] * (1 - f)[None] + A[:, i0 + 1] * f[None]


def paint_floor(model, terr):
    tid = model.texture("floor_chk").id
    T = int(model.tex_width[tid])
    assert int(model.tex_nchannel[tid]) == 3
    Hh = upsample(terr.h - terr.hmin, T)
    px = 2 * terr.half / T
    rgb = terrain_rgb(Hh, px, terr.rng) if not terr.flat else np.broadcast_to(ramp(np.array(0.12)), (T, T, 3)).copy()
    c = (np.arange(T) * px - terr.half)
    tile = ((np.floor(c / 0.5)[:, None] + np.floor(c / 0.5)[None]) % 2)
    rgb = rgb * (1 + 0.07 * (tile - 0.5))[..., None]
    adr = int(model.tex_adr[tid])
    model.tex_data[adr:adr + T * T * 3] = np.clip(rgb[::-1], 0, 255).astype(np.uint8).reshape(-1)   # texture row 0 = +y


# ------------------------------------------------------------------ simulation
def simulate(J, xml_path, sim_limit=None):
    import mujoco
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{J['cfg']}.yaml"))
    cfg["camera"]["enabled"] = False
    cfg["env"]["cmd_resample_steps"] = 10 ** 9
    if not J.get("dr"):
        cfg["env"].pop("dr", None)
    cfg["env"]["max_ep_steps"] = 10 ** 7
    cfg["env"]["xml_path"] = str(xml_path)
    if J.get("step", 0) > 0:
        cfg["terrain"] = {"type": "book", "config": {"step_y": 1.0, "book_len": 0.0, "rough_amplitude": 0.0,
                                                     "step_height_m": J["step"], "ramp_len": 0.02, "seed": 0}}
        cfg["env"]["init_yaw_random"] = False
    if J.get("perlin", 0) > 0:
        cfg["terrain"] = {"type": "perlin", "config": {"scale": 25.0, "octaves": 4, "persistence": 0.2, "lacunarity": 2.0,
                                                       "amplitude": J["perlin"], "seed": J.get("terrain_seed", 3)}}
        cfg["env"]["init_yaw_random"] = False
    ec = {k: cfg[k] for k in ("camera", "env", "logging")}
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config=ec, seed=0, eval_env=True)()
    model, norm = load_controller(J["run"], env, "final_model")
    base = env.unwrapped
    base.cmd_fixed = np.zeros(3)
    obs, _ = env.reset(seed=0)
    m, d = base.model, base.data
    bid = m.body("base").id
    start = d.xpos[bid][:2].copy()
    seq = RC.SEQS[J["seq"]]
    nav = J["seq"] == "nav"
    vnav = J.get("vnav", 0.5)
    plan, q = [start.copy()], start.copy()
    wp_abs = []
    if nav:
        pts = [start] + [start + np.array(w) for w in RC.WAYPOINTS]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(p1 - p0) / (vnav * 0.01)))
            plan.extend(list(p0 + (p1 - p0) * (np.arange(1, n + 1) / n)[:, None]))
        wp_abs = [start + np.array(w) for w in RC.WAYPOINTS]
    else:
        cum = 0.0
        for _, c, secs in seq:
            for _ in range(int(secs * 100)):
                q = q + np.asarray(c[:2], float) * 0.01
                plan.append(q.copy())
            cum += secs
            if (np.linalg.norm(c[:2]) > 1e-6 or (c is seq[-1][1])) and all(
                    np.linalg.norm(q - w[0]) > 0.25 for w in wp_abs + [(start, 0)]):   # one numbered marker per distinct place
                wp_abs.append((q.copy(), cum))
    plan = np.array(plan)
    push_sched = []
    for tok in [x for x in J.get("pushes", "").split(",") if x]:
        tt, ff = tok.split(":")
        push_sched.append([int(float(tt) * 100), float(ff), float(np.random.default_rng(int(float(tt) * 100)).uniform(0, 2 * np.pi))])
    rec = {k: [] for k in ("qpos", "pos", "v", "wz", "cmd", "off", "tilt", "ref", "err", "pushF", "pushA", "seg", "tgt")}
    seg_names, wp_i, wp_reach, fell, fell_step, why, t, end_after = [], 0, [], False, None, "", 0, None
    pushed_prev, info = False, {}

    def record(cmd_eff, seg, F, A, tgt):
        pos = d.xpos[bid]
        rec["qpos"].append(d.qpos.copy())
        rec["pos"].append(pos.copy())
        rec["v"].append(obs["vel"][:2].copy())
        rec["wz"].append(float(obs["angular_vel"][2]))
        rec["cmd"].append(np.asarray(cmd_eff, float).copy())
        rec["off"].append(float(info.get("off_ball_deg", 0.0)))
        rec["tilt"].append(float(np.degrees(np.arccos(np.clip(d.xmat[bid].reshape(3, 3)[2, 2], -1, 1)))))
        ref = tgt
        rec["ref"].append(ref)
        if nav:
            err = float(np.min(np.linalg.norm(plan - pos[:2], axis=1)))
        else:
            err = float(np.linalg.norm(pos[:2] - plan[min(t, len(plan) - 1)]))
        rec["err"].append(err)
        rec["pushF"].append(F)
        rec["pushA"].append(A)
        rec["seg"].append(seg)
        rec["tgt"].append(tgt)

    stop = False
    for si, (label, cmd, secs) in enumerate(seq):
        seg_names.append((label, tuple(cmd), t))
        cmd = np.asarray(cmd, float)
        for _ in range(int(secs * 100)):
            pos2 = d.xpos[bid][:2]
            cmd_eff = cmd.copy()
            if nav:
                if wp_i < len(wp_abs) and np.linalg.norm(wp_abs[wp_i] - pos2) < 0.15:
                    wp_reach.append((t, wp_i))
                    wp_i += 1
                    if wp_i == len(wp_abs):
                        end_after = t + 150
                if wp_i < len(wp_abs):
                    dv = wp_abs[wp_i] - pos2
                    cmd_eff[:2] = dv / max(np.linalg.norm(dv), 1e-6) * min(vnav, 1.5 * np.linalg.norm(dv))
                else:
                    cmd_eff[:2] = 0.0
                tgt = (wp_abs[min(wp_i, len(wp_abs) - 1)]).copy()
            else:
                tgt = plan[min(t, len(plan) - 1)].copy()
            base.cmd = cmd_eff
            obs["cmd"] = cmd_eff.astype(np.float32)
            with torch.no_grad():
                act, _ = model.predict(norm(obs), deterministic=True)
            F = A = 0.0
            pushed = False
            for ps in push_sched:
                if ps[0] <= t < ps[0] + 5:
                    d.xfrc_applied[bid, :3] = ps[1] * np.array([np.cos(ps[2]), np.sin(ps[2]), 0.0])
                    pushed = True
            if not pushed and pushed_prev:
                d.xfrc_applied[bid, :3] = 0.0
            pushed_prev = pushed
            for ps in push_sched:
                if ps[0] <= t < ps[0] + 60:
                    F, A = ps[1], ps[2]
            obs, r, term, trunc, info = env.step(act)
            record(cmd_eff, si, F, A, tgt)
            t += 1
            if term:
                fell, fell_step = True, t - 1
                why = "left the ball" if info.get("off_ball_deg", 0) > 25 else "tipped over"
                stop = True
                break
            if (end_after is not None and t >= end_after) or (sim_limit and t >= sim_limit * 100):
                stop = True
                break
        if stop:
            break
    n_main = t
    post = int(round(150 * J.get("slow", 1.0)))
    if fell:   # keep simulating with motors off so the fall is shown, not frozen
        for _ in range(post):
            try:
                obs, r, term2, trunc2, info2 = env.step(np.zeros(3))
            except Exception:
                break
            record(np.zeros(3), si, 0.0, 0.0, rec["tgt"][-1])
            t += 1
    for k in ("pos", "v", "cmd", "ref", "tgt", "qpos"):
        rec[k] = np.array(rec[k])
    for k in ("wz", "off", "tilt", "err", "pushF", "pushA", "seg"):
        rec[k] = np.array(rec[k], float)
    rec.update(plan=plan, fell=fell, fell_step=fell_step, why=why, n_main=n_main, seg_names=seg_names, nav=nav,
               wp=wp_abs, wp_reach=wp_reach, start=start,
               push_events=[(p[0], p[1]) for p in push_sched if p[0] < n_main])
    return env, m, rec


def stats_of(J, L):
    n = L["n_main"]
    pos = L["pos"][:n, :2]
    step_d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    err = L["err"][:n] * 100
    spd_err = np.linalg.norm(L["v"][:n] - L["cmd"][:n, :2], axis=1)
    end_ref = L["plan"][-1] if not L["nav"] else L["wp"][-1]
    s = dict(duration_s=round(n / 100, 2), fell=bool(L["fell"]), fall_t_s=round(L["fell_step"] / 100, 2) if L["fell"] else None,
             fall_reason=L["why"] or None, mean_err_cm=round(float(err.mean()), 1), max_err_cm=round(float(err.max()), 1),
             final_err_cm=round(float(err[-1]), 1), dist_travelled_m=round(float(step_d.sum()), 2),
             max_off_ball_deg=round(float(L["off"][:n].max()), 1), max_lean_deg=round(float(L["tilt"][:n].max()), 1),
             mean_speed_err_mps=round(float(spd_err.mean()), 3),
             end_to_plan_end_cm=round(float(np.linalg.norm(pos[-1] - end_ref) * 100), 1), pushes=len(L["push_events"]))
    if L["nav"]:
        s["waypoints_reached"], s["waypoints_total"] = len(L["wp_reach"]), len(L["wp"])
    return s


# ------------------------------------------------------------------ camera
class Cam:
    def __init__(self, look, dist, az, el, fovy=45.0):
        self.look, self.dist, self.az, self.el, self.fovy = np.asarray(look, float), dist, az, el, fovy
        a, e = np.radians(az), np.radians(el)
        self.fwd = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        self.right = np.cross(self.fwd, [0, 0, 1.0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.fwd)
        self.pos = self.look - self.dist * self.fwd

    def project(self, p, w=W, h=H):
        r = np.asarray(p, float) - self.pos
        zc = r @ self.fwd
        if zc < 0.05:
            return None
        f = (h / 2) / np.tan(np.radians(self.fovy) / 2)
        return w / 2 + f * (r @ self.right) / zc, h / 2 - f * (r @ self.up) / zc


def cam_track(J, L, terr):
    """Eased follow of the robot (per sim step) + per-scenario base angles."""
    n = len(L["pos"])
    kind = J["cam"]
    tau = {"chase": 0.45, "rake": 0.5, "nav": 0.7, "climb": 0.6}[kind]
    a = 1 - np.exp(-0.01 / tau)
    vs = np.zeros((n, 2))
    for i in range(n):   # 0.5 s moving average velocity for the lead
        vs[i] = L["v"][max(0, i - 50):i + 1].mean(0)
    look = np.zeros((n, 3))
    cur = np.array([*L["pos"][0, :2], 0.15])
    for i in range(n):
        lead = {"chase": 0.35, "rake": 0.45, "nav": 0.0, "climb": 0.9}[kind] * vs[i]
        tgt = np.array([L["pos"][i, 0] + lead[0], L["pos"][i, 1] + lead[1], terr.z(L["pos"][i, 0], L["pos"][i, 1]) + 0.16])
        if kind == "nav":
            tgt[:2] = 0.5 * tgt[:2] + 0.5 * np.array([0.8, 0.5])
        cur = cur + (tgt - cur) * a
        look[i] = cur
    return look


def cam_at(J, look_i, t, t_after):
    """t: video time [s]; t_after: seconds since the sim ended (result card), 0 before."""
    kind = J["cam"]
    base = {"chase": (140.0, -30.0, 2.7), "rake": (125.0, -15.0, 2.4), "nav": (130.0, -45.0, 4.6), "climb": (180.0, -10.0, 1.9)}[kind]
    az, el, dist = base
    e = smooth(t / 4.0)
    az += (1 - e) * (55 if kind != "climb" else 38)
    dist *= 1 + (1 - e) * 0.45
    el += (1 - e) * (-8)
    az += 7.0 * t_after + (0.0 if kind != "chase" else 2.0 * np.sin(t / 5.0))
    dist *= 1 - 0.05 * smooth(t_after / 3)
    return Cam(look_i, dist, az, el)


# ------------------------------------------------------------------ 3D scene decorations
def _geom(scene, mj, typ, size, pos, mat, rgba, emis=0.55):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, typ, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    g.emission, g.specular, g.shininess, g.reflectance = emis, 0.0, 0.0, 0.0
    scene.ngeom += 1
    return g


def seg3(scene, mj, p0, p1, rgba, r, typ=None, emis=0.55):
    g = _geom(scene, mj, typ or mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba, emis)
    if g is not None:
        mj.mjv_connector(g, typ or mj.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(p0, float), np.asarray(p1, float))


def err_rgba(e_m, a=1.0):
    import matplotlib
    c = matplotlib.colormaps["inferno"](0.3 + 0.68 * np.clip(e_m * 100 / ERR_MAX_CM, 0, 1))
    return [c[0], c[1], c[2], a]


class Stage:
    """Holds a model + replay data + renderer + the static scene decorations."""

    def __init__(self, m, terr, J, L, w=W, h=H):
        import mujoco
        self.mj, self.m, self.terr, self.J, self.L = mujoco, m, terr, J, L
        self.d = mujoco.MjData(m)
        self.r = mujoco.Renderer(m, height=h, width=w, max_geom=12000)
        self.w, self.h = w, h
        plan = L["plan"]
        keep, last = [0], plan[0]
        for i in range(1, len(plan)):
            if np.linalg.norm(plan[i] - last) >= 0.04:
                keep.append(i)
                last = plan[i]
        self.pl = plan[keep]
        self.plz = terr.z(self.pl[:, 0], self.pl[:, 1]) + 0.009

    def render(self, i, cam, hud=True):
        mj, d, L, J, terr = self.mj, self.d, self.L, self.J, self.terr
        d.qpos[:] = L["qpos"][i]
        mj.mj_forward(self.m, d)
        mc = mj.MjvCamera()
        mc.type = mj.mjtCamera.mjCAMERA_FREE
        mc.lookat[:], mc.distance, mc.azimuth, mc.elevation = cam.look, cam.dist, cam.az, cam.el
        self.r.update_scene(d, camera=mc)
        sc = self.r.scene
        P3 = lambda xy, dz=0.009: np.array([xy[0], xy[1], float(terr.z(xy[0], xy[1])) + dz])
        for k in range(len(self.pl) - 1):   # planned path ribbon
            seg3(sc, mj, [*self.pl[k], self.plz[k]], [*self.pl[k + 1], self.plz[k + 1]], [0.92, 0.97, 1.0, 0.95], 0.006)
        wp_list = L["wp"] if L["nav"] else [w[0] for w in L["wp"]]
        reached = len(L["wp_reach"]) if False else None
        for k, w in enumerate(wp_list):
            if L["nav"]:
                done = any(r[0] <= i and r[1] == k for r in L["wp_reach"])
            else:
                done = i >= int(L["wp"][k][1] * 100)
            col = [*(np.array(C["good"]) / 255), 1.0] if done else [1, 1, 1, 1.0]
            p = P3(w, 0.004)
            _geom(sc, mj, mj.mjtGeom.mjGEOM_CYLINDER, [0.07, 0.004, 0], p, np.eye(3), col, 0.7)
        n_trail = min(i, L["n_main"] - 1)
        idx = np.arange(0, n_trail + 1, 5)
        if len(idx) and idx[-1] != n_trail:
            idx = np.append(idx, n_trail)
        for a_, b_ in zip(idx[:-1], idx[1:]):
            pa, pb = L["pos"][a_, :2], L["pos"][b_, :2]
            seg3(sc, mj, P3(pa, 0.016), P3(pb, 0.016), err_rgba(L["err"][b_]), 0.010, emis=0.8)
        alive = not (L["fell"] and i > L["fell_step"])
        if alive:
            pos, tg = L["pos"][i], L["tgt"][i]
            _geom(sc, mj, mj.mjtGeom.mjGEOM_SPHERE, [0.032, 0, 0], P3(tg, 0.04), np.eye(3), [1, 1, 1, 1], 0.9)
            seg3(sc, mj, P3(pos[:2], 0.03), P3(tg, 0.04), [*err_rgba(L["err"][i])[:3], 0.45], 0.007)
        if L["pushF"][i] > 0:
            ang, F = L["pushA"][i], L["pushF"][i]
            tip = L["pos"][i] + np.array([0, 0, 0.18])
            from render_control import arrow
            arrow(sc, mj, tip, np.array([np.cos(ang), np.sin(ang)]) * F / 60.0, [1.0, 0.7, 0.2, 1.0], 1.0)
        return self.r.render()

    def close(self):
        self.r.close()


# ------------------------------------------------------------------ minimap
class MiniMap:
    def __init__(self, terr, L, size=300, ss=2, extra=None):
        from PIL import Image
        self.L, self.size, self.ss, self.S = L, size, ss, size * ss
        pts = [L["plan"], L["pos"][:, :2]] + (extra or [])
        P = np.vstack(pts)
        lo, hi = P.min(0), P.max(0)
        self.c = (lo + hi) / 2
        self.span = max(float((hi - lo).max()), 1.6) * 1.12 + 0.2
        S = self.S
        g = (np.arange(S) + 0.5) / S - 0.5
        xs = self.c[0] + g * self.span
        ys = self.c[1] - g * self.span          # image row 0 = top = +y
        X, Y = np.meshgrid(xs, ys)
        Z = terr.z(X, Y) - terr.hmin
        if terr.flat:
            rgb = np.broadcast_to(ramp(np.array(0.12)), (S, S, 3)).copy()
        else:
            rgb = terrain_rgb(Z[::-1], self.span / S, terr.rng, sun=(-0.7, 0.45), lw=1.1)[::-1]
        rgb = rgb * 0.82
        for gx in np.arange(np.floor(xs.min()), xs.max() + 0.5, 0.5):       # 0.5 m grid
            px = int((gx - self.c[0]) / self.span * S + S / 2)
            if 0 <= px < S:
                rgb[:, px] = rgb[:, px] * 0.8 + 255 * 0.2 * (1.0 if abs(gx - round(gx)) < 1e-6 else 0.5)
        for gy in np.arange(np.floor(ys.min()), ys.max() + 0.5, 0.5):
            py = int((self.c[1] - gy) / self.span * S + S / 2)
            if 0 <= py < S:
                rgb[py, :] = rgb[py, :] * 0.8 + 255 * 0.2 * (1.0 if abs(gy - round(gy)) < 1e-6 else 0.5)
        self.base = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGBA")
        from PIL import ImageDraw
        dr = ImageDraw.Draw(self.base, "RGBA")
        pl = [self.px(p) for p in L["plan"][::5]]
        dr.line(pl, fill=(240, 247, 255, 235), width=3, joint="curve")
        f = font(13 * ss, "bold")
        wps = L["wp"] if L["nav"] else [w[0] for w in L["wp"]]
        for k, w in enumerate(wps):
            x, y = self.px(w)
            dr.ellipse([x - 9 * ss, y - 9 * ss, x + 9 * ss, y + 9 * ss], fill=(12, 18, 28, 235), outline=(240, 247, 255, 255), width=2)
            dr.text((x, y), str(k + 1), font=f, fill=(255, 255, 255, 255), anchor="mm")

    def px(self, p):
        return ((p[0] - self.c[0]) / self.span + 0.5) * self.S, (0.5 - (p[1] - self.c[1]) / self.span) * self.S

    def frame(self, i, fam, extra_trails=None):
        from PIL import Image, ImageDraw
        L, ss = self.L, self.ss
        im = self.base.copy()
        dr = ImageDraw.Draw(im, "RGBA")
        n = min(i, L["n_main"] - 1)
        idx = list(range(0, n + 1, 5))
        if idx and idx[-1] != n:
            idx.append(n)
        for a_, b_ in zip(idx[:-1], idx[1:]):
            col = tuple(int(255 * c) for c in err_rgba(L["err"][b_])[:3]) + (255,)
            dr.line([self.px(L["pos"][a_, :2]), self.px(L["pos"][b_, :2])], fill=col, width=5 * ss // 2 + 1)
        alive = not (L["fell"] and i > L["fell_step"])
        if alive:
            tx, ty = self.px(L["tgt"][i])
            dr.ellipse([tx - 6 * ss, ty - 6 * ss, tx + 6 * ss, ty + 6 * ss], outline=(255, 255, 255, 255), width=2 * ss // 2 + 1)
            x, y = self.px(L["pos"][i, :2])
            dr.line([(x, y), (tx, ty)], fill=(255, 255, 255, 120), width=2)
        x, y = self.px(L["pos"][min(i, L["fell_step"] if L["fell"] else i), :2])
        r = 6 * ss
        if L["fell"] and i >= L["fell_step"]:
            dr.line([x - r, y - r, x + r, y + r], fill=C["bad"] + (255,), width=4)
            dr.line([x - r, y + r, x + r, y - r], fill=C["bad"] + (255,), width=4)
        else:
            dr.ellipse([x - r, y - r, x + r, y + r], fill=C[fam] + (255,), outline=(255, 255, 255, 255), width=2)
        return im.resize((self.size, self.size), Image.LANCZOS)


# ------------------------------------------------------------------ HUD
def card(dr, box, r=16, fill=None, outline=None):
    dr.rounded_rectangle(box, radius=r, fill=fill or C["card"], outline=outline or C["line"], width=1)


def label(dr, xy, s, f, fill=(255, 255, 255, 255), anchor="la", stroke=0):
    dr.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=(0, 0, 0, 255) if stroke else None)


def pill(dr, x, y, s, f, col, tcol=(8, 12, 18, 255), padx=12, h=26):
    w = int(dr.textlength(s, font=f)) + 2 * padx
    dr.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=col + (255,))
    dr.text((x + w / 2, y + h / 2 + 0.5), s, font=f, fill=tcol, anchor="mm")
    return w


def arrow2d(dr, c, v, col, scale, width=4):
    x, y = c
    tx, ty = x + v[0] * scale, y - v[1] * scale
    L = np.hypot(tx - x, ty - y)
    if L < 3:
        return
    dr.line([(x, y), (tx, ty)], fill=col, width=width)
    u = np.array([tx - x, ty - y]) / L
    n = np.array([-u[1], u[0]])
    hl = min(11, L)
    dr.polygon([(tx, ty), tuple(np.array([tx, ty]) - u * hl + n * 5), tuple(np.array([tx, ty]) - u * hl - n * 5)], fill=col)


def terrain_legend(dr, terr, J, x, y, w=326, hgt=92):
    card(dr, [x, y, x + w, y + hgt])
    label(dr, (x + 16, y + 10), "TERRAIN", font(11, "bold"), C["dim"] + (255,))
    if terr.flat:
        label(dr, (x + 16, y + 28), "Flat floor", font(17, "semi"), C["ink"] + (255,))
        label(dr, (x + 16, y + 52), "no relief, 0.5 m checker tiles", font(13), C["dim"] + (255,))
    else:
        lab = f"Height above lowest point: 0 to {terr.rng * 100:.1f} cm"
        label(dr, (x + 16, y + 28), lab, font(14, "semi"), C["ink"] + (255,))
        bar = ramp(np.linspace(0, 1, 200))
        for k in range(200):
            dr.line([(x + 16 + k * (w - 32) / 200, y + 52), (x + 16 + (k + 1.2) * (w - 32) / 200, y + 52)],
                    fill=tuple(int(c) for c in bar[k]) + (255,), width=10)
        label(dr, (x + 16, y + 62), "0", font(11, "mono"), C["dim"] + (255,))
        label(dr, (x + w - 16, y + 62), f"{terr.rng * 100:.0f} cm", font(11, "mono"), C["dim"] + (255,), "ra")
        label(dr, (x + 16, y + 74), "contour lines every 1 cm (white: 5 cm), amber: steep edge", font(11), C["dim"] + (255,))
    label(dr, (x + w - 12, y + 10), "SIMULATION ONLY", font(10, "bold"), C["dim"] + (200,), "ra")


# ------------------------------------------------------------------ main compose for a single scenario
class Timeline:
    def __init__(self, J, L, slow):
        self.J, self.L, self.slow = J, L, slow
        n = len(L["pos"])
        self.n_sim_frames = int(np.ceil((n / 100.0) / slow * FPS))
        self.n_result = int(3.6 * FPS)
        self.total = self.n_sim_frames + self.n_result

    def idx(self, k):
        if k >= self.n_sim_frames:
            return len(self.L["pos"]) - 1
        return int(min(round(k / FPS * self.slow * 100), len(self.L["pos"]) - 1))

    def t_after(self, k):
        return max(0.0, (k - self.n_sim_frames) / FPS)


def hud_frame(img, k, tl, J, L, terr, mm, cam, fam, st, tot_title, vsm):
    from PIL import Image, ImageDraw
    i = tl.idx(k)
    t_vid = k / FPS
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov, "RGBA")
    fcol = C[fam]
    fell = L["fell"] and i >= L["fell_step"]
    sim_t = min(i, L["n_main"] - 1) / 100.0
    # ---- title card (fades in after the intro banner)
    ta = smooth((t_vid - 2.9) / 0.8)
    tlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(tlay, "RGBA")
    card(td, [24, 22, 456, 130])
    td.rounded_rectangle([24, 22, 30, 130], radius=3, fill=fcol + (255,))
    label(td, (46, 30), J["title"], font(21, "bold"), C["ink"] + (255,))
    name, sub = ctrl_label(J["run"])
    w = pill(td, 46, 64, name, font(14, "bold"), fcol)
    label(td, (46 + w + 10, 67), sub, font(13), C["dim"] + (255,))
    label(td, (46, 100), f"{J['terrain']}   |   DR: {'yes' if J.get('dr') else 'no'}   |   sim only", font(13), C["dim"] + (255,))
    tlay.putalpha(tlay.getchannel("A").point(lambda v: int(v * ta)))
    ov.alpha_composite(tlay)
    dr = ImageDraw.Draw(ov, "RGBA")
    # ---- status chip
    pushed_now = (not fell) and L["pushF"][i] > 0 and (i - 0) >= 0
    if fell:
        txt, col = f"FELL  t = {L['fell_step'] / 100:.1f} s", C["bad"]
    elif pushed_now:
        txt, col = f"PUSHED {L['pushF'][i]:.0f} N", C["warn"]
    else:
        txt, col = "BALANCING", C["good"]
    f = font(17, "bold")
    tw = dr.textlength(txt, font=f)
    cx0 = 640 - (tw + 54) / 2 - 20
    dr.rounded_rectangle([cx0, 22, cx0 + tw + 54, 54], radius=16, fill=(10, 14, 22, 200), outline=col + (255,), width=2)
    dr.ellipse([cx0 + 14, 33, cx0 + 26, 45], fill=col + (255,))
    label(dr, (cx0 + 36, 38), txt, f, col + (255,), "lm")
    # ---- stats card
    sx, sy, sw, sh = 24, 446, 440, 192
    card(dr, [sx, sy, sx + sw, sy + sh])
    cmd, v = L["cmd"][i], vsm[i]
    cc = (sx + 82, sy + 92)
    R = 58
    dr.ellipse([cc[0] - R, cc[1] - R, cc[0] + R, cc[1] + R], outline=(255, 255, 255, 70), width=2)
    dr.ellipse([cc[0] - R // 2, cc[1] - R // 2, cc[0] + R // 2, cc[1] + R // 2], outline=(255, 255, 255, 35), width=1)
    dr.line([(cc[0] - R, cc[1]), (cc[0] + R, cc[1])], fill=(255, 255, 255, 35))
    dr.line([(cc[0], cc[1] - R), (cc[0], cc[1] + R)], fill=(255, 255, 255, 35))
    label(dr, (cc[0], cc[1] - R - 3), "+y", font(11, "bold"), C["dim"] + (255,), "ms")
    label(dr, (cc[0] + R + 3, cc[1]), "+x", font(11, "bold"), C["dim"] + (255,), "lm")
    arrow2d(dr, cc, cmd[:2], (240, 244, 250, 255), R * 0.95, 4)
    arrow2d(dr, cc, v, fcol + (255,), R * 0.95, 4)
    dr.ellipse([cc[0] - 4, cc[1] - 4, cc[0] + 4, cc[1] + 4], fill=(255, 255, 255, 255))
    label(dr, (sx + 24, sy + 14), f"WORLD VELOCITY (top view)", font(10, "bold"), C["dim"] + (255,))
    dr.ellipse([sx + 22, sy + 169, sx + 32, sy + 179], fill=(240, 244, 250, 255))
    label(dr, (sx + 37, sy + 175), "command", font(12), C["dim"] + (255,), "lm")
    dr.ellipse([sx + 104, sy + 169, sx + 114, sy + 179], fill=fcol + (255,))
    label(dr, (sx + 119, sy + 175), "actual", font(12), C["dim"] + (255,), "lm")
    mono, monob, lab = font(15, "mono"), font(15, "monob"), font(12)
    x0 = sx + 176
    rows = [
        ("time", f"{sim_t:6.2f} s", None),
        ("command", f"{cmd[0]:+5.2f} {cmd[1]:+5.2f} m/s", None),
        ("actual", f"{v[0]:+5.2f} {v[1]:+5.2f} m/s", None),
        ("speed err", f"{np.linalg.norm(v - cmd[:2]):5.2f} m/s", None),
        ("yaw rate", f"{cmd[2]:+5.2f} /{vsm_w(L, i):+5.2f}", None),
        ("lean", f"{L['tilt'][i]:5.1f} deg", None),
        ("off-ball", f"{L['off'][i]:5.1f} deg", L["off"][i] > 20),
        ("travelled", f"{st['dist'][i]:6.2f} m", None),
    ]
    for r_, (a_, b_, warn) in enumerate(rows):
        y = sy + 14 + r_ * 20
        label(dr, (x0, y + 8), a_, lab, C["dim"] + (255,), "lm")
        label(dr, (sx + sw - 18, y + 8), b_, mono if not warn else monob, (C["warn"] if warn else C["ink"]) + (255,), "rm")
    # current command label
    segname = L["seg_names"][int(L["seg"][i])][0]
    label(dr, (sx + sw - 18, sy + 175), segname, font(12, "semi"), C["dim"] + (255,), "rm")
    # ---- minimap card
    mx, my = 930, 22
    card(dr, [mx, my, mx + 326, my + 398])
    label(dr, (mx + 16, my + 10), "TOP VIEW   plan, waypoints, trail", font(11, "bold"), C["dim"] + (255,))
    ov.alpha_composite(mm.frame(i, fam), (mx + 13, my + 30))
    dr = ImageDraw.Draw(ov, "RGBA")
    gy = my + 340
    for kk in range(240):
        c = err_rgba(kk / 239 * ERR_MAX_CM / 100)
        dr.line([(mx + 16 + kk, gy), (mx + 16 + kk, gy + 8)], fill=tuple(int(255 * q) for q in c[:3]) + (255,))
    label(dr, (mx + 16, gy + 12), "0", font(11, "mono"), C["dim"] + (255,))
    label(dr, (mx + 16 + 240, gy + 12), f"{ERR_MAX_CM:.0f}+ cm", font(11, "mono"), C["dim"] + (255,), "ra")
    label(dr, (mx + 270, gy - 2), "position", font(11), C["dim"] + (255,))
    label(dr, (mx + 270, gy + 11), "error", font(11), C["dim"] + (255,))
    label(dr, (mx + 16, gy + 28), f"now {L['err'][i] * 100:4.0f} cm   mean {st['mean_err'][i] * 100:3.0f} cm   o = target", font(12, "mono"), C["ink"] + (255,))
    # ---- terrain legend
    terrain_legend(dr, terr, J, 930, 544, 326, 94)
    # ---- time bar
    card(dr, [16, 652, 1264, 706], r=14)
    T_end = L["n_main"] / 100.0
    T_tot = len(L["pos"]) / 100.0
    t_show = sim_t
    label(dr, (34, 679), f"{t_show:5.1f} / {T_end:4.1f} s", font(16, "monob"), C["ink"] + (255,), "lm")
    x0b, x1b, yb = 196, 1244, 688
    dr.rounded_rectangle([x0b, yb - 3, x1b, yb + 3], radius=3, fill=(255, 255, 255, 45))
    fx = x0b + (x1b - x0b) * min(1.0, (min(i, L["n_main"]) / 100.0) / T_end)
    dr.rounded_rectangle([x0b, yb - 3, fx, yb + 3], radius=3, fill=fcol + (255,))
    X = lambda ts: x0b + (x1b - x0b) * ts / T_end
    for (ts, F) in L["push_events"]:
        if ts > i:      # no spoilers: markers appear when the event happens
            continue
        x = X(ts / 100.0)
        dr.polygon([(x - 6, yb - 20), (x + 6, yb - 20), (x, yb - 10)], fill=C["warn"] + (255,))
        label(dr, (x, yb - 31), f"{F:.0f} N", font(11, "bold"), C["warn"] + (255,), "ms")
    if L["nav"]:
        for ts, k_ in L["wp_reach"]:
            if ts > i:
                continue
            x = X(ts / 100.0)
            dr.ellipse([x - 5, yb - 5, x + 5, yb + 5], fill=(255, 255, 255, 255))
            label(dr, (x, yb - 14), str(k_ + 1), font(11, "bold"), (255, 255, 255, 255), "ms")
    else:
        for k_, (_, ts) in enumerate(L["wp"]):
            if ts <= T_end + 1e-6:
                x = X(ts)
                dr.ellipse([x - 4, yb - 4, x + 4, yb + 4], fill=(255, 255, 255, 255))
                label(dr, (x, yb - 14), str(k_ + 1), font(11, "bold"), (255, 255, 255, 220), "ms")
    if J.get("step"):
        ie = np.where(L["pos"][:L["n_main"], 1] >= 1.0)[0]
        if len(ie) and ie[0] <= i:
            x = X(ie[0] / 100.0)
            dr.polygon([(x, yb - 12), (x + 6, yb - 6), (x, yb), (x - 6, yb - 6)], fill=C["hybrid"] + (255,))
            label(dr, (x, yb - 22), f"{J['step'] * 100:.0f} cm step", font(11, "bold"), C["hybrid"] + (255,), "ms")
    if L["fell"] and L["fell_step"] <= i:
        x = X(L["fell_step"] / 100.0)
        dr.line([(x - 6, yb - 8), (x + 6, yb + 8)], fill=C["bad"] + (255,), width=3)
        dr.line([(x - 6, yb + 8), (x + 6, yb - 8)], fill=C["bad"] + (255,), width=3)
        label(dr, (x, yb - 22), "FELL", font(11, "bold"), C["bad"] + (255,), "ms")
    if J.get("slow", 1.0) != 1.0:
        label(dr, (1244, 664), f"{J['slow']:g}x slow motion", font(12, "bold"), C["warn"] + (255,), "ra")
    # ---- 3D labels (waypoint numbers do not draw over the robot; step label)
    if J.get("step"):
        pr = cam.project([0.0, 1.0, J["step"] + 0.025])
        if pr and 20 < pr[0] < W - 20 and 60 < pr[1] < H - 90 and not any(b[0] - 60 < pr[0] < b[2] + 60 and b[1] - 20 < pr[1] < b[3] + 10 for b in HUD_BOXES):
            s = f"{J['step'] * 100:.0f} cm step"
            label(dr, (pr[0], pr[1] - 24), s, font(20, "bold"), C["warn"] + (255,), "ms", stroke=3)
            dr.line([(pr[0], pr[1] - 16), (pr[0], pr[1] - 2)], fill=C["warn"] + (255,), width=2)
    if L["pushF"][i] > 0 and not fell:
        tip = L["pos"][i] + np.array([np.cos(L["pushA"][i]), np.sin(L["pushA"][i]), 0]) * L["pushF"][i] / 60.0 + np.array([0, 0, 0.18])
        pr = cam.project(tip)
        if pr:
            label(dr, (pr[0], pr[1] - 14), f"{L['pushF'][i]:.0f} N push", font(20, "bold"), C["warn"] + (255,), "ms", stroke=3)
    wl = L["wp"] if L["nav"] else [w[0] for w in L["wp"]]
    for k_, w in enumerate(wl):
        pr = cam.project([w[0], w[1], terr.z(w[0], w[1]) + 0.12])
        if pr and 10 < pr[0] < W - 10 and 70 < pr[1] < H - 70 and not any(b[0] - 10 < pr[0] < b[2] + 10 and b[1] - 10 < pr[1] < b[3] + 10 for b in HUD_BOXES):
            if np.hypot(*(np.array(pr) - np.array(cam.project(L["pos"][i] + [0, 0, 0.25]) or (-999, -999)))) > 130:
                label(dr, pr, str(k_ + 1), font(17, "bold"), (255, 255, 255, 255), "mm", stroke=3)
    # ---- fall banner (1.5 s hold) / intro / result
    if L["fell"] and i >= L["fell_step"] and k < tl.n_sim_frames:
        since = (k / FPS) - (L["fell_step"] / 100.0 / tl.slow)
        if since <= 1.5 + 1e-6:
            dr.rounded_rectangle([340, 92, 940, 156], radius=18, fill=(150, 20, 28, 235), outline=(255, 160, 160, 255), width=2)
            label(dr, (640, 112), f"FELL at t = {L['fell_step'] / 100:.1f} s", font(26, "bold"), (255, 255, 255, 255), "mm")
            label(dr, (640, 141), f"{L['why']}; motors off, run ends here", font(15), (255, 225, 225, 255), "mm")
    if t_vid < 3.6:
        a = smooth(t_vid / 0.5) * (1 - smooth((t_vid - 2.7) / 0.9))
        if a > 0:
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ld = ImageDraw.Draw(lay, "RGBA")
            ld.rectangle([0, 66, W, 192], fill=(6, 9, 14, 165))
            ld.rectangle([0, 66, W, 70], fill=fcol + (255,))
            label(ld, (640, 104), J["title"], font(44, "bold"), (255, 255, 255, 255), "mm")
            label(ld, (640, 144), f"{name}   |   {sub}", font(21), fcol + (255,), "mm")
            label(ld, (640, 174), f"{J['terrain']}   |   MuJoCo simulation, no hardware", font(16), (200, 210, 224, 255), "mm")
            lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a)))
            ov.alpha_composite(lay)
            dr = ImageDraw.Draw(ov, "RGBA")
    if k >= tl.n_sim_frames:
        a = smooth(tl.t_after(k) / 0.5)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay, "RGBA")
        ld.rectangle([0, 0, W, H], fill=(5, 8, 12, 175))
        bx0, by0, bx1, by1 = 330, 150, 950, 580
        ld.rounded_rectangle([bx0, by0, bx1, by1], radius=22, fill=(10, 14, 22, 252), outline=(255, 255, 255, 40), width=1)
        okc = C["bad"] if L["fell"] else C["good"]
        ld.rounded_rectangle([bx0, by0, bx0 + 8, by1], radius=4, fill=fcol + (255,))
        label(ld, (bx0 + 36, by0 + 26), "RESULT", font(13, "bold"), C["dim"] + (255,))
        label(ld, (bx0 + 36, by0 + 48), J["title"], font(26, "bold"), C["ink"] + (255,))
        label(ld, (bx0 + 36, by0 + 88), f"{name}   {sub}", font(15), fcol + (255,))
        if L["fell"]:
            out = f"FELL at t = {st['s']['fall_t_s']:.1f} s ({L['why']})"
        elif L["nav"] and st["s"].get("waypoints_reached", 0) < st["s"].get("waypoints_total", 0):
            out = f"No fall; {st['s']['waypoints_reached']} of {st['s']['waypoints_total']} waypoints reached"
        else:
            out = "No fall"
        pill(ld, bx0 + 36, by0 + 120, out, font(18, "bold"), okc, h=34, padx=16)
        s = st["s"]
        rows = [("Falls", "1" if L["fell"] else "0"),
                ("Mean position error", f"{s['mean_err_cm']:.0f} cm"),
                ("Final position error", f"{s['final_err_cm']:.0f} cm   (max {s['max_err_cm']:.0f} cm)"),
                ("Max off-ball angle", f"{s['max_off_ball_deg']:.1f} deg"),
                ("Mean speed error", f"{s['mean_speed_err_mps']:.2f} m/s"),
                ("Distance travelled", f"{s['dist_travelled_m']:.1f} m in {s['duration_s']:.1f} s")]
        if L["nav"]:
            rows.append(("Waypoints reached", f"{s['waypoints_reached']} of {s['waypoints_total']}"))
        for r_, (a_, b_) in enumerate(rows):
            y = by0 + 182 + r_ * 36
            label(ld, (bx0 + 36, y), a_, font(17), C["dim"] + (255,), "lm")
            label(ld, (bx1 - 32, y), b_, font(18, "monob"), C["ink"] + (255,), "rm")
            ld.line([(bx0 + 36, y + 18), (bx1 - 32, y + 18)], fill=(255, 255, 255, 25))
        label(ld, (bx0 + 36, by1 - 22), "Simulation only. Nominal physics unless stated. Position error = distance to the planned point.",
              font(12), C["dim"] + (255,), "lm")
        lay.putalpha(lay.getchannel("A").point(lambda v: int(v * a)))
        ov.alpha_composite(lay)
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    return out


def vsm_w(L, i):
    return float(L["wz"][max(0, i - 20):i + 1].mean())


def precompute_stats(L):
    n = len(L["pos"])
    vsm = np.zeros((n, 2))
    for i in range(n):
        vsm[i] = L["v"][max(0, i - 20):i + 1].mean(0)
    step = np.r_[0, np.linalg.norm(np.diff(L["pos"][:, :2], axis=0), axis=1)]
    step[min(L["n_main"], n):] = 0
    dist = np.cumsum(step)
    ec = np.cumsum(L["err"][:L["n_main"]])
    mean_err = np.r_[ec / np.arange(1, len(ec) + 1), np.full(n - len(ec), ec[-1] / len(ec))] if n > len(ec) else ec / np.arange(1, n + 1)
    return vsm, dict(dist=dist, mean_err=mean_err)


# ------------------------------------------------------------------ driver
def open_ffmpeg(path):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_job(name, args):
    J = dict(JOBS[name])
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"cine_{J['scn']}_{J['run'].split('@')[0].rsplit('_s', 1)[0]}"
    t0 = _time.time()
    xml = make_xml()
    env, m, L = simulate(J, xml, args.sim_limit)
    print(f"[{name}] simulated {len(L['pos']) / 100:.1f} s, fell={L['fell']} ({_time.time() - t0:.0f} s)", flush=True)
    terr = Terrain(m)
    paint_floor(m, terr)
    S = stats_of(J, L)
    fam = family(J["run"])
    vsm, st = precompute_stats(L)
    st["s"] = S
    slow = J.get("slow", 1.0)
    tl = Timeline(J, L, slow)
    look = cam_track(J, L, terr)
    stage = Stage(m, terr, J, L)
    mm = MiniMap(terr, L)
    dump = {int(float(x) * FPS) for x in args.dump.split(",") if x}
    poster_k = int((L["fell_step"] / 100.0 / slow + 0.5) * FPS) if L["fell"] else int(0.6 * tl.n_sim_frames)
    ff = None if args.no_video else open_ffmpeg(OUT / f"{stem}.mp4")
    (OUT / "_dbg").mkdir(exist_ok=True)
    for k in range(tl.total):
        i = tl.idx(k)
        cam = cam_at(J, look[i], k / FPS, tl.t_after(k))
        raw = stage.render(i, cam)
        from PIL import Image
        img = hud_frame(Image.fromarray(raw), k, tl, J, L, terr, mm, cam, fam, st, J["title"], vsm)
        if k in dump:
            img.save(OUT / "_dbg" / f"{stem}_{k / FPS:05.1f}s.png")
        if k == poster_k:
            img.save(OUT / f"{stem}.jpg", quality=90)
        if ff is not None:
            ff.stdin.write(np.asarray(img).tobytes())
        if k % 150 == 0:
            print(f"  frame {k}/{tl.total} ({_time.time() - t0:.0f} s)", flush=True)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    stage.close()
    outcome = (f"Fell at t = {S['fall_t_s']} s ({L['why']})." if L["fell"] else "No fall.") + \
              f" Mean position error {S['mean_err_cm']} cm, max {S['max_err_cm']} cm, final {S['final_err_cm']} cm; max off-ball angle {S['max_off_ball_deg']} deg."
    side = dict(id=stem, title=J["title"], controller=J["run"], controller_family=fam, scenario=J["scn"], terrain=J["terrain"],
                description=J["desc"], outcome=outcome, stats=S, duration_s=round(tl.total / FPS, 1), sim_only=True,
                domain_randomisation=bool(J.get("dr")), slow_motion=slow, pushes=J.get("pushes", ""),
                config=f"configs/env/{J['cfg']}.yaml", sequence=J["seq"],
                command=f"python scripts/study/render_cinematic.py --job {name}")
    (OUT / f"{stem}.json").write_text(json.dumps(side, indent=1))
    np.savez(OUT / f"{stem}_traj.npz", pos=L["pos"], err=L["err"], plan=L["plan"], fell=L["fell"])
    print(f"[{name}] done {stem} {tl.total / FPS:.1f} s video in {_time.time() - t0:.0f} s | {outcome}", flush=True)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=list(JOBS))
    ap.add_argument("--compare", nargs=2, metavar=("JOB_A", "JOB_B"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sim-limit", type=float, default=None, help="truncate the simulation [s] (test clips)")
    ap.add_argument("--dump", default="", help="comma list of video times [s] to save as PNG under outputs/study/cine/_dbg")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if args.list:
        for k, v in JOBS.items():
            print(k, "|", v["run"], v["seq"])
        return
    if args.compare:
        import render_compare
        render_compare.run(args.compare, args, sys.modules[__name__])
    elif args.job:
        render_job(args.job, args)


if __name__ == "__main__":
    main()
