"""Render recorded trajectories (record.py npz) in the MuJoCo renderer, side by side: left = one engine, right = another.

Both panels use the same renderer and the same drawing code: poses come from each engine's own state log, the wheel-ball
contact points and normal/friction force vectors come from each engine's own contact data (MuJoCo mj_contactForce, PhysX
contact reporting).  So the picture shows the engines' numbers, not a re-simulation.  Simulation only.

  python render.py --left mj_nom_track04 --right px_nom_track04 --title "LQR 0.4 scaled speed" --out track04_nom
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import ROOT, DesignCfg

OUT = ROOT / "outputs/study/isaac_port"
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
C = dict(model=(74, 168, 255), px=(118, 185, 0), bad=(255, 107, 107), good=(94, 224, 138), ink=(236, 241, 248),
         dim=(150, 163, 182), warn=(255, 180, 84))
PW, PH = 640, 720
_fc = {}


def font(sz, kind="reg"):
    k = (sz, kind)
    if k not in _fc:
        name = {"reg": "segoeui.ttf", "bold": "segoeuib.ttf", "mono": "consola.ttf"}[kind]
        try:
            _fc[k] = ImageFont.truetype(str(FONT_DIR / name), sz)
        except OSError:
            _fc[k] = ImageFont.load_default()
    return _fc[k]


class Panel:
    def __init__(self, design):
        cfg = DesignCfg(design)
        spec = mujoco.MjSpec.from_file(str(cfg.xml))
        spec.visual.global_.offwidth, spec.visual.global_.offheight = PW, PH
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, PH, PW)
        m = self.m
        self.bq = m.joint("base_free_joint").qposadr[0]
        self.kq = m.joint("ball_free_joint").qposadr[0]
        self.wq = [m.joint(f"wheel_joint_{k}").qposadr[0] for k in range(3)]
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.62, 140.0, -28.0
        m.geom_rgba[m.geom('the_ball').id] = [1, 1, 1, 0.4]
        for k in range(3):
            m.geom_rgba[m.geom(f'wheel_mesh_{k}').id, 3] = 0.3
        m.geom_rgba[m.geom('tower_collision').id] = [0.18, 0.8, 0.44, 0.35]
        self.look = None

    def draw(self, rec, i, scale=0.008):
        d, m = self.d, self.m
        d.qpos[self.bq:self.bq + 3] = rec["base_pos"][i]
        d.qpos[self.bq + 3:self.bq + 7] = rec["base_quat"][i]
        d.qpos[self.kq:self.kq + 3] = rec["ball_pos"][i]
        d.qpos[self.kq + 3:self.kq + 7] = rec["ball_quat"][i]
        for k in range(3):
            d.qpos[self.wq[k]] = rec["wheel_q"][i][k]
        mujoco.mj_forward(m, d)
        tgt = rec["ball_center"][i] + np.array([0, 0, 0.06])
        self.look = tgt if self.look is None else self.look + 0.45 * (tgt - self.look)
        self.cam.lookat[:] = self.look
        self.r.update_scene(d, camera=self.cam)
        sc = self.r.scene
        for k in range(3):
            c = rec["contacts"][i, k]
            if c[8] < 0.5:
                continue
            p, F, fn, ft = c[0:3], c[3:6], c[6], c[7]
            # sphere at the contact point
            self._geom(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.007, 0, 0], p, np.eye(3).ravel(), [1, 0.15, 0.15, 1])
            n = F / (np.linalg.norm(F) + 1e-9)
            # decompose: normal part (along ball-centre -> contact line) and friction part
            nrm = p - rec["ball_center"][i]
            nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
            if F @ nrm > 0:
                F = -F            # draw the force acting on the ball (compressive normal)
            Fn = (F @ nrm) * nrm
            Ft = F - Fn
            self._arrow(sc, p, Fn * scale, [0.2, 0.75, 1.0, 1])
            self._arrow(sc, p, Ft * scale * 4.0, [1.0, 0.65, 0.1, 1])
        return self.r.render()

    @staticmethod
    def _geom(sc, typ, size, pos, mat, rgba):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, typ, np.array(size, dtype=float), np.array(pos, dtype=float), np.array(mat, dtype=float), np.array(rgba, dtype=np.float32))
        sc.ngeom += 1

    def _arrow(self, sc, p, v, rgba):
        if np.linalg.norm(v) < 0.004 or sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, np.array(p, dtype=float), np.array(p + v, dtype=float))
        sc.ngeom += 1


def hud(img, label, color, sub, rec, i, title, scenario_note, fell_at, tcolor_ok):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    dr.rounded_rectangle((10, 10, PW - 10, 112), 10, fill=(10, 14, 22, 190))
    dr.rounded_rectangle((10, PH - 112, PW - 10, PH - 34), 10, fill=(10, 14, 22, 190))
    dr.rectangle((0, PH - 26, PW, PH), fill=(10, 14, 22, 220))
    dr.rectangle((10, 10, 16, 112), fill=color)
    dr.text((26, 16), label, font=font(24, "bold"), fill=color)
    dr.text((26, 48), sub, font=font(15), fill=C["dim"])
    dr.text((26, 72), title, font=font(17, "bold"), fill=C["ink"])
    t, tilt, sp, cmd = rec["t"][i], rec["tilt"][i], rec["speed"][i], rec["cmd"][i]
    dr.text((26, PH - 106), f"t = {t:5.2f} s     speed {sp:4.2f} m/s (cmd {cmd:4.2f})", font=font(17, "mono"), fill=C["ink"])
    dr.text((26, PH - 80), f"tilt {tilt:5.1f} deg", font=font(17, "mono"), fill=C["bad"] if tilt > 25 else C["ink"])
    if rec["push"][i] > 0:
        dr.text((300, PH - 80), f"PUSH {rec['push'][i]:.0f} N", font=font(17, "bold"), fill=C["warn"])
    if fell_at >= 0 and i >= fell_at:
        dr.text((PW - 150, 18), "FELL", font=font(30, "bold"), fill=C["bad"])
    dr.text((14, PH - 24), "SIMULATION ONLY | blue = normal force, orange = friction (4x), red dot = contact | " + scenario_note,
            font=font(11), fill=C["dim"])
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))


def load(name):
    f = OUT / "rec" / f"{name}.npz"
    return {k: v for k, v in np.load(f).items()}


def label_for(name):
    eng = name.split("_")[0]
    if eng.startswith("mj"):
        return "MuJoCo (CPU)", C["model"], "elliptic cone, anisotropic wheel friction, RK4/implicitfast, 500 Hz"
    if "naive" in eng:
        return "PhysX, naive import", C["bad"], "capsule wheels, isotropic friction (what a plain MJCF import gives)"
    return "PhysX (Isaac Sim)", C["px"], "passive-roller omni wheel emulation, TGS, 500 Hz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", default=None)
    ap.add_argument("--design", default="nom")
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="same LQR gains")
    ap.add_argument("--slow", action="store_true", help="play every control step at 30 fps (0.3x)")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    names = [a.left] + ([a.right] if a.right else [])
    recs = [load(n) for n in names]
    panels = [Panel(a.design) for _ in names]
    n = max(len(r["t"]) for r in recs)
    idx = np.arange(n) if a.slow else np.unique(np.round(np.arange(0, n, 100.0 / a.fps)).astype(int))
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    out = OUT / "videos" / f"{a.out}.mp4"
    import imageio_ffmpeg
    W = PW * len(names)
    p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{PH}",
                          "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow",
                          "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in idx:
        frames = []
        for nm, rec, pn in zip(names, recs, panels):
            j = min(i, len(rec["t"]) - 1)
            img = pn.draw(rec, j)
            lab, col, sub = label_for(nm)
            frames.append(hud(img, lab, col, sub, rec, j, a.title, a.note, int(rec["fell_at"]), True))
        p.stdin.write(np.hstack(frames).tobytes())
    p.stdin.close()
    p.wait()
    print(out, out.stat().st_size / 1e6, "MB", len(idx), "frames")


if __name__ == "__main__":
    main()
