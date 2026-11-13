"""3-D pictures of parametrised ballbot layouts: a visual-only MuJoCo model generated from the layout parameters (see ballbot_layout.py).

Every length follows the parameters (ball radius, wheel radius and width, wheel count, azimuths and polar angles, drive type, arrestor reach, body size):
  * drive unit: a double omni-wheel (two staggered roller rings) touching the ball; its shaft lies along the ball's meridian at the contact, so the wheel drives the
    ball sideways (azimuthally). Direct drive: the motor is coaxial ABOVE the wheel (toward the body) and bolts to the collar. Geared: the motor sits further out and up,
    parallel to the wheel shaft, with a belt between two pulleys. If the motors would collide above the ball (wheels very high on the ball) they are shortened, and
    only if that is not enough are they mounted below the wheel (reported in checks()).
  * collar: a plate and ring under the body that carries the drive units and the ball guards.
  * ball guards (arrestors): curved arms from the collar that follow the ball surface with a small gap, between the wheels, down past the equator to arr_theta.
  * body: torso with a face screen, shoulders, two arms (one carrying a tray) and an antenna; an original design in the style of a service ballbot.
  python scripts/study/layout_render.py hero OUT_prefix        # large front and rear views of the nominal robot + cutaways of the drives
  python scripts/study/layout_render.py gallery OUT.png        # the example layouts
  python scripts/study/layout_render.py proof OUT_prefix       # one-parameter-at-a-time strips with geometry checks
Simulation-only visual model; it is not the physics model."""
import sys
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ballbot_layout as BL

FD = Path("C:/Windows/Fonts")
Z = np.array([0.0, 0.0, 1.0])
COL = dict(ball="0.15 0.16 0.18", seam="0.32 0.34 0.38", wheel="0.16 0.42 0.72", roller="0.07 0.08 0.1", hub="0.1 0.11 0.13", motor="0.72 0.74 0.77", gear="0.28 0.3 0.33",
           steel="0.42 0.45 0.5", dark="0.12 0.13 0.15", white="0.94 0.95 0.97", blue="0.14 0.36 0.62", arm="0.15 0.4 0.66", armj="0.09 0.24 0.42", screen="0.84 0.93 0.98",
           belt="0.05 0.05 0.05", pulley="0.9 0.55 0.15", pad="0.2 0.5 0.8", tray="0.8 0.82 0.85", red="0.85 0.1 0.1")


def _f(sz, bold=False):
    try:
        return ImageFont.truetype(str(FD / ("segoeuib.ttf" if bold else "segoeui.ttf")), sz)
    except OSError:
        return ImageFont.load_default()


def v3(a):
    return " ".join(f"{x:.5f}" for x in a)


class Parts:
    def __init__(self):
        self.g = []

    def _add(self, typ, body, col, a=1.0):
        self.g.append(f'<geom type="{typ}" {body} rgba="{COL.get(col, col)} {a}" contype="0" conaffinity="0"/>')

    def cyl(self, p0, p1, r, col, a=1.0): self._add("cylinder", f'fromto="{v3(np.asarray(p0, float))} {v3(np.asarray(p1, float))}" size="{r:.5f}"', col, a)
    def cap(self, p0, p1, r, col, a=1.0): self._add("capsule", f'fromto="{v3(np.asarray(p0, float))} {v3(np.asarray(p1, float))}" size="{r:.5f}"', col, a)
    def sph(self, p, r, col, a=1.0): self._add("sphere", f'pos="{v3(np.asarray(p, float))}" size="{r:.5f}"', col, a)
    def box(self, p, h, col, a=1.0): self._add("box", f'pos="{v3(np.asarray(p, float))}" size="{v3(np.asarray(h, float))}"', col, a)

    def chain(self, pts, r, col, jcol=None, a=1.0):
        for q0, q1 in zip(pts[:-1], pts[1:]): self.cap(q0, q1, r, col, a)
        if jcol:
            for q in pts[1:-1]: self.sph(q, r * 1.12, jcol, a)


def build(L, cutaway=False):
    """Returns (list of geom xml, info dict with the generated dimensions and flags)."""
    R, rw, ww, n = L["R"], L["rw"], L["ww"], L["n"]; sc = R / 0.09; c = np.array([0, 0, R]); p, t, J = BL.geometry(L); P = Parts(); info = dict(motor_below=False, motor_len_mm=[])
    P.sph(c, R, "ball"); P.cyl(c - Z * 0.0009 * sc, c + Z * 0.0009 * sc, R * 1.003, "seam")
    for ang in (0.0, np.pi / 2):
        ax = np.array([np.cos(ang), np.sin(ang), 0.0]); P.cyl(c - ax * 0.0009 * sc, c + ax * 0.0009 * sc, R * 1.003, "seam")
    phis = np.radians(L["phi"]); order = np.argsort(phis); gaps = np.diff(np.r_[phis[order], phis[order][0] + 2 * np.pi]); gmin = float(gaps.min()) if n > 1 else 2 * np.pi
    rm = rw * 0.62; b0 = ww * 0.62; tops = []; mounts = []
    for i in range(n):
        pn = p[i] / np.linalg.norm(p[i]); tt = t[i]; m = Z - (Z @ pn) * pn; m /= np.linalg.norm(m)       # m: wheel shaft direction, up along the meridian (toward the body)
        hub = c + (R + rw) * pn
        for s, off in ((-1, 0.0), (1, 0.5)):                                                             # double omni-wheel, roller rings staggered by half a pitch
            d0 = hub + m * (s * ww * 0.27); P.cyl(d0 - m * ww * 0.2, d0 + m * ww * 0.2, rw * 0.78, "wheel")
            for k in range(10):
                a = 2 * np.pi * (k + off) / 10; cen = d0 + rw * 0.87 * (np.cos(a) * pn + np.sin(a) * tt); d = -np.sin(a) * pn + np.cos(a) * tt
                P.cap(cen - d * rw * 0.13, cen + d * rw * 0.13, rw * 0.13, "roller")
        P.cyl(hub - m * ww * 0.6, hub + m * ww * 0.6, rw * 0.3, "hub")
        if L["drive"] == "direct":
            Lm = 0.075 * sc; md = m

            def clear(length, direction):
                e = hub + direction * (b0 + length); return np.hypot(e[0], e[1]) * np.sin(gmin / 2) >= rm * 1.08
            while Lm > 0.04 * sc and not clear(Lm, m): Lm -= 0.005 * sc
            if not clear(Lm, m): md = -m; Lm = 0.075 * sc; info["motor_below"] = True                  # no room above the ball: mount the motor on the other end of the shaft
            P.cyl(hub + md * b0, hub + md * (b0 + 0.02 * sc), rm * 1.08, "gear"); P.cyl(hub + md * (b0 + 0.02 * sc), hub + md * (b0 + Lm), rm, "motor")
            P.cyl(hub + md * (b0 + Lm), hub + md * (b0 + Lm + 0.006 * sc), rm * 0.8, "dark"); mp = hub + md * (b0 + Lm * 0.55); top = hub + md * (b0 + Lm + 0.006 * sc)
            info["motor_len_mm"].append(round(Lm * 1000))
        else:                                                                                            # geared: parallel-shaft motor further out and up, belt between pulleys
            Lm = 0.07 * sc; D = 2.3 * rw; mpos = hub + pn * D; h0 = b0 + 0.012 * sc; h1 = b0 + 0.028 * sc
            P.cyl(hub + m * b0, hub + m * h0, rw * 0.5, "gear"); P.cyl(hub + m * h0, hub + m * h1, rw * 0.58, "pulley"); P.cyl(mpos + m * h0, mpos + m * h1, rw * 0.26, "pulley")
            for sg in (-1, 1):
                P.cap(hub + m * (h0 + h1) / 2 + tt * sg * rw * 0.58, mpos + m * (h0 + h1) / 2 + tt * sg * rw * 0.26, 0.0032 * sc, "belt")
            P.cap(hub + m * b0 * 0.9, mpos + m * b0 * 0.9, 0.006 * sc, "steel")                        # carrier plate between the wheel bearing and the motor
            P.cyl(mpos + m * (h1 + 0.002 * sc), mpos + m * (h1 + 0.02 * sc), rm * 1.08, "gear"); P.cyl(mpos + m * (h1 + 0.02 * sc), mpos + m * (h1 + Lm), rm, "motor")
            mp = mpos + m * (h1 + Lm * 0.5); top = mpos + m * (h1 + Lm); info["motor_len_mm"].append(round(Lm * 1000))
        tops.append(max(top[2], hub[2] + rw)); mounts.append(mp)
    zc = max(tops) + 0.014 * sc; rc = L["body_r"] * 0.98                                                # collar plate under the body
    P.cyl([0, 0, zc - 0.005 * sc], [0, 0, zc + 0.005 * sc], rc, "gear")
    ring = [np.array([rc * np.cos(a), rc * np.sin(a), zc]) for a in np.linspace(0, 2 * np.pi, 33)]; P.chain(ring, 0.008 * sc, "steel")
    for mp in mounts:                                                                                    # bracket from each motor up to the collar
        rr = np.hypot(mp[0], mp[1]); k = min(1.0, rc * 0.9 / max(rr, 1e-9)); P.cap(mp, [mp[0] * k, mp[1] * k, zc], 0.0065 * sc, "steel")
    if L["arr_theta"] is not None:                                                                       # ball guards: curved arms hugging the ball between the wheels
        ra = R + L["arr_gap_mm"] / 1000.0 + 0.007 * sc
        psis = [phis[order][k] + gaps[k] * f for k in range(n) for f in ((1 / 3, 2 / 3) if gaps[k] > np.radians(100) else (0.5,))]   # guards sit in the gaps between wheels, symmetric
        for psi in psis:
            th = np.radians(np.linspace(min(62.0, L["arr_theta"] - 20), L["arr_theta"], 10))
            pts = [c + ra * np.array([np.sin(a) * np.cos(psi), np.sin(a) * np.sin(psi), np.cos(a)]) for a in th]
            P.cap([rc * np.cos(psi), rc * np.sin(psi), zc], pts[0], 0.0065 * sc, "steel"); P.chain(pts, 0.0075 * sc, "steel"); P.sph(pts[-1], 0.013 * sc, "pad")
    # ---- body: torso, face, shoulders, arms, tray, antenna
    a = 0.22 if cutaway else 1.0; zb = zc + 0.005 * sc; hy = L["body_r"]; hx = hy * 0.86; bh = L["body_h"]; band = 0.07 * sc
    P.box([0, 0, zb + band / 2], [hx * 1.03, hy * 1.03, band / 2], "blue", a); P.box([0, 0, zb + band + (bh - band) / 2], [hx, hy, (bh - band) / 2], "white", a)
    for sx in (-1, 1):                                                                                   # rounded vertical edges
        for sy in (-1, 1): P.cap([sx * hx * 0.97, sy * hy * 0.97, zb + band + 0.012 * sc], [sx * hx * 0.97, sy * hy * 0.97, zb + bh - 0.012 * sc], 0.014 * sc, "white", a)
    P.box([0, 0, zb + bh + 0.006 * sc], [hx * 0.9, hy * 0.9, 0.006 * sc], "steel", a)
    zf = zb + bh * 0.7; xf = hx + 0.008 * sc
    P.box([xf, 0, zf], [0.009 * sc, hy * 0.68, bh * 0.21], "dark", a); P.box([xf + 0.0095 * sc, 0, zf], [0.002 * sc, hy * 0.58, bh * 0.175], "screen", a)
    xe = xf + 0.0115 * sc
    for s, dz in ((-1, 0.0), (1, 0.012 * sc)):                                                          # eyes (rings) and brows, one brow raised
        e = np.array([xe, s * hy * 0.25, zf + bh * 0.035]); P.cyl(e, e + [0.002 * sc, 0, 0], bh * 0.045, "dark", a); P.cyl(e + [0.002 * sc, 0, 0], e + [0.003 * sc, 0, 0], bh * 0.026, "screen", a)
        P.cap([xe + 0.002 * sc, s * hy * 0.38, zf + bh * 0.115 + dz], [xe + 0.002 * sc, s * hy * 0.12, zf + bh * 0.105 + dz * 1.6], 0.004 * sc, "dark", a)
    P.cap([xe + 0.002 * sc, -hy * 0.22, zf - bh * 0.105], [xe + 0.002 * sc, hy * 0.22, zf - bh * 0.105], 0.0035 * sc, "dark", a)
    P.box([hx + 0.018 * sc, 0, zb + bh * 0.36], [0.018 * sc, hy * 0.72, bh * 0.075], "gear", a); P.cyl([hx + 0.03 * sc, hy * 0.42, zb + bh * 0.42], [hx + 0.045 * sc, hy * 0.42, zb + bh * 0.42], bh * 0.085, "white", a)
    P.box([-hx - 0.006 * sc, 0, zb + bh * 0.58], [0.007 * sc, hy * 0.62, bh * 0.3], "dark", a)
    for dzf in (0.13, -0.1): P.cyl([-hx - 0.013 * sc, 0, zb + bh * (0.58 + dzf)], [-hx - 0.016 * sc, 0, zb + bh * (0.58 + dzf)], hy * 0.34, "gear", a)
    P.cyl([-hx * 0.45, hy * 0.5, zb + bh], [-hx * 0.45, hy * 0.5, zb + bh + 0.11 * sc], 0.007 * sc, "arm", a)
    zs = zb + bh * 0.5
    for s in (-1, 1):
        P.cyl([0, s * hy, zs], [0, s * (hy + 0.014 * sc), zs], 0.042 * sc, "white", a); P.cyl([0, s * (hy + 0.014 * sc), zs], [0, s * (hy + 0.018 * sc), zs], 0.024 * sc, "pulley", a)
    o = lambda dx, dy, dz, s: np.array([dx * sc, s * (hy + dy * sc), zs + dz * sc])
    right = [o(0, 0.02, 0, -1), o(0.01, 0.075, -0.04, -1), o(0.04, 0.11, -0.085, -1), o(0.09, 0.125, -0.085, -1), o(0.135, 0.12, -0.045, -1), o(0.16, 0.11, 0.005, -1), o(0.17, 0.105, 0.05, -1)]
    left = [o(0, 0.02, 0, 1), o(0.0, 0.07, -0.045, 1), o(0.02, 0.09, -0.105, 1), o(0.06, 0.075, -0.15, 1), o(0.1, 0.045, -0.165, 1)]
    if L.get("arms", True):                                                                              # arms are optional (visual only unless the spec gives them mass)
        P.chain(right, 0.017 * sc, "arm", "armj", a); P.chain(left, 0.017 * sc, "arm", "armj", a)
        hand = right[-1]; P.cyl(hand + [0, 0, 0.018 * sc], hand + [0, 0, 0.022 * sc], 0.062 * sc, "tray", a)
        P.cyl(hand + [0.01 * sc, 0, 0.022 * sc], hand + [0.01 * sc, 0, 0.06 * sc], 0.003 * sc, "screen", a); P.cyl(hand + [0.01 * sc, 0, 0.06 * sc], hand + [0.01 * sc, 0, 0.085 * sc], 0.022 * sc, "screen", 0.55 * a)
    if cutaway: P.sph([0, 0, zb + bh * 0.5], 0.009 * sc, "red")
    info.update(collar_z=zc, top_z=zb + bh, body_clear_mm=round((zc - 0.005 * sc - max(tops)) * 1000, 1))
    return P.g, info


def layout_xml(L, cutaway=False):
    g, _ = build(L, cutaway)
    return f"""<mujoco><visual><global offwidth="1400" offheight="1400"/><headlight ambient="0.42 0.42 0.44" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/><quality shadowsize="4096"/></visual>
<asset><texture type="skybox" builtin="gradient" rgb1="0.78 0.8 0.84" rgb2="0.96 0.96 0.97" width="256" height="256"/>
<texture name="chk" type="2d" builtin="checker" rgb1="0.9 0.91 0.93" rgb2="0.83 0.84 0.87" width="256" height="256"/><material name="floor" texture="chk" texrepeat="30 30" reflectance="0.0"/></asset>
<worldbody><light pos="0.9 -0.9 2.2" dir="-0.35 0.35 -1" diffuse="0.75 0.75 0.75" castshadow="true"/><geom type="plane" size="3 3 0.1" material="floor" contype="0" conaffinity="0"/>
{chr(10).join(g)}</worldbody></mujoco>"""


def render(L, w=560, h=620, az=150.0, el=-10.0, dist=1.15, look_z=None, cutaway=False):
    """az 180 looks at the robot's face; 0 at its back."""
    m = mujoco.MjModel.from_xml_string(layout_xml(L, cutaway)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, h, w); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.27 if look_z is None else look_z]; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, camera=cam); img = Image.fromarray(r.render()); r.close(); return img


def checks(L):
    """Geometry sanity checks of a generated layout (mm; negative = intersecting)."""
    R, rw = L["R"], L["rw"]; p, t, J = BL.geometry(L); hubs = (R + rw) * p / np.linalg.norm(p, axis=1, keepdims=True)
    gap = max(abs(np.linalg.norm(h) - (R + rw)) for h in hubs) * 1000
    wwc = min((np.linalg.norm(hubs[i] - hubs[j]) - 2 * rw for i in range(len(hubs)) for j in range(i + 1, len(hubs))), default=9.9) * 1000
    _, info = build(L)
    return dict(ball_gap_mm=round(float(gap), 3), wheel_clear_mm=round(float(wwc), 1), body_clear_mm=info["body_clear_mm"], motor_below=info["motor_below"], motor_len_mm=min(info["motor_len_mm"]))


def label(im, title, sub=None, foot=None, ok=None):
    d = ImageDraw.Draw(im); H = 48 if sub else 28
    d.rectangle([0, 0, im.width, H], fill=(16, 22, 34)); d.text((10, 5), title, fill=(236, 241, 248), font=_f(16, True))
    if sub: d.text((10, 28), sub, fill=(150, 163, 182), font=_f(11))
    if foot:
        d.rectangle([0, im.height - 22, im.width, im.height], fill=(16, 22, 34)); d.text((10, im.height - 19), foot, fill=(150, 163, 182) if ok is None else ((94, 224, 138) if ok else (255, 107, 107)), font=_f(11, ok is not None))
    return im


def scaled(name, k, **kw):
    L = BL.make(name, *BL.even(3), **kw)
    for key in ("R", "rw", "ww", "body_r", "body_h"): L[key] *= k
    return L


GALLERY = [
    BL.family("A  nominal: 3 wheels at 51 deg", 3, arr_theta=112.0),
    BL.family("B  3 wheels higher on the ball (40 deg)", 3, 40.0, arr_theta=112.0),
    BL.family("C  3 wheels lower on the ball (62 deg)", 3, 62.0, arr_theta=112.0),
    BL.family("D  4 wheels, even", 4, arr_theta=112.0),
    BL.family("E  5 wheels, even", 5, arr_theta=112.0),
    BL.family("F  6 wheels in two rings (42 / 60 deg), belt drive", 6, 42.0, 60.0, drive="geared", N=4.0, eta=0.85, tau_max=8.0, rotor_J=0.0004, arr_theta=112.0),
    BL.family("G  belt drive (N = 4), 3 wheels", 3, drive="geared", N=4.0, eta=0.85, tau_max=8.0, rotor_J=0.0004, arr_theta=112.0),
    scaled("H  small robot (ball 6.75 cm)", 0.75, arr_theta=112.0),
    scaled("I  large robot (ball 13.5 cm)", 1.5, arr_theta=112.0),
]
REJECTED = [
    BL.make("3 wheels grouped on one side", [51] * 3, [130, 180, 230], arr_theta=112.0),
    BL.make("4 wheels at alternating heights", [40, 62, 40, 62], [0, 90, 180, 270], arr_theta=112.0),
    BL.make("3 wheels, uneven spacing", [51] * 3, [0, 110, 240], arr_theta=112.0),
    BL.family("3 wheels too high (25 deg)", 3, 25.0, arr_theta=112.0),
    BL.family("8 wheels (they touch)", 8, arr_theta=112.0),
    BL.family("6 wheels in two rings, direct drive (no room for motors above)", 6, 42.0, 60.0, arr_theta=112.0),
]


def _row(ims, out):
    W, H = ims[0].size; sheet = Image.new("RGB", (W * len(ims), H)); [sheet.paste(im, (i * W, 0)) for i, im in enumerate(ims)]; sheet.save(out); print("saved", out)


def hero(prefix):
    L = GALLERY[0]
    _row([label(render(L, 900, 1000, az=152, el=-8, dist=1.05, look_z=0.26), "Front"), label(render(L, 900, 1000, az=28, el=-8, dist=1.05, look_z=0.26), "Rear"),
          label(render(L, 900, 1000, az=152, el=-14, dist=0.62, look_z=0.13, cutaway=True), "Drive units, collar and ball guards (body cut away)")], prefix + "_hero.png")
    G = GALLERY[6]
    _row([label(render(G, 900, 1000, az=152, el=-8, dist=1.05, look_z=0.26), "Geared (belt drive): front"), label(render(G, 900, 1000, az=120, el=-12, dist=0.62, look_z=0.14, cutaway=True), "Belt drive close-up (body cut away)"),
          label(render(GALLERY[5], 900, 1000, az=152, el=-14, dist=0.62, look_z=0.13, cutaway=True), "6 wheels in two rings: drive units close-up")], prefix + "_drives.png")


def gallery(out, layouts=GALLERY, cols=3):
    ims = []
    for L in layouts:
        s = BL.summary(L); c = checks(L)
        sub = f"ball R {L['R'] * 100:.1f} cm | {L['n']} wheels | {'direct' if L['drive'] == 'direct' else 'geared N=' + format(L['N'], 'g')} | weakest/strongest push {s['auth_aniso']:.2f}" + (" | motors below wheels" if c["motor_below"] else "")
        v = BL.validate(L)
        bad = [k for k, x in v["rules"].items() if not x] + (["motors do not fit above the wheels"] if c["motor_below"] else [])
        ims.append(label(render(L, 620, 700, az=150, el=-8, dist=1.75, look_z=0.34), L["name"], sub, "passes all layout rules" if not bad else "FAILS: " + "; ".join(bad), not bad))
    rows = (len(ims) + cols - 1) // cols; W, H = ims[0].size; sheet = Image.new("RGB", (W * cols, H * rows), (16, 22, 34))
    for i, im in enumerate(ims): sheet.paste(im, ((i % cols) * W, (i // cols) * H))
    sheet.save(out); print("saved", out, sheet.size)


def rejected(out):
    ims = []
    for L in REJECTED:
        v = BL.validate(L); bad = [k for k, x in v["rules"].items() if not x] + (["motors do not fit above the wheels"] if checks(L)["motor_below"] else [])
        ims.append(label(render(L, 520, 560, az=150, el=-14, dist=0.66, look_z=0.14, cutaway=True), "REJECTED: " + L["name"], None, "fails: " + "; ".join(bad), False))
    _row(ims, out)


def sweep_strips(out):
    def mk(name, **kw):
        th = kw.pop("theta", 51.0); n = kw.pop("n", 3); k = kw.pop("k", 1.0); arr = kw.pop("arr", 112.0)
        L = BL.make(name, *BL.even(n, th), arr_theta=arr, **kw)
        for key in ("R", "rw", "ww", "body_r", "body_h"): L[key] *= k
        return L
    rows = [("wheel height on the ball (polar angle), body cut away", [mk(f"{t} deg", theta=float(t)) for t in (30, 40, 51, 60, 70)], True, 0.62, 0.13),
            ("number of wheels (even), body cut away", [mk(f"{n} wheels", n=n) for n in (3, 4, 5, 6)], True, 0.62, 0.13),
            ("robot size (ball radius, all lengths scaled), one camera", [mk(f"ball {9 * k:.1f} cm", k=k) for k in (0.67, 1.0, 1.33, 1.67)], False, 2.0, 0.36),
            ("drive and ball guards, body cut away", [mk("direct, no guards", arr=None), mk("direct + guards to 112 deg"), mk("guards to 95 deg", arr=95.0), mk("belt drive N=4", drive="geared", N=4.0, eta=0.85), mk("belt drive, 4 wheels", n=4, drive="geared", N=4.0, eta=0.85)], True, 0.62, 0.13)]
    W, H = 420, 440; maxc = max(len(r[1]) for r in rows); sheet = Image.new("RGB", (W * maxc, (H + 34) * len(rows)), (16, 22, 34)); d = ImageDraw.Draw(sheet); info = []
    for ri, (title, Ls, cut, dist, lz) in enumerate(rows):
        y0 = ri * (H + 34); d.text((10, y0 + 8), title, fill=(236, 241, 248), font=_f(17, True))
        for ci, L in enumerate(Ls):
            c = checks(L); foot = f"wheel-ball gap {c['ball_gap_mm']:.2f} mm | wheel clear {c['wheel_clear_mm']:.0f} mm | motor {c['motor_len_mm']} mm" + (" BELOW" if c["motor_below"] else "")
            sheet.paste(label(render(L, W, H, az=150, el=-14, dist=dist, look_z=lz, cutaway=cut), L["name"], None, foot), (ci * W, y0 + 34)); info.append((title, L["name"], c))
    sheet.save(out); print("saved", out, sheet.size)
    for tt, nm, c in info: print(f"{tt[:30]:30s} {nm:28s} {c}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "hero": hero(sys.argv[2])
    elif cmd == "gallery": gallery(sys.argv[2])
    elif cmd == "rejected": rejected(sys.argv[2])
    elif cmd == "proof": sweep_strips(sys.argv[2] + "_sweep.png")
    else: print(__doc__)
