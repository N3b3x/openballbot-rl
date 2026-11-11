"""Write the README and docs diagrams as hand-laid-out SVG (docs/assets/*-light.svg and *-dark.svg).

Each diagram is written twice from one layout: a light variant for GitHub's light theme and a dark variant for its dark
themes. The Markdown picks one with <picture> and prefers-color-scheme. The SVGs use only system font stacks, no external
fonts, images or scripts (GitHub sanitises SVG), and have a transparent background outside their cards.

Numbers in results-*.svg come from Readme.md and docs/research/tracking_study_log.md (S-numbers shown next to each);
all of them are simulation results.

Usage (any Python 3.8+, no dependencies):
    python scripts/docs/make_diagrams.py
"""
import os
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "assets")

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# GitHub Primer-like colours: each pair passes contrast on its own theme background (#ffffff / #0d1117).
THEMES = {
    "light": dict(text="#1f2328", muted="#59636e", faint="#818b98", card="#ffffff", stroke="#d1d9e0", panel="#f6f8fa",
                  arrow="#59636e", onacc="#ffffff", tint=0.10, track="#eaeef2",
                  blue="#0969da", green="#1a7f37", purple="#8250df", orange="#bc4c00", pink="#bf3989",
                  teal="#1b7c83", yellow="#9a6700", red="#cf222e",
                  hero0="#eef5ff", hero1="#f5f0ff", grid="#0969da", gridop=0.07),
    "dark": dict(text="#f0f6fc", muted="#9198a1", faint="#6e7681", card="#151b23", stroke="#3d444d", panel="#0d1117",
                 arrow="#9198a1", onacc="#0d1117", tint=0.16, track="#262c36",
                 blue="#4493f8", green="#3fb950", purple="#ab7df8", orange="#f0883e", pink="#f778ba",
                 teal="#39c5cf", yellow="#d29922", red="#f85149",
                 hero0="#0d1a2d", hero1="#1a1430", grid="#4493f8", gridop=0.10),
}


def circ(cx, cy, r):
    return f"M{cx - r} {cy}a{r} {r} 0 1 0 {2 * r} 0a{r} {r} 0 1 0 {-2 * r} 0"


# Stroke icons in a 24 x 24 box.
ICONS = {
    "doc": "M6 3h8l4 4v14H6z M14 3v4h4 M9 12h6 M9 16h6",
    "cube": "M12 3l8 4.5v9L12 21l-8-4.5v-9z M12 12l8-4.5 M12 12v9 M12 12L4 7.5",
    "chip": "M7 7h10v10H7z M10 10h4v4h-4z M9 3v4 M15 3v4 M9 17v4 M15 17v4 M3 9h4 M3 15h4 M17 9h4 M17 15h4",
    "chart": "M4 20h16 M7 20v-6 M12 20V8 M17 20V11",
    "radar": "M12 20L5 7 M12 20l7-13 M6.6 10a8 8 0 0 1 10.8 0 M8.6 14a5 5 0 0 1 6.8 0",
    "package": "M3 7l9-4 9 4v10l-9 4-9-4z M3 7l9 4 9-4 M12 11v10",
    "nodes": circ(12, 5.5, 2.5) + " " + circ(5.5, 18, 2.5) + " " + circ(18.5, 18, 2.5) + " M10.8 7.7L6.7 15.8 M13.2 7.7l4.1 8.1 M8 18h8",
    "route": circ(6, 19, 2) + " " + circ(18, 5, 2) + " M8 19h7.5a3.5 3.5 0 0 0 0-7h-7a3.5 3.5 0 0 1 0-7H16",
    "wave": "M3 17c2.5 0 3.5-10 6-10s3.5 10 6 10 3.5-10 6-10",
    "matrix": "M7 4H5v16h2 M17 4h2v16h-2 M9.5 9h1 M13.5 9h1 M9.5 15h1 M13.5 15h1",
    "net": circ(5, 6, 1.6) + " " + circ(5, 12, 1.6) + " " + circ(5, 18, 1.6) + " " + circ(19, 9, 1.6) + " " + circ(19, 15, 1.6)
           + " M6.6 6l10.8 3 M6.6 12l10.8-3 M6.6 12l10.8 3 M6.6 18l10.8-3 M6.6 6l10.8 9 M6.6 18l10.8-9",
    "gauge": "M4 17a8 8 0 1 1 16 0 M12 17l4-5 M4 17h2 M18 17h2",
    "shield": "M12 3l7 3v5c0 5-3 8.5-7 10-4-1.5-7-5-7-10V6z M9 12l2 2 4-4",
    "wheel": circ(12, 12, 8) + " " + circ(12, 12, 2) + " M12 4v3 M12 17v3 M4 12h3 M17 12h3",
    "imu": circ(12, 12, 7) + " " + circ(12, 12, 1.5) + " M12 2v4 M12 18v4 M2 12h4 M18 12h4",
    "ball": circ(12, 13, 8) + " M4.5 11c4 3 11 3 15 0 M12 5v-2",
    "check": "M9 4h6v3H9z M7 5H5v16h14V5h-2 M9 14l2 2 4-4",
    "estim": "M3 17l5-6 4 3 5-8 4 5 M3 21h18",
}


class Svg:
    def __init__(self, w, h, theme, title, desc):
        self.w, self.h, self.t = w, h, THEMES[theme]
        self.theme = theme
        self.parts = []
        self.defs = []
        self.title, self.desc = title, desc

    def add(self, s):
        self.parts.append(s)

    def c(self, key):
        return self.t.get(key, key)

    # primitives -------------------------------------------------------------------------------------------------
    def text(self, x, y, s, size=13, color="text", weight=400, anchor="start", mono=False, italic=False, op=None, ls=None):
        fam = MONO if mono else SANS
        extra = ""
        if italic:
            extra += ' font-style="italic"'
        if op is not None:
            extra += f' opacity="{op}"'
        if ls is not None:
            extra += f' letter-spacing="{ls}"'
        self.add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{self.c(color)}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def rich(self, x, y, runs, size=13, anchor="start"):
        """runs: list of (text, color, weight, size or None, mono)."""
        sp = []
        for r in runs:
            s, col, wt = r[0], r[1], r[2]
            sz = r[3] if len(r) > 3 and r[3] else size
            fam = MONO if len(r) > 4 and r[4] else SANS
            sp.append(f'<tspan fill="{self.c(col)}" font-weight="{wt}" font-size="{sz}" font-family="{fam}">{escape(s)}</tspan>')
        self.add(f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" text-anchor="{anchor}">{"".join(sp)}</text>')

    def rect(self, x, y, w, h, rx=12, fill="card", stroke="stroke", sw=1, op=None, dash=None):
        extra = f' fill-opacity="{op}"' if op is not None else ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        st = f' stroke="{self.c(stroke)}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{self.c(fill)}"{st}{extra}/>')

    def icon(self, name, cx, cy, size=18, color="onacc", sw=2):
        s = size / 24.0
        self.add(f'<g transform="translate({cx - size / 2} {cy - size / 2}) scale({s:.4f})" fill="none" stroke="{self.c(color)}" '
                 f'stroke-width="{sw / s * s:.2f}" stroke-linecap="round" stroke-linejoin="round"><path d="{ICONS[name]}"/></g>')

    def arrow(self, d, color="arrow", sw=1.8, dash=None, head=True):
        mid = f"ah-{color}"
        if head and mid not in [m for m, _ in self.defs]:
            self.defs.append((mid, f'<marker id="{mid}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
                                   f'orient="auto-start-reverse"><path d="M1 1.5L8.5 5L1 8.5z" fill="{self.c(color)}"/></marker>'))
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#{mid})"' if head else ""
        self.add(f'<path d="{d}" fill="none" stroke="{self.c(color)}" stroke-width="{sw}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{extra}{mk}/>')

    def pill(self, x, y, s, color="blue", size=11, solid=False, anchor="start", mono=False, h=20):
        w = len(s) * size * (0.62 if mono else 0.53) + 18
        if anchor == "end":
            x = x - w
        elif anchor == "middle":
            x = x - w / 2
        if solid:
            self.rect(x, y, w, h, rx=h / 2, fill=color, stroke=None)
            self.text(x + w / 2, y + h / 2 + size * 0.36, s, size, "onacc", 600, "middle", mono=mono)
        else:
            self.rect(x, y, w, h, rx=h / 2, fill=color, stroke=None, op=self.t["tint"] + 0.04)
            self.text(x + w / 2, y + h / 2 + size * 0.36, s, size, color, 600, "middle", mono=mono)
        return w

    # composite card -----------------------------------------------------------------------------------------------
    def card(self, x, y, w, h, acc, title, lines=(), foot=None, icon=None, num=None, line_size=12.5, top=None):
        self.rect(x, y, w, h, rx=14)
        hh = 46
        # tinted header with rounded top corners only
        r = 14
        self.add(f'<path d="M{x} {y + hh}V{y + r}a{r} {r} 0 0 1 {r} -{r}H{x + w - r}a{r} {r} 0 0 1 {r} {r}V{y + hh}z" '
                 f'fill="{self.c(acc)}" fill-opacity="{self.t["tint"]}"/>')
        self.add(f'<path d="M{x} {y + hh}H{x + w}" stroke="{self.c("stroke")}" stroke-width="1"/>')
        if icon:
            self.add(f'<circle cx="{x + 27}" cy="{y + hh / 2}" r="15" fill="{self.c(acc)}"/>')
            self.icon(icon, x + 27, y + hh / 2, 18)
        tx = x + (52 if icon else 16)
        self.text(tx, y + hh / 2 + 5, title, 14.5, "text", 650)
        if num is not None:
            self.text(x + w - 14, y + hh / 2 + 4.5, num, 12, acc, 700, "end", mono=True)
        yy = y + hh + 22 if top is None else y + top
        for ln in lines:
            if isinstance(ln, tuple):
                self.text(x + 16, yy, ln[0], line_size, ln[1], ln[2] if len(ln) > 2 else 400)
            else:
                self.text(x + 16, yy, ln, line_size, "muted")
            yy += line_size + 6
        if foot:
            self.text(x + 16, y + h - 14, foot, 11, acc, 500, mono=True)
        return yy

    def render(self):
        defs = "".join(d for _, d in self.defs)
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'role="img" aria-labelledby="t d">'
                f'<title id="t">{escape(self.title)}</title><desc id="d">{escape(self.desc)}</desc>')
        return head + (f"<defs>{defs}</defs>" if defs else "") + "\n" + "\n".join(self.parts) + "\n</svg>\n"


# ===================================================================================================================
def hero(theme):
    W, H = 1280, 380
    s = Svg(W, H, theme, "openballbot-rl", "An end-to-end ballbot pipeline from robot spec to ROS 2, simulation only.")
    t = s.t
    s.defs.append(("bg", f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t["hero0"]}"/>'
                         f'<stop offset="1" stop-color="{t["hero1"]}"/></linearGradient>'))
    s.defs.append(("grid", f'<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" '
                           f'stroke="{t["grid"]}" stroke-opacity="{t["gridop"]}" stroke-width="1"/></pattern>'))
    s.defs.append(("ballg", '<radialGradient id="ballg" cx="0.35" cy="0.3" r="0.8"><stop offset="0" stop-color="#f08a4b"/>'
                            '<stop offset="0.55" stop-color="#c2410c"/><stop offset="1" stop-color="#6b230a"/></radialGradient>'))
    s.defs.append(("bodyg", '<linearGradient id="bodyg" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#14532d"/>'
                            '<stop offset="0.45" stop-color="#22a35a"/><stop offset="1" stop-color="#0f3d22"/></linearGradient>'))
    s.defs.append(("wheelg", '<linearGradient id="wheelg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#64748b"/>'
                             '<stop offset="1" stop-color="#1e293b"/></linearGradient>'))
    s.defs.append(("cone", f'<linearGradient id="cone" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t["teal"]}" stop-opacity="0.45"/>'
                           f'<stop offset="1" stop-color="{t["teal"]}" stop-opacity="0.04"/></linearGradient>'))
    s.defs.append(("fade", '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#fff" stop-opacity="0"/>'
                           '<stop offset="0.45" stop-color="#fff" stop-opacity="1"/></linearGradient>'))
    s.defs.append(("gm", '<mask id="gm"><rect width="1280" height="380" fill="url(#fade)"/></mask>'))
    s.defs.append(("clip", '<clipPath id="clip"><rect x="0" y="0" width="1280" height="380" rx="22"/></clipPath>'))
    s.add('<g clip-path="url(#clip)">')
    s.add(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    s.add(f'<rect width="{W}" height="{H}" fill="url(#grid)" mask="url(#gm)"/>')
    # floor in perspective
    gx, gy = 1010, 318
    for i in range(-7, 8):
        s.add(f'<path d="M{gx + i * 34} {gy} L{gx + i * 95} {H}" stroke="{t["grid"]}" stroke-opacity="{t["gridop"] * 1.8}" stroke-width="1"/>')
    for k, yy in enumerate([gy, gy + 12, gy + 28, gy + 50]):
        s.add(f'<path d="M{gx - 300 - k * 60} {yy}H{gx + 300 + k * 60}" stroke="{t["grid"]}" stroke-opacity="{t["gridop"] * 1.8}" stroke-width="1"/>')
    s.add("</g>")
    s.add(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="22" fill="none" stroke="{t["stroke"]}"/>')

    # text block
    x0 = 64
    px_ = x0
    for lab, col, solid in [("SIMULATION ONLY", "orange", True), ("MuJoCo-Warp GPU", "blue", False), ("ROS 2", "purple", False)]:
        px_ += s.pill(px_, 58, lab, col, 12, solid=solid, h=24) + 8
    s.text(x0 - 3, 150, "openballbot-rl", 64, "text", 750, ls="-1.5")
    s.text(x0, 196, "An end-to-end ballbot pipeline: from a robot spec to a", 22, "muted", 400)
    s.text(x0, 226, "GPU-trained balance controller, navigation and ROS 2.", 22, "muted", 400)
    chips = [("LQR + PPO residual", "blue", "net"), ("11 robot designs", "purple", "cube"),
             ("ToF navigation", "teal", "radar"), ("ROS 2 bundle", "green", "nodes")]
    cx = x0
    for lab, col, ic in chips:
        w = len(lab) * 8.1 + 50
        s.rect(cx, 262, w, 36, rx=18, fill="card", stroke="stroke")
        s.add(f'<circle cx="{cx + 19}" cy="280" r="12" fill="{t[col]}"/>')
        s.icon(ic, cx + 19, 280, 14, sw=2.4)
        s.text(cx + 38, 285, lab, 14.5, "text", 600)
        cx += w + 12

    # ballbot illustration
    bx, by, br = 1010, 268, 46
    # sensor field of view: a cone from the head to a zoned footprint on the floor
    sx0, sy0 = bx + 36, 74
    P = [(1078, 338), (1150, 368), (1262, 322), (1198, 298)]      # near-left, near-right, far-right, far-left

    def bil(u, v):
        x = (1 - u) * (1 - v) * P[0][0] + u * (1 - v) * P[1][0] + u * v * P[2][0] + (1 - u) * v * P[3][0]
        y = (1 - u) * (1 - v) * P[0][1] + u * (1 - v) * P[1][1] + u * v * P[2][1] + (1 - u) * v * P[3][1]
        return x, y
    s.add(f'<path d="M{sx0} {sy0}L{P[0][0]} {P[0][1]}L{P[1][0]} {P[1][1]}L{P[2][0]} {P[2][1]}L{P[3][0]} {P[3][1]}Z" fill="url(#cone)"/>')
    for (ex_, ey_) in P:
        s.add(f'<path d="M{sx0} {sy0}L{ex_} {ey_}" stroke="{t["teal"]}" stroke-opacity="0.5" stroke-width="1.1" stroke-dasharray="3 4"/>')
    s.add(f'<path d="M{P[0][0]} {P[0][1]}L{P[1][0]} {P[1][1]}L{P[2][0]} {P[2][1]}L{P[3][0]} {P[3][1]}Z" fill="{t["teal"]}" '
          f'fill-opacity="0.16" stroke="{t["teal"]}" stroke-opacity="0.8" stroke-width="1.4"/>')
    for k in range(1, 6):
        (x1, y1), (x2, y2) = bil(0, k / 6), bil(1, k / 6)
        s.add(f'<path d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}" stroke="{t["teal"]}" stroke-opacity="0.45" stroke-width="1"/>')
    for k in range(1, 8):
        (x1, y1), (x2, y2) = bil(k / 8, 0), bil(k / 8, 1)
        s.add(f'<path d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}" stroke="{t["teal"]}" stroke-opacity="0.45" stroke-width="1"/>')
    # planned path and goal
    s.add(f'<path d="M760 352 C 840 340, 900 336, {bx - 10} {by + br + 6}" fill="none" stroke="{t["blue"]}" stroke-width="2.5" '
          f'stroke-dasharray="7 6" stroke-linecap="round"/>')
    s.add(f'<ellipse cx="1200" cy="338" rx="14" ry="4.5" fill="#000" fill-opacity="0.2"/>')
    s.add(f'<circle cx="1200" cy="325" r="10" fill="{t["red"]}"/><circle cx="1196.5" cy="321.5" r="3" fill="#fff" fill-opacity="0.55"/>')
    # shadow
    s.add(f'<ellipse cx="{bx + 6}" cy="{by + br + 4}" rx="62" ry="10" fill="#000" fill-opacity="0.22"/>')
    # rod and tower
    s.add(f'<rect x="{bx - 4}" y="150" width="8" height="70" rx="3" fill="#94a3b8"/>')
    s.add(f'<rect x="{bx - 38}" y="76" width="76" height="98" rx="14" fill="url(#bodyg)"/>')
    s.add(f'<ellipse cx="{bx}" cy="80" rx="38" ry="10" fill="#2fbf6c"/>')
    s.add(f'<rect x="{bx + 14}" y="64" width="30" height="20" rx="5" fill="#334155"/>'
          f'<circle cx="{bx + 36}" cy="74" r="5" fill="{t["teal"]}"/>')
    # ball
    s.add(f'<circle cx="{bx}" cy="{by}" r="{br}" fill="url(#ballg)"/>')
    s.add(f'<path d="M{bx - br + 3} {by + 4} C {bx - 22} {by + 24}, {bx + 22} {by + 24}, {bx + br - 3} {by + 4}" fill="none" '
          f'stroke="#3b1406" stroke-opacity="0.45" stroke-width="2"/>')
    # base plate with three motors; omniwheels touch the ball at about 45 degrees from the top
    s.add(f'<rect x="{bx - 46}" y="{by - 66}" width="92" height="12" rx="6" fill="#475569"/>')
    import math
    for sx in (-1, 0, 1):
        ang = math.radians(44 * sx) if sx else 0.0
        cxw = bx + (br + 3) * math.sin(math.radians(46)) * sx
        cyw = by - (br + 3) * math.cos(math.radians(46)) if sx else by - br - 6
        if sx == 0:
            s.add(f'<rect x="{bx - 13}" y="{cyw - 5}" width="26" height="10" rx="4" fill="#334155"/>')
            continue
        s.add(f'<path d="M{cxw:.1f} {cyw:.1f}L{bx + sx * 34} {by - 60}" stroke="#334155" stroke-width="6" stroke-linecap="round"/>')
        s.add(f'<g transform="rotate({-sx * 44} {cxw:.1f} {cyw:.1f})"><rect x="{cxw - 15:.1f}" y="{cyw - 6:.1f}" width="30" height="12" '
              f'rx="5" fill="#1e293b"/>' + "".join(f'<rect x="{cxw - 13 + 6.5 * k:.1f}" y="{cyw - 4.5:.1f}" width="4.5" height="9" '
                                                    f'rx="2" fill="#94a3b8"/>' for k in range(4)) + '</g>')
    # caption under the robot
    s.text(bx, 358, "simulated three-omniwheel ballbot", 12.5, "muted", 500, "middle", italic=True)
    return s


# ===================================================================================================================
def pipeline(theme):
    W, H = 1000, 556
    s = Svg(W, H, theme, "Pipeline overview",
            "Robot spec, MuJoCo model and design variants, GPU training on MuJoCo-Warp, evaluation, navigation, "
            "export and ROS 2 deployment.")
    cw, gap = 216, 32
    xs = [20 + i * (cw + gap) for i in range(4)]
    y1, h1 = 30, 212
    y2, h2 = 312, 196
    # row 1
    s.card(xs[0], y1, cw, h1, "blue", "Robot spec", [
        "Physical spec in YAML:", "ball radius and mass, body", "mass, COM height, wheel", "torque, rotor inertia,",
        "contact angle", ("Physics + feasibility margins", "text", 500)], foot="robot_spec.py", icon="doc", num="01")
    s.card(xs[1], y1, cw, h1, "purple", "Model + designs", [
        "MuJoCo three-omniwheel", "ballbot model", "Parametric design variants:", "MuJoCo XML, planar model,",
        "LQR gains per design", ("11 training designs", "text", 500)], foot="design_space.py", icon="cube", num="02")
    tx, tw = xs[2], W - 20 - xs[2]
    s.card(tx, y1, tw, h1, "green", "GPU training on MuJoCo-Warp", [
        ("LQR base + PPO residual, design-conditioned", "text", 500),
        "4096 parallel worlds, float32, 100 Hz control, 200M steps (g39)"], foot="train_dev.py  ·  warp_sim.py  ·  lqr_gpu.py",
        icon="chip", num="03")
    chips = [["Domain randomisation", "11 designs", "8 x 8 terrain tiles"],
             ["Pushes + sustained pulls", "Sensor noise + latency", "Motion layer"]]
    for r, row in enumerate(chips):
        cx = tx + 16
        for lab in row:
            w = s.pill(cx, y1 + 122 + r * 30, lab, "green", 11.5, h=22)
            cx += w + 8
    # row 2: ROS 2 <- export <- (training) ; navigation -> evaluation
    s.card(xs[0], y2, cw, h2, "pink", "ROS 2", [
        "ballbot_control package:", "pure-Python core,", "controller + estimator", "nodes at 100 Hz,",
        "pytest parity vs simulator", ("not yet run under ROS", "orange", 600)], foot="ros2/ballbot_control", icon="nodes", num="07")
    s.card(xs[1], y2, cw, h2, "orange", "Export", [
        "ONNX, TorchScript and", "numpy actor weights", "bundle.json: gains, obs", "layout, limits, estimator",
        "Checked vs torch actor"], foot="export_policy.py", icon="package", num="06")
    s.card(xs[2], y2, cw, h2, "teal", "Navigation", [
        "Simulated 18 x 14-zone ToF", "Wavefront planner +", "smooth follower", "Blind contact escape",
        "IMU + encoder odometry"], foot="nav.py  ·  odometry.py", icon="radar", num="05")
    s.card(xs[3], y2, cw, h2, "yellow", "Evaluation", [
        "Goal + velocity battery", "on terrain tiles at 3 DR", "scales, pushes on and off", "Goal hold, curve, zigzag",
        "Rendered clips"], foot="eval_battery.py", icon="chart", num="04")
    # arrows
    ym = y1 + h1 / 2
    s.arrow(f"M{xs[0] + cw + 3} {ym}H{xs[1] - 5}")
    s.arrow(f"M{xs[1] + cw + 3} {ym}H{xs[2] - 5}")
    bus = y1 + h1 + 35
    cxs = [x + cw / 2 for x in xs]
    s.arrow(f"M{tx + tw / 2} {y1 + h1 + 2}V{bus}", head=False)
    s.arrow(f"M{cxs[1]} {bus}H{cxs[3]}", head=False)
    for c in cxs[1:]:
        s.arrow(f"M{c} {bus}V{y2 - 5}")
    s.add(f'<circle cx="{tx + tw / 2}" cy="{bus}" r="3.5" fill="{s.c("arrow")}"/>')
    s.text(cxs[1] + 8, bus - 7, "checkpoint", 11, "faint", 500, italic=True)
    ym2 = y2 + h2 / 2
    s.arrow(f"M{xs[1] - 3} {ym2}H{xs[0] + cw + 5}")
    s.arrow(f"M{xs[2] + cw + 3} {ym2}H{xs[3] - 5}")
    s.text(W / 2, H - 16, "Simulation only. Every number is in docs/research/tracking_study_log.md (S0 to S122).",
           12, "faint", 500, "middle", italic=True)
    return s


# ===================================================================================================================
def controller(theme):
    W, H = 1000, 658
    s = Svg(W, H, theme, "Controller architecture",
            "Planner velocity command, motion-reference layer, LQR base plus PPO residual with adaptive authority, "
            "safety and torque limits, wheels, and the state estimator feeding back.")
    # enclosure for the 100 Hz loop
    ex, ey, ew, eh = 204, 26, 776, 432
    s.rect(ex, ey, ew, eh, rx=18, fill="panel", stroke="stroke", dash="6 5", sw=1.3)
    s.text(ex + 18, ey + 24, "BALANCE CONTROLLER  ·  100 Hz", 12, "muted", 700, ls="0.8")
    s.text(ex + ew - 18, ey + 24, "GpuBallbot.step in training  ·  controller_node in ROS 2", 11.5, "faint", 500, "end")

    # planner
    s.card(20, 150, 164, 160, "blue", "Planner", [("/cmd_vel", "blue", 600), "Nav2 in ROS 2", "(10 to 20 Hz) or the",
                                                  "study's wavefront", "planner"], icon="route")
    # motion layer
    mx, my, mw, mh = 228, 150, 206, 214
    s.card(mx, my, mw, mh, "teal", "Motion layer", [
        "Shapes the command a", "ballbot can follow:", "accel 0.4, brake 0.5 m/s²", "lateral accel 0.35 m/s²",
        "arrival braking curve", "goal hold on ball centre"], foot="motion_ref.py", icon="wave")
    # column B
    bx, bw = 476, 216
    s.card(bx, 52, bw, 120, "purple", "LQR base", ["Per-design discrete LQR", "on the planar model"],
           foot="lqr_gpu.py", icon="matrix")
    s.card(bx, 182, bw, 140, "green", "PPO residual actor", [
        "obs 25 + history 8 x 28", "+ design vector 6 = 255", "MLP 3 x 512, 3 outputs"], foot="train_dev.py", icon="net")
    s.card(bx, 332, bw, 112, "orange", "Adaptive authority", [
        "rs 0.15 cruising, to 0.3 on", "fast command change, stop", "(ball > 0.05 m/s) or tilt"], icon="gauge", line_size=12)
    # mixer
    sx, sy = 742, 236
    s.add(f'<circle cx="{sx}" cy="{sy}" r="24" fill="{s.c("card")}" stroke="{s.c("text")}" stroke-width="1.8"/>')
    s.add(f'<path d="M{sx - 9} {sy}h18 M{sx} {sy - 9}v18" stroke="{s.c("text")}" stroke-width="2.4" stroke-linecap="round"/>')
    s.text(sx, sy + 44, "applied =", 11.5, "muted", 500, "middle")
    s.text(sx, sy + 60, "clip(u + rs·a)", 12, "text", 600, "middle", mono=True)
    # safety
    fx, fy, fw, fh = 800, 52, 164, 250
    s.card(fx, fy, fw, fh, "red", "Safety", [
        "HOLD / ACTIVE /", "FAULT modes", "tilt > 30 deg: FAULT", "stale state: FAULT", "torque clamp to", "design limit",
        "(30 N m, nom)"], foot="safety.py", icon="shield", line_size=12)
    # arrows inside
    s.arrow(f"M187 225H{mx - 5}")
    s.text(206, 216, "v, goal", 11, "faint", 500, "middle", italic=True)
    s.arrow(f"M{mx + mw + 3} 236H{bx - 5}")
    s.text(455, 228, "cmd", 11, "faint", 500, "middle", italic=True)
    s.arrow(f"M455 236V112H{bx - 5}")
    s.arrow(f"M455 236V388H{bx - 5}")
    s.arrow(f"M{bx + bw + 3} 112H{sx}V{sy - 29}")
    s.text(bx + bw + 14, 106, "u", 13, "purple", 700, italic=True)
    s.arrow(f"M{bx + bw + 3} 243H{sx - 29}")
    s.text(bx + bw + 14, 237, "a", 13, "green", 700, italic=True)
    s.arrow(f"M{bx + bw + 3} 388H{sx + 60}V{sy + 70}", head=False)
    s.arrow(f"M{sx + 60} {sy + 70}V{sy + 40}L{sx + 19} {sy + 15}")
    s.text(bx + bw + 14, 382, "rs", 13, "orange", 700, italic=True)
    s.arrow(f"M{sx + 27} {sy}H{fx - 5}")

    # plant row
    py, ph = 496, 120
    s.card(800, py, 164, ph, "red", "Wheels", ["3 omniwheels,", "torque in N m"], foot="/wheel_torque_cmd", icon="wheel")
    s.card(560, py, 196, ph, "yellow", "Ballbot", ["simulated: MuJoCo-Warp", ("no physical robot yet", "orange", 600)],
           icon="ball")
    s.card(316, py, 200, ph, "blue", "IMU + encoders", ["gyro, accelerometer,", "wheel encoder rates"],
           foot="/imu  ·  /joint_states", icon="imu")
    s.card(20, py, 252, ph, "purple", "State estimator", ["attitude filter, gyro bias,", "calibrated encoders, Kalman"],
           foot="odometry.py  ·  estimator.py", icon="estim")
    s.arrow(f"M{fx + fw / 2} {fy + fh + 3}V{py - 5}")
    s.text(fx + fw / 2 + 8, 410, "torques", 11, "faint", 500, italic=True)
    s.arrow(f"M797 {py + ph / 2}H{756 + 5}")
    s.arrow(f"M557 {py + ph / 2}H{516 + 5}")
    s.arrow(f"M313 {py + ph / 2}H{272 + 5}")
    # feedback
    fb = 476
    s.arrow(f"M146 {py - 3}V{fb}H{bx + 40}", color="purple", dash="5 5", head=False)
    s.arrow(f"M{mx + mw / 2} {fb}V{my + mh + 5}", color="purple", dash="5 5")
    s.arrow(f"M{bx + 40} {fb}V{444 + 5}", color="purple", dash="5 5")
    s.arrow(f"M146 {fb}V{310 + 5}", color="purple", dash="5 5")
    s.text(176, fb - 6, "state estimate", 11.5, "purple", 600, italic=True)
    s.text(112, 392, "/odom", 11.5, "purple", 600, "middle", italic=True)
    s.text(W / 2, H - 14, "State feeds the motion layer, the observation, the LQR and the authority rule; /odom feeds the "
                          "planner. Simulation only.", 12, "faint", 500, "middle", italic=True)
    return s


# ===================================================================================================================
def results(theme):
    W, H = 1000, 604
    s = Svg(W, H, theme, "Results at a glance",
            "Headline simulation results: g39 navigation success, goal hold error before and after the goal-centre fix, "
            "motion layer, ROS 2 parity and design-sweep findings, each with its study-log entry.")

    def panel(x, y, w, h, title, sub, tag):
        s.rect(x, y, w, h, rx=16)
        s.text(x + 20, y + 32, title, 16, "text", 700)
        s.text(x + 20, y + 52, sub, 12, "muted", 400)
        s.pill(x + w - 16, y + 18, tag, "orange", 11, anchor="end")

    # --- navigation panel
    px, py, pw, ph = 20, 20, 470, 304
    panel(px, py, pw, ph, "Navigation to a goal 1.2 m away", "g39 · behind 8 cm blocks · 528 worlds · 20 s · seed 0",
          "simulation · S121")
    rows = [("Perfect height map", 0.953, 0.001, 0.045, "3 seeds: 0.94 to 0.95"),
            ("Simulated ToF 18 x 14", 0.866, 0.077, 0.057, "3 seeds: 0.87 to 0.89"),
            ("ToF + pushes and pulls", 0.544, None, 0.413, "3 seeds: 0.49 to 0.55")]
    bx0, bw = px + 20, pw - 40
    yy = py + 84
    for lab, suc, stk, fal, seeds in rows:
        s.text(bx0, yy, lab, 13, "text", 600)
        s.text(bx0 + bw, yy, seeds, 11.5, "muted", 400, "end")
        by = yy + 9
        s.rect(bx0, by, bw, 18, rx=5, fill="track", stroke=None)
        w1 = bw * suc
        s.rect(bx0, by, w1, 18, rx=5, fill="green", stroke=None)
        s.text(bx0 + 8, by + 13.5, f"success {suc:.3f}", 11.5, "onacc", 700)
        x = bx0 + w1
        if stk:
            ws = max(bw * stk, 2)
            s.rect(x + 1, by, ws, 18, rx=2, fill="yellow", stroke=None)
            x += ws + 1
        wf = bw * fal
        s.rect(bx0 + bw - wf, by, wf, 18, rx=5, fill="red", stroke=None)
        sub = (f"stuck {stk:.3f}  ·  " if stk else "") + f"fall {fal:.3f}"
        s.text(bx0 + bw, by + 34, sub, 11.5, "muted", 600, "end")
        yy += 64
    # legend
    lx, ly = bx0, py + ph - 22
    for lab, col in [("success", "green"), ("stuck", "yellow"), ("fall", "red"), ("not reported", "track")]:
        s.rect(lx, ly - 9, 10, 10, rx=2, fill=col, stroke=None)
        s.text(lx + 15, ly, lab, 11.5, "muted", 500)
        lx += 15 + len(lab) * 6.6 + 18

    # --- goal hold panel
    qx, qy, qw, qh = 510, 20, 470, 304
    panel(qx, qy, qw, qh, "Goal hold error on flat ground", "RMS · 528 worlds · motion layer · range over three seeds",
          "simulation · S118")
    ax0, axw = qx + 30, qw - 60
    vmax = 15.0

    def X(v):
        return ax0 + axw * v / vmax

    ybase = qy + 236
    for v in range(0, 16, 5):
        s.add(f'<path d="M{X(v):.1f} {qy + 78}V{ybase}" stroke="{s.c("stroke")}" stroke-width="1" stroke-dasharray="2 4"/>')
        s.text(X(v), ybase + 18, f"{v} cm", 11, "faint", 500, "middle")
    bars = [("g34: goals on the ball body origin", 11.4, 13.6, "red", "11.4 to 13.6 cm"),
            ("g39: goals on the true ball centre (--goal-centre)", 3.6, 4.9, "green", "3.6 to 4.9 cm")]
    yy = qy + 100
    for lab, lo, hi, col, val in bars:
        s.text(ax0, yy, lab, 13, "text", 600)
        s.rect(X(lo), yy + 12, X(hi) - X(lo), 22, rx=11, fill=col, stroke=None)
        s.text(X(hi) + 10 if hi < 10 else X(lo) - 10, yy + 28, val, 13, col, 700, "start" if hi < 10 else "end")
        yy += 66
    s.text(qx + 20, qy + qh - 14, "Fix (S118): goals at the ball centre  ·  g39 seed 0: 3.6 cm, falls 0.070", 11.5, "muted", 500)

    # --- stat tiles
    ty, th = 344, 196
    tw, tg = 230, 16
    tiles = [
        ("purple", "wave", "simulation · S114", [("1.79", "text"), (" → ", "faint"), ("1.31", "purple")], "rad/s",
         ["heading-rate RMS with the", "motion layer (g34, ToF nav):", "a quarter less wobble"]),
        ("blue", "nodes", "simulation · S118", [("1.3e-3", "blue")], "N m",
         ["max wheel-torque difference,", "ROS 2 core vs simulator,", "g39, 13299 control ticks"]),
        ("teal", "ball", "simulation · S119", [("≥ 4", "teal")], "% of body mass",
         ["ball mass; 8 to 15 percent", "is better (design sweep)", ""]),
        ("orange", "gauge", "simulation · S119", [("≥ 15", "orange")], "N m per wheel",
         ["peak motor torque needed", "for a 4 kg robot", "(design sweep)"]),
    ]
    for i, (acc, ic, tag, big, unit, lines) in enumerate(tiles):
        x = 20 + i * (tw + tg) + (2 if i == 3 else 0)
        s.rect(x, ty, tw, th, rx=16)
        s.add(f'<rect x="{x}" y="{ty}" width="5" height="{th}" rx="2.5" fill="{s.c(acc)}"/>')
        s.add(f'<circle cx="{x + 32}" cy="{ty + 32}" r="15" fill="{s.c(acc)}"/>')
        s.icon(ic, x + 32, ty + 32, 18)
        s.pill(x + tw - 14, ty + 22, tag, "orange", 10.5, anchor="end", h=19)
        runs = [(t_, c_, 750, 30) for t_, c_ in big] + [("  " + unit, "muted", 600, 13)]
        s.rich(x + 20, ty + 98, runs)
        yy = ty + 130
        for ln in lines:
            s.text(x + 20, yy, ln, 12.5, "muted")
            yy += 19
    s.text(W / 2, H - 40, "MuJoCo-Warp float32 on the GPU, domain randomisation and sensor noise on unless noted.",
           12, "faint", 500, "middle", italic=True)
    s.text(W / 2, H - 20, "All numbers are simulation results from Readme.md and docs/research/tracking_study_log.md; "
                          "no hardware results exist.", 12, "faint", 500, "middle", italic=True)
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("hero", hero), ("pipeline", pipeline), ("controller", controller), ("results", results)]:
        for theme in ("light", "dark"):
            p = os.path.join(OUT, f"{name}-{theme}.svg")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(fn(theme).render())
            print(f"{p}  {os.path.getsize(p) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
