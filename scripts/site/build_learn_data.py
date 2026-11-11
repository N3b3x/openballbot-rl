"""Build site/learn/data/learn.json, the data behind the interactive course in site/learn/.

Every number the course widgets compute from or plot comes from this file, and every value here is read from a repo file:
  ballbot_gym/models/ballbot_planar_model.json        nominal planar model (scripts/study/ballbot_model.py --json)
  ballbot_gym/models/variants/dz_<design>.json         planar model of each of the 11 training designs (design_space.py)
  ros2/ballbot_control/config/bundle/<run>/bundle.json deployed LQR weights, per-design discrete gains, motion layer, authority
  outputs/study/specs/ota_sim.csv, ota_sweep.csv,      design sweep S119 (simulation falls and analytic physics per variant)
    spec_nominal_sim.csv
  outputs/study/warp_ppo/odom/odom_results_flat.json   odometry study S108
  outputs/study/warp_ppo/g34_mref_rsad_db_s0/motion_goal_ref_allflat{,_lqronly}.json   flat goal test, S114 addendum
Numbers quoted from the study log (docs/research/tracking_study_log.md) are listed in LOG_FACTS with the exact text they are
read from; the script checks that each text still occurs in the log entry it cites and fails loudly if not.

  python scripts/site/build_learn_data.py            # writes site/learn/data/learn.json
  python scripts/site/build_learn_data.py --check    # only verify the log citations
  python scripts/site/build_learn_data.py --links    # link and dash check over site/learn/
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "learn" / "data" / "learn.json"
LOG = ROOT / "docs" / "research" / "tracking_study_log.md"
BUNDLE = ROOT / "ros2" / "ballbot_control" / "config" / "bundle" / "g39_goalcentre_s0" / "bundle.json"
G = 9.81

DESIGNS = [  # name, short label, what differs (design_space.py generator, docs/research/design_guidelines.md, S114 naming note)
    ("nom", "Nominal", "the reference robot"),
    ("zhigh", "High COM", "ballast moved up, body centre of mass higher"),
    ("zlow", "Low COM", "ballast moved down, body centre of mass lower"),
    ("kb4", "Heavy ball", "4 x ball mass"),
    ("kb025", "Light ball", "0.25 x ball mass"),
    ("bal4", "Heavy body", "4 x ballast"),
    ("bal025", "Light body", "0.25 x ballast"),
    ("J02", "Big rotor", "wheel rotor inertia 0.02 kg m^2 (4 x)"),
    ("tau6", "Weak motors", "6 N m wheel torque limit instead of 30"),
    ("zeta41", "Contact 41 deg", "wheels touch the ball 10 deg higher"),
    ("zeta66", "Contact 66 deg", "wheels touch the ball 15 deg lower"),
]

# (S-number, exact text in the log entry, value(s) the course uses, what it means)
LOG_FACTS = [
    ("S114", "ToF: success / fall / stuck", None, "navigation table header"),
    ("S114", "1.79 without, 1.31 with", {"no_layer": 1.79, "layer": 1.31}, "heading-rate RMS of the path, rad/s, g34 s0, ToF"),
    ("S114", "no layer 8.9 s, default layer 9.2 s", {"no_layer": 8.9, "layer": 9.2}, "median time to goal"),
    ("S114", "full controller 0.068 (0.104 on the tiled map), LQR alone 0.324", {"full": 0.068, "lqr": 0.324}, "flat goal test falls"),
    ("S104", "| ToF 18 x 14 | 0.84 / 0.07 / 0.08 / 0.04 | 10.8 s", {"reached": 0.84, "fell": 0.07, "stuck": 0.08}, "ToF navigation, g26"),
    ("S104", "| perfect map | 0.89 / 0.07 / 0.04 / 0.01 | 9.5 s", {"reached": 0.89, "fell": 0.07, "stuck": 0.04}, "perfect-map navigation, g26"),
    ("S104", "reached 0.09, fell 0.91", {"reached": 0.09, "fell": 0.91}, "blind straight at the goal"),
    ("S109", "| blind + escape, 0.3 m/s | 20 s | 0.54 / 0.35 / 0.11 / 0.50", {"reached": 0.54, "fell": 0.35, "stuck": 0.11}, "blind with contact escape"),
    ("S115", "success 0.634, fall 0.261, stuck 0.104", {"reached": 0.634, "fell": 0.261, "stuck": 0.104}, "escape on g34"),
    ("S121", "| g39 s0 (goal centre) | 0.866 / 0.077 / 0.057", {"reached": 0.866, "stuck": 0.077, "fell": 0.057}, "ToF navigation, recommended g39"),
    ("S121", "0.953 / 0.001 / 0.045", {"reached": 0.953, "stuck": 0.001, "fell": 0.045}, "perfect map, g39"),
    ("S118", "the origin is 0.14 m from the ball centre", {"offset_m": 0.14}, "ball body origin offset"),
    ("S118", "falls 0.070, hold error 3.6 cm", {"fall": 0.070, "hold_cm": 3.6}, "g39 goal hold"),
    ("S118", "hold error 12.6 / 11.4 / 13.6 cm RMS", {"hold_cm": [12.6, 11.4, 13.6]}, "g34 seeds, true centre"),
    ("S113", "On a fully flat map (TILES_FLAT=1) the same zigzag falls 0.051", {"flat": 0.051}, "zigzag exposure artefact"),
    ("S113", "the zigzag's net forward travel (about 2 m)", None, "zigzag exposure artefact"),
    ("S113", "g32 s0: 4.94 command reversals per world", {"reversals": 4.94}, "the one hunting seed"),
    ("S103", "the halving was an exposure artefact", None, "speed layer exposure artefact"),
    ("S103", "none / perfect map / ToF: 0.38 / 0.33 / 0.35 overall", {"none": 0.38, "oracle": 0.33, "tof": 0.35}, "40 s re-test"),
    ("S103", "falls 0.38 to 0.39 without a speed layer, 0.15 to 0.17 with the ToF-driven layer", {"none": 0.385, "tof": 0.16}, "15 s result, withdrawn"),
    ("S116", "Falls: pulls 0.75 to 3 N 0.078 (as without pulls), 1.5 to 6 N 0.139, 2.25 to 9 N 0.221, 3 to 12 N 0.361",
     {"ranges": [[0.75, 3, 0.078], [1.5, 6, 0.139], [2.25, 9, 0.221], [3, 12, 0.361]]}, "pull-strength sweep"),
    ("S117", "pull capacity about m_a g sin 5 deg = 3.3 N", {"F": 3.3}, "static pull estimate"),
    ("S108", "| realistic: 4096-count wheel encoders, MEMS gyro", None, "odometry sensor model"),
    ("S108", "12 degrees of yaw error after a minute", {"yaw_deg": 12}, "heading drift"),
    ("S108", "about 0.65 times W", {"factor": 0.65}, "uncalibrated wheel matrix"),
    ("S108", "0.06 m after 10 s (1.7 m driven, 3.6 percent", None, "terrain addendum"),
    ("S110", "| g26 s1, LQR alone (residual zeroed), raw commands | 0.53", {"fall": 0.53}, "LQR alone on goals"),
    ("S112", "authority 0.15 while cruising, rising at once to 0.3", None, "adaptive authority"),
    ("S112", "final training fall rate 0.0016 for both, the lowest of the study", None, "adaptive authority"),
    ("S111", "Residual authority is a dial between tracking and recovery", None, "authority dial"),
    ("S85-R", "| g16_pure_ret100 s1 (pure learning) | yes | no | 0.09 / 0.00 | 0.15 / 0.00 |", {"s15": 0.09}, "pure learning, unseen 1.5x robot"),
    ("S85-R", "| g18_ret_rs025 s1 (LQR + residual 0.25) | yes | no | 0.71 / 0.30 | 0.66 / 0.15 |", {"s15": 0.71}, "LQR + residual, unseen 1.5x robot"),
    ("S83", "| 0.15 | g18_ret_rs015 s0 | 1.01-1.02 |", None, "authority curve"),
    ("S119", "per-wheel |torque| median 0.73 N m, p90 2.6, p99 8.1", {"median": 0.73, "p90": 2.6, "p99": 8.1}, "torque used"),
    ("S119", "12 cm ball 0.69 / 0.022 falls, 7 cm ball 0.11 / 0.28", {"r12": [0.69, 0.022], "r7": [0.11, 0.28]}, "LQR-only check"),
    ("S119", "rotor inertia 0.001 0.45 / 0.49, motor torque 3 0.48 / 0.46", None, "LQR-only check, other weak variants"),
    ("S116", "essentially all of the about 0.5 pushed-battery gap is the sustained 3 to 12 N pulls", None, "pulls cause the pushed falls"),
    ("S118", "max torque difference 1.5e-3 N m", {"max_dtau": 1.5e-3}, "ROS 2 parity"),
    ("S118", "14,155 control ticks, 33 worlds, all 11 designs", None, "ROS 2 parity"),
    ("S118", "matches the torch actor to below 5e-7", None, "ROS 2 export"),
    ("S106", "on the goal test falls double (0.15 to 0.31", {"raw": 0.15, "ref": 0.31}, "bolted-on layer"),
]

# The authority curve (S83 table, S82 pure-learning row): flat achieved / command and disturbed goal success at DR scale 1.
AUTHORITY_CURVE = {
    "src": "S82, S83",
    "rows": [
        {"label": "0.15", "rs": 0.15, "tracking": [1.01, 1.02], "robust": [0.35, 0.35]},
        {"label": "0.25", "rs": 0.25, "tracking": [0.99, 1.01], "robust": [0.42, 0.43]},
        {"label": "0.35", "rs": 0.35, "tracking": [0.92, 0.97], "robust": [0.45, 0.45]},
        {"label": "0.5", "rs": 0.5, "tracking": [0.80, 0.90], "robust": [0.46, 0.49]},
        {"label": "pure learning", "rs": 1.0, "tracking": [0.66, 0.83], "robust": [0.50, 0.50]},
    ],
    "needles": [("S83", "| 0.35 | g18_ret_rs035 s0 | 0.92-0.97 |"), ("S83", "| 0.5 | g16_ret100 s0, s1 | 0.80-0.90 |"),
                ("S82", "| pure learning (g16_pure_ret100, s0 and s1) | 0.66-0.83 |")],
}

# Goal hunting by seed (S113 tables and addendum): command reversals per world in the flat goal test.
SEEDS = {
    "src": "S112, S113",
    "rows": [["g32 s0", 4.94], ["g32 s1", 0.57], ["g33 s0", 0.99], ["g33 s1", 1.38], ["g34 s0", 0.67], ["g34 s1", 0.86], ["g34 s2", 0.48]],
    "needles": [("S113", "| g32 s0 (S112) | 0.083 / 4.94"), ("S113", "| g34 s1 (dead band) | 0.091 / 0.86"),
                ("S113", "goal test falls 0.081, 1.38 reversals"), ("S113", "goal test falls 0.110, 0.48 reversals")],
}


# Navigation with and without the motion layer (S114 addenda): ToF source, g34_mref_rsad_db_s0, 528 worlds.
MOTION_NAV = {
    "src": "S114 addenda (splitting the gain; the layer's speed cost and its limits)",
    "rows": [
        dict(short="no layer", long="planner command raw (NO_MREF=1)", success=0.881, fall=0.070, stuck=0.049, rms=1.79, time=8.9,
             needle=("S114", "ToF success 0.881, fall 0.070, stuck 0.049, heading-rate RMS 1.79")),
        dict(short="default", long="default layer: accelerate 0.4, brake 0.5, turn 1.5 rad/s", success=0.877, fall=0.080, stuck=0.044, rms=1.31, time=9.2,
             needle=("S114", "0.877 / 0.080 / 0.044 / 1.31 / 9.2 s")),
        dict(short="faster", long="accelerate 0.6, brake 0.8", success=0.890, fall=0.070, stuck=0.040, rms=1.30, time=9.1,
             needle=("S114", "0.890 / 0.070 / 0.040 / 1.30 / 9.1 s")),
        dict(short="faster + lateral", long="accelerate 0.8, brake 1.0, lateral 0.5", success=0.884, fall=0.080, stuck=0.036, rms=1.37, time=9.2,
             needle=("S114", "0.884 / 0.080 / 0.036 / 1.37 / 9.2 s")),
        dict(short="turn 2.0", long="accelerate 0.6, brake 0.8, lateral 0.5, turn rate 2.0 rad/s", success=0.877, fall=0.072, stuck=0.051, rms=1.72, time=9.0,
             needle=("S114", "0.877 / 0.072 / 0.051 / 1.72 / 9.0 s")),
    ],
}


# Seed spread of the battery goal success (DR 1, no pushes) for the motion-layer recipes (S113 tables and addenda).
SPREAD = {
    "src": "S113",
    "groups": [
        dict(name="g32 adaptive authority", values=[0.76, 0.75], needles=[("S113", "| g32 s0 (S112) | 0.083 / 4.94 / 8.0 cm / 0.096 m/s | 0.034 / 0.75 | 0.30 | 0.76 0.50"), ("S113", "| g32 s1 (S112) | 0.104 / 0.57 / 7.1 cm / 0.063 m/s | 0.040 / 0.77 | 0.32 | 0.75 0.52")]),
        dict(name="g33 tracking weight 3", values=[0.77, 0.77], needles=[("S113", "| g33 s0 (tracking 3) | 0.100 / 0.99 / 6.9 cm / 0.068 m/s | 0.040 / 0.80 | 0.31 | 0.77 0.53"), ("S113", "= 0.77 0.51 0.73 0.45 0.67 0.38")]),
        dict(name="g34 dead band", values=[0.79, 0.77, 0.78], needles=[("S113", "| g34 s0 (dead band) | 0.104 / 0.67 / 7.1 cm / 0.058 m/s | 0.047 / 0.78 | 0.33 | 0.79 0.51"), ("S113", "| g34 s1 (dead band) | 0.091 / 0.86 / 6.7 cm / 0.065 m/s | 0.049 / 0.73 | 0.28 | 0.77 0.51"), ("S113", "battery goal success 0.78 0.52 0.75 0.45 0.69 0.39")]),
    ],
}


def log_entries():
    text = LOG.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^## ", text)
    ent = {}
    for p in parts[1:]:
        m = re.match(r"(S\d+(?:-R)?)\b", p)
        if m:
            ent.setdefault(m.group(1), "")
            ent[m.group(1)] += p
    # addenda filed under a later heading ("S108 addendum, ...") count for the entry they name
    for line in text.splitlines():
        m = re.match(r"(S\d+(?:-R)?) (?:addendum|note|zigzag addendum)", line)
        if m:
            ent.setdefault(m.group(1), "")
            ent[m.group(1)] += "\n" + line
    return ent


def check_facts():
    ent = log_entries()
    bad = []
    needles = [(s, t) for s, t, *_ in LOG_FACTS] + AUTHORITY_CURVE["needles"] + SEEDS["needles"] + [r["needle"] for r in MOTION_NAV["rows"]] + [n for g in SPREAD["groups"] for n in g["needles"]]
    for s, t in needles:
        if t not in ent.get(s, ""):
            bad.append(f"{s}: {t!r}")
    if bad:
        print("Log citations not found (update the course text and LOG_FACTS):\n  " + "\n  ".join(bad))
        sys.exit(1)
    print(f"log citations ok ({len(needles)} checked)")


def parity():
    """Parity table of ros2/README.md: max and median abs difference core vs simulator, per quantity, for g39 and g34."""
    txt = (ROOT / "ros2" / "README.md").read_text(encoding="utf-8")
    sec = txt.split("### Parity results", 1)[1].split("Exported actor", 1)[0]
    rows = []
    for line in sec.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.match(r"[0-9.]+e-?\d+ / ", cells[1]):
            g39 = [float(v) for v in cells[1].split("/")]
            g34 = [float(v) for v in cells[2].split("/")]
            rows.append(dict(name=cells[0], g39=g39, g34=g34))
    ticks = re.findall(r"\((\d+) ticks\)", sec)
    ntests = re.search(r"(\d+) tests, about", txt)
    return dict(src="ros2/README.md, Parity results", rows=rows, ticks=ticks, n_tests=int(ntests.group(1)) if ntests else None)


def check_links():
    """Link check over site/learn/*.html: relative files and anchors must exist; GitHub blob links must name a file in this repo;
    no em or en dashes in any text under site/learn/. Prints problems and exits 1 if there are any."""
    from html.parser import HTMLParser
    learn = ROOT / "site" / "learn"
    gh = "https://github.com/N3b3x/openballbot-rl/blob/master/"
    ids_cache, problems, count = {}, [], 0

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links, self.ids = [], set()

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if "id" in a:
                self.ids.add(a["id"])
            for k in ("href", "src"):
                if k in a and a[k]:
                    self.links.append(a[k])

    def parse(path):
        if path not in ids_cache:
            pr = P()
            pr.feed(path.read_text(encoding="utf-8"))
            ids_cache[path] = pr
        return ids_cache[path]

    for f in sorted(learn.glob("*.html")):
        for link in parse(f).links:
            count += 1
            if link.startswith(("data:", "mailto:")):
                continue
            if link.startswith(gh):
                if not (ROOT / link[len(gh):].split("#")[0]).exists():
                    problems.append(f"{f.name}: missing repo file {link}")
                continue
            if link.startswith("http"):
                continue
            target, _, frag = link.partition("#")
            tp = (f.parent / target).resolve() if target else f
            if not tp.exists():
                problems.append(f"{f.name}: missing {link}")
            elif frag and tp.suffix == ".html" and frag not in parse(tp).ids:
                problems.append(f"{f.name}: missing anchor {link}")
    for f in sorted(learn.rglob("*")):
        if f.is_file() and f.suffix in (".html", ".js", ".css", ".md", ".json", ".py"):
            t = f.read_text(encoding="utf-8")
            if "\u2014" in t or "\u2013" in t:
                problems.append(f"{f.relative_to(ROOT)}: contains an em or en dash")
    for f in sorted((learn / "assets" / "js").glob("*.js")):
        for imp in re.findall(r"from '(\./[^']+)'", f.read_text(encoding="utf-8")):
            if not (f.parent / imp).exists():
                problems.append(f"{f.name}: missing import {imp}")
    print(f"checked {count} links in {len(list(learn.glob('*.html')))} pages")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("link check ok")


def rnd(x, n=6):
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (int,)) or x is None:
        return x
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def planar(path):
    M = json.load(open(path))
    P = M["params"]
    A, B = np.array(M["A"]), np.array(M["B"])
    ev = np.linalg.eigvals(A)
    p = float(max(ev.real))
    z = math.sqrt(P["m_a"] * G * P["l"] / (P["I_a"] + P["m_a"] * P["l"] ** 2 + P["m_a"] * P["l"] * P["r"]))
    kappa = ((P["m_k"] + P["m_a"]) * P["r"] ** 2 + P["I_k"] + P["m_a"] * P["l"] * P["r"]) / (P["m_a"] * G * P["l"] * P["r"])
    keep = {k: rnd(P[k]) for k in ("r", "m_k", "I_k", "m_a", "l", "I_a", "J_w", "d_w", "J_eff", "d_eff", "zenith_deg") if k in P}
    return dict(params=keep, A=rnd(A.tolist()), B=rnd(B.ravel().tolist()), poles=rnd(sorted(ev.real.tolist())),
                pole=rnd(p, 4), zero=rnd(z, 4), lean_rad=rnd(kappa, 4), lean_deg=rnd(math.degrees(kappa), 4),
                K_cont=rnd(M["K"][0]) if M.get("K") else None, n=rnd(M.get("n")))


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return rnd(x, 5)
    except (TypeError, ValueError):
        return {"True": True, "False": False}.get(v, v)


def sweep():
    sim = {r["name"]: r for r in read_csv(ROOT / "outputs/study/specs/ota_sim.csv")}
    phy = {r["name"]: r for r in read_csv(ROOT / "outputs/study/specs/ota_sweep.csv")}
    nom = read_csv(ROOT / "outputs/study/specs/spec_nominal_sim.csv")[0]
    cols_sim = ["sim_goal_still_fall", "sim_goal_pushes_fall", "sim_goal_pulls_half_fall", "sim_drive_0.3_fall", "sim_goal_terrain_fall"]
    cols_phy = ["phys_r", "phys_m_k", "phys_m_a", "phys_total_mass", "phys_l", "phys_p", "phys_z", "phys_kappa_deg",
                "phys_pull_capacity_5deg_N", "phys_binding", "phys_feasible", "phys_v_max", "phys_a_max"]
    nominal_vals = {"ball_mass_kg": 0.1679, "body_com_height_m": 0.1306, "body_mass_kg": 3.8531, "motor_torque_limit": 30.0,
                    "rotor_inertia": 0.005, "contact_zenith_deg": 51.0, "ball_radius_m": 0.09}
    spec = json.load(open(ROOT / "outputs/study/specs/spec_nominal.json"))["spec"]
    for k in nominal_vals:
        nominal_vals[k] = spec.get(k, nominal_vals[k])
    groups = {}
    for name, r in phy.items():
        if name not in sim:
            continue
        g = r["vary"]
        groups.setdefault(g, []).append(dict(name=name, value=num(r["value"]),
                                             sim={c.replace("sim_", "").replace("_fall", ""): num(sim[name][c]) for c in cols_sim},
                                             phys={c.replace("phys_", ""): num(r[c]) for c in cols_phy}))
    nomrow = dict(name="spec_nominal", sim={c.replace("sim_", "").replace("_fall", ""): num(nom[c]) for c in cols_sim},
                  phys={c.replace("phys_", ""): num(nom[c]) for c in cols_phy})
    out = []
    for g, rows in groups.items():
        rows.append(dict(nomrow, value=nominal_vals.get(g), nominal=True))
        rows.sort(key=lambda x: x["value"])
        out.append(dict(vary=g, nominal=nominal_vals.get(g), rows=rows))
    order = ["ball_mass_kg", "motor_torque_limit", "rotor_inertia", "body_com_height_m", "body_mass_kg", "contact_zenith_deg", "ball_radius_m"]
    out.sort(key=lambda x: order.index(x["vary"]) if x["vary"] in order else 99)
    return dict(src="S119; outputs/study/specs/ota_sim.csv, ota_sweep.csv, spec_nominal_sim.csv", groups=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--links", action="store_true", help="link and dash check over site/learn/")
    a = ap.parse_args()
    if a.links:
        check_links()
        return
    check_facts()
    if a.check:
        return
    b = json.load(open(BUNDLE))
    nominal = planar(ROOT / "ballbot_gym/models/ballbot_planar_model.json")
    designs = []
    for name, label, what in DESIGNS:
        d = planar(ROOT / f"ballbot_gym/models/variants/dz_{name}.json")
        t = b["designs"]["table"][name]
        d.update(name=name, label=label, what=what, tau_w=t["tau"], m_total=rnd(t["m_total"], 4), K_disc=rnd(t["lqr"]["K"]))
        designs.append(d)
    flat = ROOT / "outputs/study/warp_ppo/g34_mref_rsad_db_s0"
    full = json.load(open(flat / "motion_goal_ref_allflat.json"))
    lqro = json.load(open(flat / "motion_goal_ref_allflat_lqronly.json"))
    odo = json.load(open(ROOT / "outputs/study/warp_ppo/odom/odom_results_flat.json"))
    odo_res = []
    for k, r in odo["results"].items():
        label = k.split(" | ", 1)[1]
        odo_res.append(dict(label=label, model=odo["models"].get(label),
                            err_median=[rnd(r[f"err_{t}s_median_m"], 3) for t in (10, 30, 60)],
                            err_p90=[rnd(r[f"err_{t}s_p90_m"], 3) for t in (10, 30, 60)],
                            pct=[rnd(r[f"pct_of_distance_{t}s_median"], 3) for t in (10, 30, 60)],
                            dist=[rnd(r[f"dist_{t}s_median_m"], 3) for t in (10, 30, 60)],
                            yaw_deg=[rnd(r[f"yaw_err_{t}s_median_deg"], 3) for t in (10, 30, 60)]))
    data = dict(
        generated_by="scripts/site/build_learn_data.py",
        note="Simulation only. Every value is read from the repo files named in each block's src.",
        model=dict(nominal, src="ballbot_gym/models/ballbot_planar_model.json (scripts/study/ballbot_model.py)"),
        lqr=dict(b["lqr"], K_nom=rnd(b["designs"]["table"]["nom"]["lqr"]["K"]), src=f"{BUNDLE.relative_to(ROOT).as_posix()} lqr"),
        authority=dict(b["action"]["rs_adapt"], rs_fixed=b["action"]["rs"], combine=b["action"]["combine"], src="bundle.json action.rs_adapt; ros2 core/authority.py"),
        motion_ref=dict(b["motion_ref"], src="bundle.json motion_ref; experiments/warp_ppo/motion_ref.py"),
        goal_rule=b["goal_rule"],
        control=b["control"],
        designs=dict(rows=designs, src="ballbot_gym/models/variants/dz_<design>.json; bundle.json designs.table"),
        flat_goal=dict(src="S114 addendum; outputs/study/warp_ppo/g34_mref_rsad_db_s0/motion_goal_ref_allflat{,_lqronly}.json",
                       worlds_per_design=48, full=rnd(full["by_design"], 3), lqr=rnd(lqro["by_design"], 3),
                       full_all=rnd(full["fall"], 3), lqr_all=rnd(lqro["fall"], 3)),
        sweep=sweep(),
        odometry=dict(src="S108; outputs/study/warp_ppo/odom/odom_results_flat.json", rows=odo_res,
                      estimator={k: b["estimator"][k] for k in ("k_acc", "still_w", "still_g", "bias_tau", "q_acc", "r_enc")}),
        authority_curve={k: v for k, v in AUTHORITY_CURVE.items() if k != "needles"},
        seeds={k: v for k, v in SEEDS.items() if k != "needles"},
        spread=dict(src=SPREAD["src"], groups=[{k: v for k, v in g.items() if k != "needles"} for g in SPREAD["groups"]]),
        parity=parity(),
        motion_nav=dict(src=MOTION_NAV["src"], rows=[{k: v for k, v in r.items() if k != "needle"} for r in MOTION_NAV["rows"]]),
        facts=[dict(src=s, text=t, values=v, what=w) for s, t, v, w in LOG_FACTS if v is not None],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.1f} KiB)")
    print(f"nominal: pole {nominal['pole']} 1/s, zero {nominal['zero']}, lean {nominal['lean_deg']} deg per m/s^2")


if __name__ == "__main__":
    main()
