"""Regenerate every figure, table and cross-check of the generalist-study pages from the raw result files.

    python site/build_data.py            # figures, HTML table fragments, LaTeX tables, study.json, cross-check report, videos

Reads (read only):  outputs/study/warp_ppo/<run>/{log.csv,args.json,eval2_*.json}, eval_g2s0_curve.txt, percep/enc_v1.json,
                    wpdr_*/eval_{cpu,warp}.json, wpgen_none_s0/eval_tiles_v*.json, wpperc_*/eval_terrain.json,
                    outputs/study/mjx_port/bench_*.json, outputs/study/results/tracking_summary.csv, experiments/warp_ppo/tiles.py
Writes:             site/data/study.json, site/figures/study/*.svg (light and dark), paper_v2/figures/study_*.pdf,
                    paper_v2/tab_gen_*.tex, the BUILD blocks inside site/generalist.html and site/method.html,
                    site/videos/ (13 render clips), docs/research/site_crosscheck.md

Runs on the CPU only (numpy + matplotlib). When training finishes and new eval2_*.json files appear in the g9_* run folders,
re-run this script: the robustness tables and figure fill themselves in; nothing is edited by hand.

Numbers that exist only in the study log (no data file) live in LOG_ONLY below, each with its log entry, and are marked with a dagger in the output.
"""
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
WP = ROOT / "outputs/study/warp_ppo"
RES = ROOT / "outputs/study/results"
PAPER = ROOT / "paper_v2"
# --site-only: rebuild the site without touching paper_v2/ or docs/ (paper tables and figures and the cross-check report go to a temp folder)
SITE_ONLY = "--site-only" in sys.argv
if SITE_ONLY:
    import tempfile
    PAPER = Path(tempfile.gettempdir()) / "ballbot_site_build" / "paper_v2"
    PAPER.mkdir(parents=True, exist_ok=True)
FIG = SITE / "figures/study"
FIG.mkdir(parents=True, exist_ok=True)
(PAPER / "figures").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments/warp_ppo"))

KINDS = ["flat", "step_up", "step_down", "small_drop", "large_drop", "ramp_up", "ramp_down", "stairs", "bump", "rough"]
KIND_LABEL = {k: k.replace("_", " ") for k in KINDS}
KIND_LABEL["blocks"] = "blocks (8 cm, navigation map)"
DAG = "&dagger;"          # html dagger: value taken from the study log (no data file)
DAGT = r"$^\dagger$"
CHECKS = []               # (id, where, quantity, computed, logged, tol, note)


# ----------------------------------------------------------------------------------------------------------- data layer
SNAP = SITE / "data/eval_snapshot"
REFRESH = "--refresh" in sys.argv
# Evaluation JSON names do not encode DR_SCALE / PUSH_SCALE / EVAL_NOEXIT, so a later evaluation with other settings silently overwrites them.
# The site therefore reads the frozen copy in data/eval_snapshot/ when it exists (python site/build_data.py --refresh re-copies the live files).
QUARANTINE = {("g7_lqrres_s0", "eval2_goal_goal_dr_surf.json"), ("g7_lqrres_s0", "eval2_goal_goal_dr_surf_pull.json")}   # overwritten 2026-11-07 07:19 to 07:24 by evaluations with other settings
FALLBACK = {
    # first read of g7_lqrres_s0 goal evaluation (before the overwrite); overall values agree with log S60/S62; disturbed per-kind values were not read
    ("g7_lqrres_s0", "eval2_goal_goal_dr_surf.json"): dict(
        o=dict(fall=0.530, success=0.470, median_time_s=2.23),
        k={"flat": 0.6534, "step_up": 0.25, "step_down": 0.6364, "small_drop": 0.5739, "large_drop": 0.3580, "ramp_up": 0.4886, "ramp_down": 0.6932, "stairs": 0.3352, "bump": 0.1307, "rough": 0.5852}),
    ("g7_lqrres_s0", "eval2_goal_goal_dr_surf_pull.json"): dict(o=dict(fall=0.819, success=0.181, median_time_s=2.22), k={}),
}


def lqr_alone_rows():
    """Latest printed rows of lqr_gpu.py for the nominal design at 0.3, 0.6, 0.9 m/s (interior-tile re-measurement, S92 addendum); None if no log holds all three."""
    rows = {}
    for f in sorted(WP.glob("*.log"), key=lambda q: q.stat().st_mtime):
        try:
            txt = f.read_text(encoding="utf8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r"LQR nom\s+cmd (0\.[369]): steady-state err ([0-9.]+|nan), achieved/command ([0-9.]+|nan), falls ([0-9.]+)", txt):
            sp = float(m.group(1))
            e = None if m.group(2) == "nan" else float(m.group(2))
            r = None if m.group(3) == "nan" else float(m.group(3))
            rows[sp] = (e, r, float(m.group(4)), f.name)
    return rows if len(rows) == 3 else None


LQR_ROWS = None


def refresh_snapshot():
    n = 0
    for pat in ("g*/eval2_*.json", "wpdr_*/eval_*.json", "wpgen_none_s0/eval_tiles_v*.json", "wpperc_*/eval_terrain.json", "wpedge_*/eval_terrain.json"):
        for f in WP.glob(pat):
            (SNAP / f.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, SNAP / f.parent.name / f.name)
            n += 1
    print(f"snapshot refreshed: {n} files")


def J(run, name):
    for p in (SNAP / run / name, WP / run / name):
        if p == WP / run / name and (run, name) in QUARANTINE and not REFRESH:
            return None
        if p.exists():
            return json.load(open(p))
    return None


def tile_eval(run, name):
    """Tile evaluation (eval_tiles2.py): overall and per-kind cells with world counts."""
    d = J(run, name)
    if d is None:
        fb = FALLBACK.get((run, name))
        if fb is None:
            return None
        return dict(d=None, o=fb["o"], k={k: dict(success=fb["k"].get(k, float("nan")), fall=float("nan")) for k in KINDS}, n_overall=1760, n_kind=176, args=dict(n=16), fallback=True)
    a = d["args"]
    D = len(d["by_design"])
    n_cell = a["n"] * D                    # worlds per terrain kind
    return dict(d=d, o=d["overall"], k=d["by_kind"], n_overall=n_cell * len(KINDS), n_kind=n_cell, args=a, fallback=False)


def flat_cell(run, sp):
    """Clean flat benchmark row (TILES_FLAT=1 EVAL_NOEXIT=1): the 'flat' slot of the evaluation, 2 x 32 worlds."""
    d = J(run, f"eval2_vel_{sp}.json")
    if d is None:
        return None
    r = d["by_kind"]["flat"]
    a = d["args"]
    n = a["n"] * len(a["eval_designs"].split(","))
    return dict(err=r["late_err"], ratio=r["late_speed_ratio"], fall=r["fall"], n=n)


def curve_txt():
    """eval_g2s0_curve.txt: disturbance-scale curve of g2_s0 (the matching JSON was overwritten by later runs)."""
    out, cur = {}, None
    for fn in ("eval_g2s0_curve.txt", "eval_g2s0_pull.txt"):
        for ln in open(WP / fn, encoding="utf8"):
            if ln.startswith("##"):
                cur = ln.strip()
            elif ln.startswith("{") and cur:
                v = json.loads(ln)
                if "scaled" in cur:
                    out[float(cur.split("scaled")[1])] = v
                elif cur.endswith("push_scale 1 flags: --pull"):
                    out[1.0] = v
                elif "push_scale 0" in cur:
                    out["off"] = v
                elif cur.endswith("push_scale 1 flags:"):
                    out["push_only"] = v
    return out


def cmp_txt():
    """eval_cmp.txt: fixed-eval comparison of the gfix runs (push_scale 0 and 1, DR + surfaces + pulls, 0.3 m/s)."""
    out, cur = {}, None
    for ln in open(WP / "eval_cmp.txt", encoding="utf8"):
        if ln.startswith("##"):
            parts = ln.split()
            cur = (parts[1], int(float(parts[3])))
        elif ln.startswith("{") and cur:
            out[cur] = json.loads(ln)
    return out


def wilson(p, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def check(cid, where, quantity, computed, logged, tol=0.006, note=""):
    CHECKS.append((cid, where, quantity, computed, logged, tol, note))


def read_log(run):
    p = WP / run / "log.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return a


# numbers that exist only in the study log (value, log entry)
LOG_ONLY = {
    "lqr_gpu": {0.3: (0.014, 1.00, 0.00), 0.6: (0.017, 1.00, 0.00), 0.9: (None, None, 0.89)},   # S92 addendum: LQR alone re-measured on interior tiles (nominal, DR off, 64 worlds, 8 s); err and ratio at 0.9 not printed
    "s26_blind_ridge": dict(p1=(0.59, 0.84), p2=(0.16, 0.38), p3=(0.02, 0.08), up=(0.23, 0.19)),   # S26 blind row at 0.3 m/s (JSON overwritten by a 0.6 m/s run)
}

# ----------------------------------------------------------------------------------------------------------- helpers: output
def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def html_table(head, rows, cls="num", first_left=True):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody class="{cls}">{b}</tbody></table></div>'


def tex_escape(s):
    s = s.replace("<sup>2</sup>", "$^2$").replace("&dagger;", DAGT).replace("&plusmn;", r"$\pm$").replace("&times;", r"$\times$").replace("&ge;", r"$\ge$")
    s = s.replace("<br>", " ")
    s = re.sub(r"<[^>]+>", "", s)
    keep = {DAGT: "@@DAG@@", "$^2$": "@@SQ@@", r"$\pm$": "@@PM@@", r"$\times$": "@@TM@@", r"$\ge$": "@@GE@@"}
    for k, v in keep.items():
        s = s.replace(k, v)
    s = s.replace("m^2", "m@@SQ@@")
    s = s.replace("s^2", "s@@SQ@@")
    s = s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    for k, v in keep.items():
        s = s.replace(v, k)
    s = s.replace("PENDING", r"\textcolor{red}{PENDING}")
    return s


def tex_table(name, caption, label, head, rows, colspec=None, star=False, small=True):
    cs = colspec or "l" + "c" * (len(head) - 1)
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[t]\centering{'\\footnotesize' if small else ''}", rf"\caption{{{caption}}}\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%" if star else r"\resizebox{\columnwidth}{!}{%", rf"\begin{{tabular}}{{{cs}}}\toprule",
             " & ".join(tex_escape(h) for h in head) + r"\\\midrule"]
    for r in rows:
        lines.append(" & ".join(tex_escape(str(c)) if not str(c).startswith("TEX:") else str(c)[4:] for c in r) + r"\\")
    lines += [r"\bottomrule\end{tabular}}", rf"\end{{{env}}}"]
    (PAPER / f"tab_gen_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf8")


TABLE_RE = re.compile(r'<div class="tablewrap">(<table>.*?</table>)</div>\s*<p class="cap"><b>Table ([^<]+?)\.</b>\s*(.*?)</p>', re.S)


def componentize(html):
    """Rewrite table + caption pairs into the shared numbered-table component of site.css (caption above, scroll box, number kept via data-num)."""
    return TABLE_RE.sub(lambda m: f'<div class="table" data-num="{m.group(2)}"><p class="caption" data-num="{m.group(2)}">{m.group(3)}</p><div class="scroll">{m.group(1)}</div></div>', html)


def inject(page, name, html):
    html = componentize(html)
    p = SITE / page
    s = p.read_text(encoding="utf8")
    pat = re.compile(rf"(<!-- BUILD:{re.escape(name)} -->)(.*?)(<!-- /BUILD:{re.escape(name)} -->)", re.S)
    if not pat.search(s):
        print(f"WARNING: no BUILD:{name} block in {page}")
        return
    s = pat.sub(lambda m: m.group(1) + "\n" + html + "\n" + m.group(3), s)
    p.write_text(s, encoding="utf8")


def setup_cap(engine="Warp GPU (MuJoCo-Warp 3.17, float32)", dr="off", noise="off", worlds="", seeds="", protocol="", extra=""):
    parts = [f"Engine: {engine}.", f"Domain randomisation: {dr}.", f"Observation noise: {noise}.", f"Worlds per cell: {worlds}.", f"Seeds: {seeds}.", f"Protocol: {protocol}"]
    if extra:
        parts.append(extra)
    return " ".join(parts)


def figure_html(name, alt, caption, wide=False):
    return (f'<figure class="sfig"><img class="fig-light" src="figures/study/{name}_light.svg" alt="{alt}" loading="lazy">'
            f'<img class="fig-dark" src="figures/study/{name}_dark.svg" alt="{alt}" loading="lazy"><figcaption>{caption}</figcaption></figure>')


THEMES = {
    "light": dict(fg="#0f1722", mute="#4a5a70", grid="#d9e0ea", model="#1478d4", learned="#6d4ae0", hybrid="#0aa88c", warn="#c77700", bad="#d93a3a", good="#1f9d55", face="none"),
    "dark": dict(fg="#e8eef7", mute="#a9b6c9", grid="#2f3e55", model="#4aa8ff", learned="#a78bfa", hybrid="#2ee6c4", warn="#ffb454", bad="#ff6b6b", good="#5ee08a", face="none"),
    "paper": dict(fg="#000000", mute="#333333", grid="#cccccc", model="#1f4e9c", learned="#6d4ae0", hybrid="#0a8a72", warn="#b36200", bad="#c0392b", good="#1f7a3f", face="white"),
}


def render(name, draw, size=(9.0, 3.4), paper_size=None):
    """Draw with each theme: SVG (transparent) for the site, PDF (white) for the paper."""
    for th, C in THEMES.items():
        plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "ballbot", "font.family": "sans-serif", "font.size": 9, "text.color": C["fg"],
                             "axes.labelcolor": C["fg"], "xtick.color": C["mute"], "ytick.color": C["mute"], "axes.edgecolor": C["mute"],
                             "axes.facecolor": "none", "figure.facecolor": C["face"], "savefig.facecolor": C["face"], "axes.titlesize": 9.5,
                             "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8, "axes.grid": True,
                             "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
        fig = draw(C, paper_size if (th == "paper" and paper_size) else size)
        if th == "paper":
            fig.savefig(PAPER / "figures" / f"study_{name}.pdf", bbox_inches="tight")
        else:
            fig.savefig(FIG / f"{name}_{th}.svg", bbox_inches="tight", transparent=True, metadata={"Date": None})
        plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- run families
PURE = [("g2_track_s0", "tracking weight x1"), ("g4_trk_s0", "x2"), ("g6_trk4_s0", "x4"), ("g5_mix_s0", "x2, half goals, cross-focus")]
RES_RUNS = [("g7_lqrres_s0", "LQR + residual, authority 0.5, seed 0"), ("g7_lqrres_s1", "LQR + residual, authority 0.5, seed 1")]
SP = ["0.3", "0.6", "0.9"]
S = {}      # everything that goes into study.json


RUNSPEC = [("g12_pure_s0", "pure learning (no LQR), 300M steps", "pure", "seed 0"),
           ("g12_res_s0", "LQR + residual 0.5, standard randomisation", "res", "seed 0"), ("g12_res_s1", "LQR + residual 0.5, standard randomisation", "res", "seed 1"),
           ("g12_dr175_s0", "LQR + residual 0.5, randomisation ramped to 1.75x", "dr175", "seed 0"), ("g12_dr175_s1", "LQR + residual 0.5, randomisation ramped to 1.75x", "dr175", "seed 1"),
           ("g12_soft175_s0", "soft LQR base (R x3, a_max 0.3), ramped to 1.75x", "soft", "seed 0"), ("g12_soft175_s1", "soft LQR base (R x3, a_max 0.3), ramped to 1.75x", "soft", "seed 1"),
           ("g13_softrs075_s0", "soft LQR base, authority 0.75", "soft075", "seed 0"),
           ("g14_fp5_s0", "ramped 1.75x, fall penalty 5 (was 2)", "fp5", "seed 0"), ("g14_softfp5_s0", "soft base, fall penalty 5", "softfp5", "seed 0"),
           ("g14_pramp_s0", "ramped 1.75x, push and pull strength ramped 0.2 to 1", "pramp", "seed 0"),
           ("g15_edge_depth_s0", "ramped 1.75x + edge starts + depth sensor", "edgedepth", "seed 0"), ("g15_edge_blind_s0", "ramped 1.75x + edge starts, blind control", "edgeblind", "seed 0"),
           ("g12_pure_s1", "pure learning (no LQR), 300M steps", "pure", "seed 1"),
           ("g16_ret100_s0", "ramped 1.75x, trained with the ball retainer (arrestors at 100 deg)", "ret", "seed 0"), ("g16_ret100_s1", "ramped 1.75x, trained with the ball retainer (arrestors at 100 deg)", "ret", "seed 1"),
           ("g16_pure_ret100_s0", "pure learning, trained with the retainer", "pureret", "seed 0"), ("g16_pure_ret100_s1", "pure learning, trained with the retainer", "pureret", "seed 1"),
           ("g17_ret100_trk3_s0", "retainer, authority 0.5, tracking weight x3", "trk3", "seed 0"),
           ("g18_ret_rs015_s0", "retainer, authority 0.15", "ret015", "seed 0"), ("g18_ret_rs025_s0", "retainer, authority 0.25 (then-recommended)", "ret025", "seed 0"),
           ("g18_ret_rs025_s1", "retainer, authority 0.25 (then-recommended)", "ret025", "seed 1"), ("g18_ret_rs035_s0", "retainer, authority 0.35", "ret035", "seed 0"),
           ("g18_ret_soft_s0", "retainer, soft LQR base, authority 0.5", "retsoft", "seed 0"),
           ("g18_ret_rs025_s2", "retainer, authority 0.25 (then-recommended)", "ret025", "seed 2"), ("g19_ret_rs025_400m_s0", "retainer, authority 0.25, 400M steps", "ret025_400", "seed 0"),
           ("g20_ret_rs025_depth_s0", "retainer, authority 0.25, depth sensor + edge starts", "ret025depth", "seed 0"),
           ("g21_ret_rs025_tau3_s0", "retainer, authority 0.25, 12 training designs (adds tau3; 1920 worlds per goal row)", "ret025_12", "seed 0"),
           ("g21_ret_rs025_tau3_s1", "retainer, authority 0.25, 12 training designs (adds tau3; 1920 worlds per goal row)", "ret025_12", "seed 1"),
           ("g22_ret_rs025_scan_s0", "retainer, authority 0.25, privileged height scan input", "scan025", "seed 0"), ("g22_ret_rs05_scan_s0", "retainer, authority 0.5, privileged height scan input", "scan05", "seed 0"),
           ("g22_pure_ret_scan_s0", "pure learning, retainer, privileged height scan input", "scanpure", "seed 0"),
           ("g26_ret_rs025_mu01_s0", "then-recommended recipe on the corrected contact model (free-rolling friction 0.01; then-recommended checkpoint)", "ret025mu", "seed 0"),
           ("g26_ret_rs025_mu01_s1", "then-recommended recipe on the corrected contact model (free-rolling friction 0.01; then-recommended checkpoint)", "ret025mu", "seed 1"),
           ("g26_ret_rs025_mu01_s2", "then-recommended recipe on the corrected contact model (free-rolling friction 0.01; then-recommended checkpoint)", "ret025mu", "seed 2"),
           ("g23_nav_rs025_s0", "then-recommended recipe, trained with the planner in the loop (30 % of worlds on blocks tiles)", "navtrain", "seed 0"), ("g23_nav_rs025_s1", "then-recommended recipe, trained with the planner in the loop (30 % of worlds on blocks tiles)", "navtrain", "seed 1")]
PENDING_RUNS = []
PEND = '<span class="pend">PENDING</span>'
SCALES = ("1", "1.75", "2.5")
ENG = "Warp GPU (MuJoCo-Warp 3.17, float32), corrected for the S69 stale-subtree-mass error"
ENG_NAV18 = ENG + "; earlier measurement: controller g18_ret_rs025_s1 trained and evaluated at wheel-ball free-rolling friction 0.001 (the then-recommended checkpoint at 0.01 is in Table 37)"


def battery(run):
    p = WP / run / "battery.json"
    if p.exists():
        return json.load(open(p))
    d = WP / run      # battery still running: assemble the same structure from the eval2 files already written (same names and settings as eval_battery.py)
    fl = {}
    for sp in SP:
        f = d / f"eval2_vel_{sp}_flat_noexit.json"
        if f.exists():
            k = json.load(open(f))["by_kind"]["flat"]
            fl[sp] = dict(err=k["late_err"], achieved=k["late_speed_ratio"], falls=k["fall"], worlds=64)
    out = dict(run=run, flat=fl, goal={}, vel={}, partial=True)
    for task in ("goal", "vel"):
        stem = "eval2_goal_goal_dr_surf" if task == "goal" else "eval2_vel_0.3_dr_surf"
        for sc in ("1", "1.75", "2.5"):
            suf = "" if sc == "1" else f"_drs{sc}"
            for name, mid in (("none", "_ps0"), ("pushes+pulls", "_pull")):
                f = d / f"{stem}{mid}{suf}.json"
                if f.exists():
                    j = json.load(open(f))
                    out[task][f"drs{sc}_{name}"] = dict(overall=j["overall"], by_kind=j.get("by_kind"), env=j.get("env_settings"))
    need = len(fl) == 3 and all(f"drs{sc}_{n}" in out["goal"] for sc in ("1", "1.75", "2.5") for n in ("none", "pushes+pulls")) and all(f"drs1_{n}" in out["vel"] for n in ("none", "pushes+pulls"))
    return out if need else None


def run_progress(run):
    lg = read_log(run)
    a = json.load(open(WP / run / "args.json")) if (WP / run / "args.json").exists() else {}
    return "not started" if lg is None else f"training {lg['steps'][-1] / 1e6:.0f}M of {a.get('steps', 0) / 1e6:.0f}M steps"


FROZEN_PATH = SITE / "data/flat_frozen.json"
FROZEN = json.load(open(FROZEN_PATH)) if (FROZEN_PATH.exists() and not REFRESH) else {}
if REFRESH and FROZEN_PATH.exists():
    FROZEN_PATH.unlink()      # the flat re-run is complete: use the live battery rows from now on


def load_batteries():
    S["flat"], S["goal"], S["vel_tiles"], S["goalscale"] = {}, {}, {}, {}
    S["flat_frozen"] = bool(FROZEN)
    for run, lab, fam, sd in RUNSPEC:
        b = battery(run)
        if not b:
            continue
        S["flat"][run] = {sp: dict(err=v["err"], ratio=v["achieved"], fall=v["falls"], n=v["worlds"]) for sp, v in b["flat"].items()}
        if run in FROZEN:      # keep the pre-correction rows until the whole re-run is in, so the table is not a mix
            S["flat"][run] = FROZEN[run]
        g = b["goal"]
        u, d = g["drs1_none"], g["drs1_pushes+pulls"]
        S["goal"][run] = dict(u=u["overall"], d=d["overall"], ku={k: u["by_kind"][k]["success"] for k in KINDS}, kd={k: d["by_kind"][k]["success"] for k in KINDS}, fallback=False, n=1760, nk=176)
        S["goalscale"][run] = {sc: dict(u=g[f"drs{sc}_none"]["overall"], d=g[f"drs{sc}_pushes+pulls"]["overall"]) for sc in SCALES}
        v = b["vel"]
        S["vel_tiles"][run] = dict(fall_u=v["drs1_none"]["overall"]["fall"], fall_d=v["drs1_pushes+pulls"]["overall"]["fall"], err_u=v["drs1_none"]["overall"]["track_err"], err_d=v["drs1_pushes+pulls"]["overall"]["track_err"],
                                   fall_u25=(v["drs2.5_none"]["overall"]["fall"] if "drs2.5_none" in v else None), n=1760)


def build_flat_table():
    load_batteries()
    rows, texrows = [], []
    for sp_ in SP:
        pass
    global LQR_ROWS
    LQR_ROWS = lqr_alone_rows()

    def lqr_cell(sp):
        if LQR_ROWS:
            e, r, fl, _ = LQR_ROWS[float(sp)]
        else:
            e, r, fl = LOG_ONLY["lqr_gpu"][float(sp)]
        dag = "" if LQR_ROWS else DAG
        txt = f"{e:.3f} / {r:.2f} / {fl:.2f}{dag}" if e is not None else f"n/a / n/a / {fl:.2f}{dag} (most worlds fell)"
        return txt, e, r, fl
    cells = [lqr_cell(sp) for sp in SP]
    r0 = ["<b>LQR alone</b><br><small>GPU LQR (lqr_gpu.py): no learning, nominal design</small>"] + [c[0] for c in cells] + ["n/a (no learning)"]
    rows.append(r0)
    texrows.append(["LQR alone"] + [re.sub(r"<[^>]+>", "", c) for c in r0[1:]])
    S["flat"]["LQR"] = {sp: dict(err=c[1], ratio=c[2], fall=c[3], n=64) for sp, c in zip(SP, cells)}
    S["lqr_rows_source"] = LQR_ROWS[0.3][3] if LQR_ROWS else "study log (S92 addendum)"
    V["lqr.fall09"] = S["flat"]["LQR"]["0.9"]["fall"]; V["lqr.err03"] = S["flat"]["LQR"]["0.3"]["err"]; V["lqr.err06"] = S["flat"]["LQR"]["0.6"]["err"]
    for run, lab, fam, sd in RUNSPEC:
        if run not in S["flat"]:
            continue
        cells = [f"{S['flat'][run][sp]['err']:.3f} / {S['flat'][run][sp]['ratio']:.2f} / {S['flat'][run][sp]['fall']:.2f}" for sp in SP]
        rows.append([f"<b>{lab}</b><br><small>{run}, {sd}</small>"] + cells + [sd])
        texrows.append([f"{lab} ({sd})"] + cells + [sd])
    head = ["Controller", "0.3 m/s: err / achieved / falls", "0.6 m/s", "0.9 m/s", "Seed"]
    cap = setup_cap(engine=ENG, dr="off", noise="off", worlds="64 per cell", seeds="one per row as listed (seeds 0 and 1 where both are shown)",
                    protocol="perfectly flat 24 m world on interior tiles (at least 7.5 m from the map border, the S92 correction), nominal design, constant velocity command at a random heading, 800 control steps, error and achieved/commanded speed averaged over steps 300 to 799 of worlds still upright, falls = share of worlds that fell (eval_battery.py; every flat row re-run after S92).",
                    extra="Error is the Euclidean velocity error in m/s. All learned rows are corrected-engine runs." + ((" The LQR-alone row is the interior-tile re-measurement of the S92 addendum (source: " + S.get("lqr_rows_source", "study log") + "); a dagger marks values taken from the log.")))
    S["flat_caption"] = cap
    inject("generalist.html", "tab_flat", html_table(head, rows) + f'<p class="cap"><b>Table 1.</b> Clean flat-world tracking benchmark. {cap}</p>')
    tex_table("flat", "Clean flat-world velocity-tracking benchmark, corrected engine. Cells: steady-state error [m/s] / achieved over commanded speed / fall rate. " + tex_escape(cap), "tab:gen_flat", head, texrows, colspec="lcccc", star=True)


def build_goal_vel_tables():
    gr, vr, rr = [], [], []
    for run, lab, fam, sd in RUNSPEC:
        if run not in S["goal"]:
            continue
        g, v = S["goal"][run], S["vel_tiles"][run]
        u, d = g["u"], g["d"]
        gr.append([f"{lab}<br><small>{run}, {sd}</small>", f"{u['fall']:.2f} / {u['success']:.2f}", f"{u['median_time_s']:.2f}", *[f"{g['ku'][k]:.2f}" for k in ("step_up", "stairs", "bump", "ramp_up", "flat")],
                   f"{d['fall']:.2f} / {d['success']:.2f}", sd])
        vr.append([f"{lab}<br><small>{run}, {sd}</small>", f"{v['fall_u']:.2f}", f"{v['fall_d']:.2f}", f"{v['err_u']:.3f} / {v['err_d']:.3f}", sd])
        gs = S["goalscale"][run]
        rr.append([f"{lab}<br><small>{run}, {sd}</small>"] + [f"{gs[sc]['u']['success']:.2f} / {gs[sc]['d']['success']:.2f}" for sc in SCALES] + [(f"{v['fall_u25']:.2f}" if v['fall_u25'] is not None else "n/a (evaluation running)"), sd])
    for run, lab in PENDING_RUNS:
        gr.append([f"{lab}<br><small>{run}; {run_progress(run)}</small>"] + [PEND] * 8 + ["seed 0"])
        vr.append([f"{lab}<br><small>{run}; {run_progress(run)}</small>"] + [PEND] * 3 + ["seed 0"])
        rr.append([f"{lab}<br><small>{run}; {run_progress(run)}</small>"] + [PEND] * 4 + ["seed 0"])
    head2 = ["Policy", "Undisturbed: falls / success", "median time [s]", "step up", "stairs", "bump", "ramp up", "flat", "Pushes + pulls: falls / success", "Seed"]
    cap2 = setup_cap(engine=ENG, dr="on at scale 1 (mass, friction, damping, torque, latency, floor classes, 11 designs)", noise="on", worlds="16 per (design, terrain kind): 176 per terrain kind, 1760 per row",
                     seeds="one per row as listed", protocol="goal 1.1 m from a random start on a tile, success = within 0.2 m before any fall within 800 steps (eval_battery.py); terrain columns are undisturbed success per kind; undisturbed = pushes and pulls off, disturbed = pushes 5 to 20 N and pulls 3 to 12 N.",
                     extra="g14 rows are one-variable changes of the ramped (B) or soft (C) recipe.")
    inject("generalist.html", "tab_goal", html_table(head2, gr) + f'<p class="cap"><b>Table 2.</b> Goal task on the tiled terrain, corrected engine. {cap2}</p>')
    tex_table("goal", "Goal task on the tiled terrain, corrected engine. " + tex_escape(cap2), "tab:gen_goal", head2, gr, colspec="lcccccccc" + "c", star=True)
    head3 = ["Policy", "Falls, undisturbed", "Falls, pushes + pulls", "Tracking error undisturbed / disturbed [m/s]", "Seed"]
    cap3 = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="16 per (design, terrain kind): 1760 per cell", seeds="one per row as listed",
                     protocol="velocity task at 0.3 m/s at a random heading on the tiles with floor classes, 800 steps (eval_battery.py); tracking error is the mean over upright steps.")
    inject("generalist.html", "tab_veltiles", html_table(head3, vr) + f'<p class="cap"><b>Table 3.</b> Velocity task on the tiled terrain, corrected engine. {cap3}</p>')
    tex_table("veltiles", "Velocity task at 0.3\\,m/s on the tiled terrain, corrected engine. " + tex_escape(cap3), "tab:gen_veltiles", head3, vr, colspec="lcccc")
    head8 = ["Policy", "Goal success, DR 1: undisturbed / disturbed", "DR 1.75", "DR 2.5", "Tile velocity falls, DR 2.5 undisturbed", "Seed"]
    cap8 = setup_cap(engine=ENG, dr="on; evaluated at the scale in the column (policies trained at scale 1 or ramped to 1.75)", noise="on", worlds="1760 per cell", seeds="one per row as listed",
                     protocol="goal task of Table 2 at DR scales 1, 1.75 and 2.5 (the master scale multiplies mass, friction, torque, damping, latency and noise ranges, not push or pull strength).")
    inject("generalist.html", "tab_robust", html_table(head8, rr) + f'<p class="cap"><b>Table 4.</b> Goal success against randomisation scale, corrected engine. {cap8}</p>')
    tex_table("robust", "Goal success against randomisation scale, corrected engine. " + tex_escape(cap8), "tab:gen_robust", head8, rr, colspec="lcccc" + "c", star=True)
    LG = {"g12_res_s0": ((0.58, 0.52, 0.45), (0.25, 0.23, 0.18), 2.61, (0.33, 0.38, 0.17, 0.74), (0.55, 0.72)), "g12_dr175_s0": ((0.58, 0.54, 0.45), (0.29, 0.25, 0.19), 2.44, (0.31, 0.44, 0.26, 0.65), (0.54, 0.71)),
          "g12_soft175_s0": ((0.63, 0.57, 0.49), (0.33, 0.29, 0.23), 3.06, (0.35, 0.39, 0.16, 0.75), (0.55, 0.71)), "g12_res_s1": ((0.58, 0.54, 0.44), (0.27, 0.25, 0.19), 2.43, (0.28, 0.39, 0.20, 0.69), (0.55, 0.74)),
          "g12_dr175_s1": ((0.60, 0.56, 0.47), (0.30, 0.25, 0.19), 2.3, None, (0.53, 0.71)), "g12_soft175_s1": ((0.62, 0.58, 0.52), (0.34, 0.29, 0.24), 3.0, None, (0.53, 0.73)),
          "g13_softrs075_s0": ((0.58, 0.53, 0.45), (0.32, 0.26, 0.21), 3.3, None, (0.54, 0.70)), "g12_pure_s0": ((0.44, 0.43, 0.37), (0.26, 0.23, 0.18), 3.1, None, (0.32, 0.54))}
    for run, (u, d, t, kk, vv) in LG.items():
        for k, sc in enumerate(SCALES):
            check("S72-S74", f"Table 2/4 {run}", f"goal success undisturbed DR {sc}", S["goalscale"][run][sc]["u"]["success"], u[k], 0.006)
            check("S72-S74", f"Table 2/4 {run}", f"goal success disturbed DR {sc}", S["goalscale"][run][sc]["d"]["success"], d[k], 0.006)
        check("S72-S74", f"Table 2 {run}", "median time", S["goal"][run]["u"]["median_time_s"], t, 0.13)
        if kk:
            for q, c, l in zip(("step_up", "stairs", "bump", "ramp_up"), [S["goal"][run]["ku"][x] for x in ("step_up", "stairs", "bump", "ramp_up")], kk):
                check("S72/S73", f"Table 2 {run}", q, c, l, 0.006)
        check("S72-S74", f"Table 3 {run}", "vel falls undisturbed", S["vel_tiles"][run]["fall_u"], vv[0], 0.006)
        check("S72-S74", f"Table 3 {run}", "vel falls disturbed", S["vel_tiles"][run]["fall_d"], vv[1], 0.006)
    check("S73", "text", "res s0 vel err undisturbed", S["vel_tiles"]["g12_res_s0"]["err_u"], 0.112, 0.0006)
    check("S73", "text", "res s1 vel err disturbed", S["vel_tiles"]["g12_res_s1"]["err_d"], 0.205, 0.0006)


def build_g14_heldout():
    LG = {"g14_fp5_s0": ((0.57, 0.53, 0.46), (0.27, 0.25, 0.18), (0.54, 0.73)), "g14_softfp5_s0": ((0.63, 0.59, 0.50), (0.33, 0.27, 0.22), (0.54, 0.71)), "g14_pramp_s0": ((0.57, 0.52, 0.47), (0.25, 0.21, 0.16), (0.54, 0.73))}
    for run, (u, d, v) in LG.items():
        for k, sc in enumerate(SCALES):
            check("S76", f"Table 2/4 {run}", f"goal undisturbed DR {sc}", S["goalscale"][run][sc]["u"]["success"], u[k], 0.006)
            check("S76", f"Table 2/4 {run}", f"goal disturbed DR {sc}", S["goalscale"][run][sc]["d"]["success"], d[k], 0.006)
        check("S76", f"Table 3 {run}", "vel falls undisturbed", S["vel_tiles"][run]["fall_u"], v[0], 0.006)
        check("S76", f"Table 3 {run}", "vel falls disturbed", S["vel_tiles"][run]["fall_d"], v[1], 0.006)
    V["g14.max_dist_gain"] = max(S["goalscale"][r]["1"]["d"]["success"] for r in LG) - max(S["goalscale"][r]["1"]["d"]["success"] for r in ("g12_dr175_s0", "g12_dr175_s1", "g12_soft175_s0", "g12_soft175_s1"))
    V["g14.pramp_d"] = S["goalscale"]["g14_pramp_s0"]["1"]["d"]["success"]
    # held-out terrain (tile seed 1)
    rows, texrows = [], []
    S["heldout"] = {}
    for run, sd in (("g12_dr175_s0", "seed 0"), ("g12_dr175_s1", "seed 1")):
        du = json.load(open(WP / run / "eval2_goal_goal_dr_surf_ps0_ts1.json"))
        dd = json.load(open(WP / run / "eval2_goal_goal_dr_surf_pull_ts1.json"))
        uo, do = du["overall"], dd["overall"]
        S["heldout"][run] = dict(u=uo, d=do, tile_seed=du["args"]["tile_seed"])
        V[f"held.{run}.u"], V[f"held.{run}.d"] = uo["success"], do["success"]
        V[f"held.{run}.drop_u"] = S["goalscale"][run]["1"]["u"]["success"] - uo["success"]
        V[f"held.{run}.drop_d"] = S["goalscale"][run]["1"]["d"]["success"] - do["success"]
        ref_u, ref_d = S["goalscale"][run]["1"]["u"]["success"], S["goalscale"][run]["1"]["d"]["success"]
        c = [f"{uo['success']:.2f}", f"{ref_u:.2f}", f"{do['success']:.2f}", f"{ref_d:.2f}", f"{uo['median_time_s']:.2f} / {do['median_time_s']:.2f}"]
        rows.append([f"ramped to 1.75x, standard residual<br><small>{run}, {sd}</small>"] + c + [sd])
        texrows.append([f"ramped to 1.75x ({run}, {sd})"] + c + [sd])
    head = ["Policy", "Undisturbed success, tile seed 1 (held out)", "tile seed 0 (trained)", "Disturbed success, tile seed 1", "tile seed 0", "Median time [s], undisturbed / disturbed", "Seed"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="16 per (design, terrain kind): 176 per terrain kind, 1760 per cell", seeds="seeds 0 and 1 of the recipe",
                    protocol="goal task of Table 2 on a freshly generated tile map (tile seed 1: new severities and positions, same ten kinds and ranges) while the policy was trained on tile seed 0; undisturbed = pushes and pulls off, disturbed = on.")
    inject("generalist.html", "tab_heldout", html_table(head, rows) + f'<p class="cap"><b>Table 5.</b> Held-out terrain layout. {cap}</p>')
    tex_table("heldout", "Goal task on held-out terrain (tile seed 1), corrected engine. " + tex_escape(cap), "tab:gen_heldout", head, texrows, colspec="lcccccc", star=True)
    for run, (u, d, t1, t2) in {"g12_dr175_s0": (0.52, 0.26, 2.4, 2.4), "g12_dr175_s1": (0.53, 0.26, 2.3, 2.2)}.items():
        check("S76", "Table 5", f"{run} held-out undisturbed", S["heldout"][run]["u"]["success"], u, 0.006)
        check("S76", "Table 5", f"{run} held-out disturbed", S["heldout"][run]["d"]["success"], d, 0.006)
        check("S76", "Table 5", f"{run} held-out time undisturbed", S["heldout"][run]["u"]["median_time_s"], t1, 0.06)
        check("S76", "Table 5", f"{run} held-out time disturbed", S["heldout"][run]["d"]["median_time_s"], t2, 0.06)
    V["held.drop_u_min"] = min(V[f"held.{r}.drop_u"] for r in S["heldout"]); V["held.drop_u_max"] = max(V[f"held.{r}.drop_u"] for r in S["heldout"])
    V["held.drop_d_min"] = min(V[f"held.{r}.drop_d"] for r in S["heldout"]); V["held.drop_d_max"] = max(V[f"held.{r}.drop_d"] for r in S["heldout"])


def build_perception_arm():
    """Depth sensor against the blind control, both with edge-focused starts (S78); g12_dr175 as reference."""
    arms = [("g15_edge_depth_s0", "depth sensor (18x14, head mount) + edge starts"), ("g15_edge_blind_s0", "blind control + edge starts"), ("g12_dr175_s0", "reference: ramped recipe, no edge starts"), ("g12_dr175_s1", "reference: ramped recipe, no edge starts")]
    rows = []
    for run, lab in arms:
        g, gs, v, f = S["goal"][run], S["goalscale"][run], S["vel_tiles"][run], S["flat"][run]
        rows.append([f"{lab}<br><small>{run}, seed {run[-1]}</small>", *[f"{gs[sc]['u']['success']:.2f} / {gs[sc]['d']['success']:.2f}" for sc in SCALES], f"{g['u']['median_time_s']:.2f}",
                     *[f"{g['ku'][k]:.2f}" for k in ("step_up", "stairs", "bump")], f"{v['fall_u']:.2f} / {v['fall_d']:.2f}", f"{f['0.9']['err']:.3f}"])
    head = ["Policy", "Goal success DR 1: undisturbed / disturbed", "DR 1.75", "DR 2.5", "Median time [s]", "step up", "stairs", "bump", "Tile velocity falls: undisturbed / disturbed", "Flat error at 0.9 m/s"]
    cap = setup_cap(engine=ENG, dr="on (flat error column: off)", noise="on (flat error column: off)", worlds="176 per terrain kind (1760 per cell); 64 for the flat column",
                    seeds="one seed per arm (seed 0), the reference rows are the two seeds of the ramped recipe", protocol="goal task of Table 2 at DR scales 1, 1.75 and 2.5; the per-kind columns are undisturbed success at scale 1; tile velocity task at 0.3 m/s; both arms are the ramped recipe plus --edge-focus 0.5 and differ only by the depth sensor.")
    inject("generalist.html", "tab_percarm", html_table(head, rows) + f'<p class="cap"><b>Table 27.</b> Depth sensor against blind control on the corrected engine. {cap}</p>')
    tex_table("percarm", "Depth sensor against blind control, both with edge-focused starts, corrected engine. " + tex_escape(cap), "tab:gen_percarm", head, rows, colspec="lccccccccc", star=True)
    D, B = "g15_edge_depth_s0", "g15_edge_blind_s0"
    LG = {D: ((0.60, 0.53, 0.45), (0.29, 0.26, 0.18), 2.36, (0.56, 0.73), (0.020, 0.032, 0.057)), B: ((0.60, 0.54, 0.47), (0.28, 0.23, 0.19), 2.30, (0.53, 0.73), (0.021, 0.035, 0.056))}
    for run, (u, d, t, v, fe) in LG.items():
        for k, sc in enumerate(SCALES):
            check("S78", f"Table 13 {run}", f"goal undisturbed DR {sc}", S["goalscale"][run][sc]["u"]["success"], u[k], 0.006)
            check("S78", f"Table 11 {run}", f"goal disturbed DR {sc}", S["goalscale"][run][sc]["d"]["success"], d[k], 0.006)
        check("S78", f"Table 11 {run}", "median time", S["goal"][run]["u"]["median_time_s"], t, 0.06)
        check("S78", f"Table 11 {run}", "vel falls undisturbed", S["vel_tiles"][run]["fall_u"], v[0], 0.006)
        check("S78", f"Table 11 {run}", "vel falls disturbed", S["vel_tiles"][run]["fall_d"], v[1], 0.006)
    dif = [abs(S["goalscale"][D][sc][k]["success"] - S["goalscale"][B][sc][k]["success"]) for sc in SCALES for k in ("u", "d")]
    V["percarm.maxdiff"] = max(dif)
    V["percarm.vel_diff"] = S["vel_tiles"][D]["fall_u"] - S["vel_tiles"][B]["fall_u"]


R_BALL, R_ARR = 0.09, 0.008


def clearance(phi_deg):
    """Static height of the bottom of an arrestor sphere above the floor, robot upright: R + (R + r_a) cos(phi) - r_a."""
    return R_BALL + (R_BALL + R_ARR) * math.cos(math.radians(phi_deg)) - R_ARR


def build_retainer():
    D0 = WP / "g12_dr175_s0"
    fc = lambda n: json.load(open(D0 / n))
    # ---- fall diagnosis
    rows = []
    S["fallcause"] = {}
    for key, fn, lab in (("u", "fallcause_ps0.json", "undisturbed"), ("d", "fallcause_ps1.json", "pushes + pulls"), ("lqr", "fallcause_ps1_lqronly.json", "pushes + pulls, LQR alone (no residual)")):
        d = fc(fn)
        S["fallcause"][key] = d
        tu = d.get("tilt_at_unseat_deg", {}).get("median")
        rows.append([lab, f"{d['fall_rate']:.2f}", f"{d['terminated_by']['offset_only']:.2f}", f"{d['first_event']['unseated_first']:.2f}", (f"{tu:.1f}" if tu else "not recorded"),
                     f"{100 * d['at_fall']['sat_frac_last_1s_mean']:.1f} %", f"{d['at_fall']['pull_active_frac']:.2f}"])
        V[f"fc.{key}.fall"] = d["fall_rate"]; V[f"fc.{key}.offset"] = d["terminated_by"]["offset_only"]; V[f"fc.{key}.unseat"] = d["first_event"]["unseated_first"]
        V[f"fc.{key}.sat"] = 100 * d["at_fall"]["sat_frac_last_1s_mean"]; V[f"fc.{key}.pull"] = d["at_fall"]["pull_active_frac"]
        if tu:
            V[f"fc.{key}.tilt"] = tu
    V["fc.u.surv"], V["fc.u.surv90"] = S["fallcause"]["u"]["survivors_max_tilt_median"], S["fallcause"]["u"]["survivors_max_tilt_p90"]
    V["fc.sat_max"] = max(V[f"fc.{k}.sat"] for k in ("u", "d", "lqr"))
    head = ["Condition", "Fall rate", "Falls ended by the ball-offset limit (tilt below 70 deg)", "Body derailed before tipping", "Median tilt when derailing [deg]", "Wheels at the torque limit, last 1 s before the fall", "Falls with a pull active"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="1760 (16 per design and terrain kind)", seeds="one policy (g12_dr175_s0, seed 0)",
                    protocol="goal task of Table 2 with per-step logging (fall_cause.py); derailed = the base axis is misaligned with the ball-centre-to-base direction by more than 5 deg (0.1 to 0.3 deg while the wheels sit on the ball).")
    inject("generalist.html", "tab_fallcause", html_table(head, rows) + f'<p class="cap"><b>Table 6.</b> Why the robot falls. {cap}</p>')
    tex_table("fallcause", "Fall diagnosis, corrected engine. " + tex_escape(cap), "tab:gen_fallcause", head, rows, colspec="lcccccc", star=True)
    for key, (f, off, un, tu, sat) in {"u": (0.43, 0.95, 0.78, 14.7, 0.7), "d": (0.72, 0.97, 0.49, 15.4, 0.4), "lqr": (0.85, 0.98, None, None, 0.4)}.items():
        check("S79", "Table 6", f"{key} fall rate", V[f"fc.{key}.fall"], f, 0.006)
        check("S79", "Table 6", f"{key} offset-ended share", V[f"fc.{key}.offset"], off, 0.006)
        check("S79", "Table 6", f"{key} saturation percent", V[f"fc.{key}.sat"], sat, 0.06)
        if un is not None:
            check("S79", "Table 6", f"{key} derailed first", V[f"fc.{key}.unseat"], un, 0.006)
            check("S79", "Table 6", f"{key} median tilt at derail", V[f"fc.{key}.tilt"], tu, 0.06)
    check("S79", "text", "pulls active in disturbed falls", V["fc.d.pull"], 0.58, 0.006)
    # ---- mount-angle sweep (zero-shot)
    sweep = [("none", "fallcause_ps0.json", "fallcause_ps1.json", None), ("135 (opposite the wheel)", "fallcause_ps0_ret0-0-0-0.json", "fallcause_ps1_ret0-0-0-0.json", 135),
             ("120", "fallcause_ps0_ret0-0-0-120.json", "fallcause_ps1_ret0-0-0-120.json", 120), ("110", "fallcause_ps0_ret0-0-0-110.json", "fallcause_ps1_ret0-0-0-110.json", 110),
             ("100", "fallcause_ps0_ret0-0-0-100.json", "fallcause_ps1_ret0-0-0-100.json", 100), ("95", "fallcause_ps0_ret0-0-0-95.json", "fallcause_ps1_ret0-0-0-95.json", 95),
             ("90 (equator)", "fallcause_ps0_ret0-0-0-90.json", "fallcause_ps1_ret0-0-0-90.json", 90)]
    touch = {100: "57&dagger;", 120: "n/a", 135: "21&dagger;"}
    rows, S["sweep"] = [], []
    for lab, fu, fd, phi in sweep:
        u, d = fc(fu), fc(fd)
        S["sweep"].append(dict(label=lab, phi=phi, u=u["success"], d=d["success"]))
        cl = f"{100 * clearance(phi):.1f}" if phi else "n/a"
        rows.append([lab, f"{u['success']:.2f}", f"{d['success']:.2f}", f"{u['at_fall']['mis_median']:.1f} / {d['at_fall']['mis_median']:.1f}", cl, touch.get(phi, "n/a") if phi else "n/a"])
        if phi:
            V[f"sw.{phi}.u"], V[f"sw.{phi}.d"], V[f"sw.{phi}.clear"] = u["success"], d["success"], 100 * clearance(phi)
        else:
            V["sw.none.u"], V["sw.none.d"] = u["success"], d["success"]
    V["sw.gain_u"], V["sw.gain_d"] = V["sw.100.u"] - V["sw.none.u"], V["sw.100.d"] - V["sw.none.d"]
    head = ["Arrestor mount (polar angle from the top of the ball, deg)", "Goal success undisturbed", "Goal success pushes + pulls", "Median misalignment at fall, undisturbed / disturbed [deg]", "Arrestor bottom above the floor, upright [cm]", "Tilt at which it touches the floor [deg]"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="1760 per cell (176 per terrain kind)", seeds="one policy (g12_dr175_s0, seed 0), trained without arrestors (zero-shot)",
                    protocol="goal task of Table 2 with frictionless arrestors (radius 8 mm) added at evaluation; floor clearance from the geometry R + (R + r_a) cos(phi) - r_a; the floor-touch tilts marked with a dagger are from the study log (S79), and arrestor-floor contact is not simulated.")
    inject("generalist.html", "tab_sweep", html_table(head, rows) + f'<p class="cap"><b>Table 8.</b> Arrestor mount angle, zero-shot. {cap}</p>')
    tex_table("sweep", "Arrestor mount-angle sweep (zero-shot), corrected engine. " + tex_escape(cap), "tab:gen_sweep", head, rows, colspec="lccccc", star=True)
    for lab, u, d in (("none", 0.57, 0.28), ("135", 0.66, 0.37), ("120", 0.69, 0.39), ("110", 0.72, 0.40), ("100", 0.75, 0.45), ("95", 0.75, 0.43), ("90", 0.76, 0.44)):
        k = "none" if lab == "none" else lab
        check("S79", "Table 8", f"{lab} success undisturbed", V[f"sw.{k}.u"], u, 0.006)
        check("S79", "Table 8", f"{lab} success disturbed", V[f"sw.{k}.d"], d, 0.006)
    for phi, cm in ((100, 6.5), (120, 3.3), (135, 1.2)):
        check("S79", "Table 8", f"clearance at {phi} deg [cm]", V[f"sw.{phi}.clear"], cm, 0.08)
    # ---- roller friction
    rows = []
    for mu, fu, fd in ((0.0, "fallcause_ps0_ret0-0-0-100.json", "fallcause_ps1_ret0-0-0-100.json"), (0.05, "fallcause_ps0_ret0-0.05-0-100.json", "fallcause_ps1_ret0-0.05-0-100.json"), (0.2, "fallcause_ps0_ret0-0.2-0-100.json", "fallcause_ps1_ret0-0.2-0-100.json")):
        u, d = fc(fu), fc(fd)
        rows.append([f"{mu:g}", f"{u['success']:.2f}", f"{d['success']:.2f}"])
        V[f"fr.{mu:g}.u"], V[f"fr.{mu:g}.d"] = u["success"], d["success"]
    head = ["Arrestor sliding friction", "Goal success undisturbed", "Goal success pushes + pulls"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="1760 per cell", seeds="one policy (g12_dr175_s0, seed 0), zero-shot", protocol="goal task of Table 2, arrestors at 100 deg with the stated sliding friction (0 = ideal ball-transfer unit).")
    inject("generalist.html", "tab_friction", html_table(head, rows) + f'<p class="cap"><b>Table 9.</b> Arrestor roller friction. {cap}</p>')
    tex_table("friction", "Arrestor roller friction (zero-shot). " + tex_escape(cap), "tab:gen_friction", head, rows, colspec="lcc")
    for mu, u, d in ((0, 0.75, 0.45), (0.05, 0.74, 0.42), (0.2, 0.76, 0.42)):
        check("S79", "Table 9", f"friction {mu} undisturbed", V[f"fr.{mu:g}.u"], u, 0.006)
        check("S79", "Table 9", f"friction {mu} disturbed", V[f"fr.{mu:g}.d"], d, 0.006)
    # ---- training with the retainer (S80)
    zs = {sp: json.load(open(D0 / f"eval2_vel_{sp}_ret0-0-0-100_flat_noexit.json"))["by_kind"]["flat"] for sp in SP}
    zv_u = json.load(open(D0 / "eval2_vel_0.3_dr_surf_ps0_ret0-0-0-100.json"))["overall"]["fall"]
    zv_d = json.load(open(D0 / "eval2_vel_0.3_dr_surf_pull_ret0-0-0-100.json"))["overall"]["fall"]
    V["zs.velu"], V["zs.veld"] = zv_u, zv_d
    for sp in SP:
        V[f"zs.flat.{sp}.err"], V[f"zs.flat.{sp}.ratio"] = zs[sp]["late_err"], zs[sp]["late_speed_ratio"]
    V["zs.flat.0.9.err"] = zs["0.9"]["late_err"]
    rows = []
    def ach(run):
        return f"{min(S['flat'][run][sp]['ratio'] for sp in SP):.2f} to {max(S['flat'][run][sp]['ratio'] for sp in SP):.2f}"
    for run, lab in (("g12_dr175_s0", "ramped recipe, seed 0"), ("g12_dr175_s1", "ramped recipe, seed 1")):
        gs = S["goalscale"][run]
        rows.append([f"{lab}<br><small>{run}</small>", "none", " / ".join(f"{gs[sc]['u']['success']:.2f}" for sc in SCALES), " / ".join(f"{gs[sc]['d']['success']:.2f}" for sc in SCALES),
                     f"{S['vel_tiles'][run]['fall_u']:.2f} / {S['vel_tiles'][run]['fall_d']:.2f}", " / ".join(f"{S['flat'][run][sp]['err']:.3f}" for sp in SP), ach(run)])
    rows.append(["ramped recipe, seed 0<br><small>g12_dr175_s0</small>", "added at evaluation (zero-shot)", f"{V['sw.100.u']:.2f} (scale 1 only)", f"{V['sw.100.d']:.2f} (scale 1 only)",
                 f"{zv_u:.2f} / {zv_d:.2f}", " / ".join(f"{zs[sp]['late_err']:.3f}" for sp in SP), f"{min(zs[sp]['late_speed_ratio'] for sp in SP):.2f} to {max(zs[sp]['late_speed_ratio'] for sp in SP):.2f}"])
    for run, lab, arr in (("g16_ret100_s0", "trained with the retainer, seed 0", "trained with"), ("g16_ret100_s1", "trained with the retainer, seed 1", "trained with"), ("g12_pure_s0", "pure learning, seed 0", "none"), ("g12_pure_s1", "pure learning, seed 1", "none")):
        gs = S["goalscale"][run]
        rows.append([f"{lab}<br><small>{run}</small>", arr, " / ".join(f"{gs[sc]['u']['success']:.2f}" for sc in SCALES), " / ".join(f"{gs[sc]['d']['success']:.2f}" for sc in SCALES),
                     f"{S['vel_tiles'][run]['fall_u']:.2f} / {S['vel_tiles'][run]['fall_d']:.2f}", " / ".join(f"{S['flat'][run][sp]['err']:.3f}" for sp in SP), ach(run)])
    head = ["Policy", "Arrestors", "Goal success undisturbed, DR 1 / 1.75 / 2.5", "Goal success pushes + pulls, DR 1 / 1.75 / 2.5", "Tile velocity falls undisturbed / disturbed (DR 1)", "Flat error at 0.3 / 0.6 / 0.9 [m/s]", "Flat achieved / command"]
    cap = setup_cap(engine=ENG, dr="on (flat columns: off)", noise="on (flat columns: off)", worlds="1760 per goal and tile cell (176 per terrain kind); 64 per flat cell", seeds="as listed (two seeds for the trained-with-retainer, ramped and pure rows)",
                    protocol="eval_battery.py; the retainer rows use the 100 deg arrestors in training and evaluation (RETAINER=0,0,0,100), the zero-shot row adds them only at evaluation; its flat row was not part of the S92 re-run, so its flat falls are measured on the old tile positions and overstated (error and achieved speed unaffected).")
    inject("generalist.html", "tab_rettrain", html_table(head, rows) + f'<p class="cap"><b>Table 10.</b> Training with the ball retainer. {cap}</p>')
    tex_table("rettrain", "Training with the ball retainer, corrected engine. " + tex_escape(cap), "tab:gen_rettrain", head, rows, colspec="llccccc", star=True)
    LR = {"g16_ret100_s0": ((0.71, 0.70, 0.62), (0.46, 0.42, 0.35), (0.31, 0.49), (0.06, 0.15, 0.19)), "g16_ret100_s1": ((0.76, 0.73, 0.67), (0.49, 0.44, 0.37), (0.32, 0.50), (0.05, 0.11, 0.12)),
          "g12_pure_s1": ((0.51, 0.44, 0.39), (0.27, 0.23, 0.19), (0.38, 0.57), (0.11, 0.25, 0.42))}
    for run, (u, d, v, fe) in LR.items():
        for k, sc in enumerate(SCALES):
            check("S80", f"Table 10 {run}", f"goal undisturbed DR {sc}", S["goalscale"][run][sc]["u"]["success"], u[k], 0.006)
            check("S80", f"Table 10 {run}", f"goal disturbed DR {sc}", S["goalscale"][run][sc]["d"]["success"], d[k], 0.006)
        check("S80", f"Table 10 {run}", "vel falls undisturbed", S["vel_tiles"][run]["fall_u"], v[0], 0.006)
        check("S80", f"Table 10 {run}", "vel falls disturbed", S["vel_tiles"][run]["fall_d"], v[1], 0.006)
    check("S80", "Table 10 zero-shot", "vel falls undisturbed", zv_u, 0.36, 0.006)
    check("S80", "Table 10 zero-shot", "vel falls disturbed", zv_d, 0.53, 0.006)
    rr = [S["flat"][r][sp]["ratio"] for r in ("g16_ret100_s0", "g16_ret100_s1") for sp in SP]
    V["ret.ach_min"], V["ret.ach_max"] = min(rr), max(rr)
    rr = [S["flat"][r][sp]["ratio"] for r in ("g12_dr175_s0", "g12_dr175_s1") for sp in SP]
    V["dr.ach_min"], V["dr.ach_max"] = min(rr), max(rr)
    rr = [S["flat"][r][sp]["ratio"] for r in ("g12_pure_s0", "g12_pure_s1") for sp in SP]
    V["pure.ach_min"], V["pure.ach_max"] = min(rr), max(rr)
    zr = [zs[sp]["late_speed_ratio"] for sp in SP]
    V["zs.ach_min"], V["zs.ach_max"] = min(zr), max(zr)


def build_tradeoff():
    rows = []
    spec = [("g12_dr175_s0", "no arrestors (reference)"), ("g12_dr175_s1", "no arrestors (reference)"), ("ZS", "authority 0.5, arrestors added at evaluation only"),
            ("g18_ret_rs015_s0", "authority 0.15"), ("g18_ret_rs025_s0", "authority 0.25 (then-recommended)"), ("g18_ret_rs025_s1", "authority 0.25 (then-recommended)"), ("g18_ret_rs025_s2", "authority 0.25 (then-recommended)"), ("g19_ret_rs025_400m_s0", "authority 0.25, 400M steps"), ("g18_ret_rs035_s0", "authority 0.35"),
            ("g16_ret100_s0", "authority 0.5"), ("g16_ret100_s1", "authority 0.5"), ("g17_ret100_trk3_s0", "authority 0.5, tracking weight x3"), ("g18_ret_soft_s0", "soft LQR base, authority 0.5"),
            ("g16_pure_ret100_s0", "pure learning (no LQR)"), ("g16_pure_ret100_s1", "pure learning (no LQR)")]
    S["tradeoff"] = []
    for run, lab in spec:
        if run == "ZS":
            ach = [V[f"zs.flat.{sp}.ratio"] for sp in SP]
            rows.append([f"{lab}<br><small>g12_dr175_s0, seed 0</small>", f"{min(ach):.2f} to {max(ach):.2f}", f"{V['zs.flat.0.9.err']:.3f}", f"{V['sw.100.u']:.2f} (scale 1 only)", f"{V['sw.100.d']:.2f} (scale 1 only)", f"{V['zs.velu']:.2f} / {V['zs.veld']:.2f}", "n/a"])
            S["tradeoff"].append(dict(run="zero-shot", label=lab, ach=float(np.mean(ach)), d=V["sw.100.d"], kind="zs"))
            continue
        if run not in S["flat"]:
            rows.append([f"{lab}<br><small>{run}</small>"] + [PEND] * 6)
            continue
        f, gs, v = S["flat"][run], S["goalscale"][run], S["vel_tiles"][run]
        ach = [f[sp]["ratio"] for sp in SP]
        rows.append([f"{lab}<br><small>{run}, seed {run[-1]}</small>", f"{min(ach):.2f} to {max(ach):.2f}", f"{f['0.9']['err']:.3f}", " / ".join(f"{gs[sc]['u']['success']:.2f}" for sc in SCALES),
                     " / ".join(f"{gs[sc]['d']['success']:.2f}" for sc in SCALES), f"{v['fall_u']:.2f} / {v['fall_d']:.2f}", f"{S['goal'][run]['u']['median_time_s']:.2f}"])
        S["tradeoff"].append(dict(run=run, label=lab, ach=float(np.mean(ach)), d=gs["1"]["d"]["success"], kind="ref" if run.startswith("g12") else ("pure" if "pure" in run else "res")))
    head = ["Controller", "Flat achieved / command (0.3 to 0.9 m/s)", "Flat error at 0.9 m/s [m/s]", "Goal success undisturbed, DR 1 / 1.75 / 2.5", "Goal success pushes + pulls, DR 1 / 1.75 / 2.5", "Tile velocity falls undisturbed / disturbed (DR 1)", "Median goal time [s]"]
    cap = setup_cap(engine=ENG, dr="on (flat columns: off)", noise="on (flat columns: off)", worlds="1760 per goal and tile cell (176 per terrain kind); 64 per flat cell", seeds="one per row as listed; authority 0.25 has three seeds (plus a 400M-step run), authority 0.5 and pure learning two",
                    protocol="eval_battery.py; every row except the two references is trained and evaluated with the 100 deg arrestors; the zero-shot row adds them only at evaluation; its flat row was not part of the S92 re-run, so its flat falls are measured on the old tile positions and overstated (error and achieved speed unaffected).")
    inject("generalist.html", "tab_tradeoff", html_table(head, rows) + f'<p class="cap"><b>Table 11.</b> Tracking accuracy against robustness with the retainer. {cap}</p>')
    tex_table("tradeoff", "Tracking accuracy against robustness with the retainer, corrected engine. " + tex_escape(cap), "tab:gen_tradeoff", head, rows, colspec="lcccccc", star=True)
    # retainer held-out terrain
    rows = []
    S["retheld"] = {}
    for run, lab in (("g18_ret_rs025_s0", "then-recommended: LQR + residual 0.25, trained with the retainer"), ("g18_ret_rs025_s1", "then-recommended: LQR + residual 0.25, trained with the retainer"), ("g16_ret100_s0", "LQR + residual 0.5, trained with the retainer"), ("g16_ret100_s1", "LQR + residual 0.5, trained with the retainer"),
                     ("g16_pure_ret100_s0", "pure learning, trained with the retainer"), ("g16_pure_ret100_s1", "pure learning, trained with the retainer"), ("g12_dr175_s0", "no arrestors (reference)"), ("g12_dr175_s1", "no arrestors (reference)")):
        du, dd = json.load(open(WP / run / "eval2_goal_goal_dr_surf_ps0_ts1.json"))["overall"], json.load(open(WP / run / "eval2_goal_goal_dr_surf_pull_ts1.json"))["overall"]
        S["retheld"][run] = (du["success"], dd["success"])
        V[f"rh.{run}.u"], V[f"rh.{run}.d"] = du["success"], dd["success"]
        rows.append([f"{lab}<br><small>{run}, seed {run[-1]}</small>", f"{du['success']:.2f}", f"{S['goalscale'][run]['1']['u']['success']:.2f}", f"{dd['success']:.2f}", f"{S['goalscale'][run]['1']['d']['success']:.2f}"])
    head = ["Policy", "Undisturbed success, held-out map (tile seed 1)", "trained map (tile seed 0)", "Pushes + pulls success, held-out map", "trained map"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="176 per terrain kind, 1760 per cell", seeds="as listed", protocol="goal task of Table 2 on a freshly generated tile map (tile seed 1); policies were trained on tile seed 0; the retainer rows run with the 100 deg arrestors.")
    inject("generalist.html", "tab_retheld", html_table(head, rows) + f'<p class="cap"><b>Table 12.</b> The retainer on held-out terrain. {cap}</p>')
    tex_table("retheld", "The retainer on held-out terrain (tile seed 1), corrected engine. " + tex_escape(cap), "tab:gen_retheld", head, rows, colspec="lcccc", star=True)
    for run, (u, d) in {"g18_ret_rs025_s0": (0.66, 0.38), "g18_ret_rs025_s1": (0.69, 0.39), "g16_ret100_s0": (0.66, 0.42), "g16_ret100_s1": (0.68, 0.43), "g16_pure_ret100_s0": (0.66, 0.46), "g16_pure_ret100_s1": (0.71, 0.47)}.items():
        check("S80/S82", "Table 12", f"{run} held-out undisturbed", V[f"rh.{run}.u"], u, 0.006)
        check("S80/S82", "Table 12", f"{run} held-out disturbed", V[f"rh.{run}.d"], d, 0.006)
    # checks S81 to S83
    L = {"g16_pure_ret100_s0": ((0.70, 0.67, 0.64), (0.50, 0.45, 0.40), (0.16, 0.34)), "g17_ret100_trk3_s0": ((0.71, 0.68, 0.62), (0.49, 0.43, 0.36), (0.32, 0.46)),
         "g18_ret_rs025_s0": ((0.71, 0.68, 0.60), (0.42, 0.36, 0.29), (0.34, 0.55)), "g18_ret_soft_s0": ((0.75, 0.75, 0.66), (0.51, 0.47, 0.39), (0.29, 0.49)),
         "g18_ret_rs015_s0": ((0.69, 0.65, 0.56), (0.35, 0.31, 0.22), None), "g18_ret_rs025_s1": ((0.76, 0.72, 0.66), (0.43, 0.39, 0.32), None), "g18_ret_rs035_s0": ((0.72, 0.71, 0.63), (0.45, 0.40, 0.32), None)}
    for run, (u, d, v) in L.items():
        if run not in S["goalscale"]:
            continue
        for k, sc in enumerate(SCALES):
            check("S81-S83", f"Table 11 {run}", f"goal undisturbed DR {sc}", S["goalscale"][run][sc]["u"]["success"], u[k], 0.006)
            check("S81-S83", f"Table 11 {run}", f"goal disturbed DR {sc}", S["goalscale"][run][sc]["d"]["success"], d[k], 0.006)
        if v:
            check("S81-S83", f"Table 11 {run}", "vel falls undisturbed", S["vel_tiles"][run]["fall_u"], v[0], 0.006)
            check("S81-S83", f"Table 11 {run}", "vel falls disturbed", S["vel_tiles"][run]["fall_d"], v[1], 0.006)


def build_s84_s85():
    # three seeds, 400M, depth on top
    L = {"g18_ret_rs025_s2": ((0.76, 0.71, 0.63), (0.44, 0.41, 0.32)), "g20_ret_rs025_depth_s0": ((0.74, 0.69, 0.63), (0.43, 0.38, 0.31)), "g19_ret_rs025_400m_s0": ((0.75, 0.70, 0.64), (0.43, 0.36, 0.30))}
    for run, (u, d) in L.items():
        for k, sc in enumerate(SCALES):
            check("S84", f"tables {run}", f"goal undisturbed DR {sc}", S["goalscale"][run][sc]["u"]["success"], u[k], 0.006)
            check("S84", f"tables {run}", f"goal disturbed DR {sc}", S["goalscale"][run][sc]["d"]["success"], d[k], 0.006)
    check("S84", "text", "400M vel falls undisturbed", S["vel_tiles"]["g19_ret_rs025_400m_s0"]["fall_u"], 0.38, 0.006)
    check("S84", "text", "depth vel falls undisturbed", S["vel_tiles"]["g20_ret_rs025_depth_s0"]["fall_u"], 0.35, 0.006)
    # per-obstacle falls (velocity task, undisturbed)
    runs = [("g18_ret_rs025_s0", "then-recommended, seed 0"), ("g18_ret_rs025_s1", "then-recommended, seed 1"), ("g20_ret_rs025_depth_s0", "then-recommended + depth sensor, seed 0")]
    pk = {r: json.load(open(WP / r / "eval2_vel_0.3_dr_surf_ps0.json"))["by_kind"] for r, _ in runs}
    order = sorted(KINDS, key=lambda k: pk["g18_ret_rs025_s0"][k]["fall"])
    rows = [[KIND_LABEL[k]] + [f"{pk[r][k]['fall']:.2f}" for r, _ in runs] for k in order]
    for k in KINDS:
        V[f"obs.{k}"] = float(np.mean([pk[r][k]["fall"] for r in ("g18_ret_rs025_s0", "g18_ret_rs025_s1")]))
    head = ["Terrain kind"] + [f"{lab} ({r})" for r, lab in runs]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="176 per terrain kind", seeds="as in the column headers", protocol="velocity task at 0.3 m/s on the tiles, undisturbed (pushes and pulls off), 100 deg arrestors; fall rate per terrain kind, sorted by the seed-0 value.")
    inject("generalist.html", "tab_obstacles", html_table(head, rows) + f'<p class="cap"><b>Table 13.</b> Where the then-recommended controller still falls. {cap}</p>')
    tex_table("obstacles", "Where the then-recommended controller still falls (velocity task, undisturbed). " + tex_escape(cap), "tab:gen_obstacles", head, rows, colspec="lccc")
    for k, (a0, a1, dd) in {"flat": (0.08, 0.07, 0.08), "step_up": (0.61, 0.62, 0.55), "large_drop": (0.64, 0.66, 0.68), "bump": (0.77, 0.74, 0.75), "stairs": (0.48, 0.52, 0.51), "ramp_up": (0.12, 0.14, None), "ramp_down": (0.20, 0.26, None), "rough": (0.17, 0.16, None), "step_down": (0.16, 0.16, None), "small_drop": (0.18, 0.17, None)}.items():
        check("S83/S84", "Table 13", f"{k} seed 0", pk["g18_ret_rs025_s0"][k]["fall"], a0, 0.006)
        check("S83/S84", "Table 13", f"{k} seed 1", pk["g18_ret_rs025_s1"][k]["fall"], a1, 0.006)
        if dd is not None:
            check("S84", "Table 13", f"{k} depth", pk["g20_ret_rs025_depth_s0"][k]["fall"], dd, 0.006)
    # depth on top (perception chapter)
    rows = []
    for run, lab in (("g20_ret_rs025_depth_s0", "then-recommended + depth sensor (18x14) + edge starts"), ("g18_ret_rs025_s0", "then-recommended, blind"), ("g18_ret_rs025_s1", "then-recommended, blind"), ("g18_ret_rs025_s2", "then-recommended, blind")):
        gs, v = S["goalscale"][run], S["vel_tiles"][run]
        ach = [S["flat"][run][sp]["ratio"] for sp in SP]
        rows.append([f"{lab}<br><small>{run}, seed {run[-1]}</small>", " / ".join(f"{gs[sc]['u']['success']:.2f}" for sc in SCALES), " / ".join(f"{gs[sc]['d']['success']:.2f}" for sc in SCALES),
                     f"{v['fall_u']:.2f} / {v['fall_d']:.2f}", f"{pk[run]['step_up']['fall']:.2f}" if run in pk else "n/a", f"{min(ach):.2f} to {max(ach):.2f}"])
    head = ["Policy", "Goal success undisturbed, DR 1 / 1.75 / 2.5", "Goal success pushes + pulls, DR 1 / 1.75 / 2.5", "Tile velocity falls undisturbed / disturbed (DR 1)", "Step-up falls (velocity task)", "Flat achieved / command"]
    cap = setup_cap(engine=ENG, dr="on (flat column: off)", noise="on (flat column: off)", worlds="176 per terrain kind (1760 per goal or tile cell); 64 per flat cell", seeds="one seed for the depth arm, three for the blind then-recommended controller",
                    protocol="eval_battery.py with the 100 deg arrestors in training and evaluation; the depth arm also uses edge-focused starts.")
    inject("generalist.html", "tab_percarm2", html_table(head, rows) + f'<p class="cap"><b>Table 28.</b> Depth sensor on top of the then-recommended controller. {cap}</p>')
    tex_table("percarm2", "Depth sensor on top of the then-recommended controller (retainer), corrected engine. " + tex_escape(cap), "tab:gen_percarm2", head, rows, colspec="lccccc", star=True)
    V["dep.stepup_gain"] = float(np.mean([pk[r]["step_up"]["fall"] for r in ("g18_ret_rs025_s0", "g18_ret_rs025_s1")]) - pk["g20_ret_rs025_depth_s0"]["step_up"]["fall"])
    # held-out designs (S85-R: re-run after the design-vector fix)
    rows = []
    S["designs_ho"] = {}
    spec = [("g18_ret_rs025_s1", "then-recommended: LQR + residual 0.25, retainer", "no"), ("g12_dr175_s1", "LQR + residual 0.5, no retainer", "no"), ("g16_pure_ret100_s1", "pure learning, retainer", "no"),
            ("g21_ret_rs025_tau3_s0", "LQR + residual 0.25, retainer, 12 designs", "yes"), ("g21_ret_rs025_tau3_s1", "LQR + residual 0.25, retainer, 12 designs", "yes")]
    for run, lab, tr in spec:
        du = json.load(open(WP / run / "eval2_goal_goal_dr_surf_ps0_des-s1.5-tau3.json"))
        dd = json.load(open(WP / run / "eval2_goal_goal_dr_surf_pull_des-s1.5-tau3.json"))
        u, d = du["by_design"], dd["by_design"]
        S["designs_ho"][run] = dict(u={k: u[k]["success"] for k in u}, d={k: d[k]["success"] for k in d})
        for k in ("s1.5", "tau3"):
            V[f"ho.{run}.u.{k}"], V[f"ho.{run}.d.{k}"] = u[k]["success"], d[k]["success"]
        rows.append([f"{lab}<br><small>{run}, seed {run[-1]}</small>", tr, f"{u['s1.5']['success']:.2f}", f"{u['tau3']['success']:.2f}", f"{d['s1.5']['success']:.2f}", f"{d['tau3']['success']:.2f}"])
    lqr = [V[f"ho.{r}.{k}.s1.5"] for r in ("g18_ret_rs025_s1", "g12_dr175_s1") for k in ("u", "d")]
    V["ho.lqr_min"], V["ho.lqr_max"] = min(lqr), max(lqr)
    V["ho.pure_min"], V["ho.pure_max"] = min(V["ho.g16_pure_ret100_s1.u.s1.5"], V["ho.g16_pure_ret100_s1.d.s1.5"]), max(V["ho.g16_pure_ret100_s1.u.s1.5"], V["ho.g16_pure_ret100_s1.d.s1.5"])
    V["ho.tau3_max"] = max(V[f"ho.{r}.u.tau3"] for r, _, _ in spec)
    head = ["Policy", "Trained on tau3", "Undisturbed: s1.5 (larger robot, unseen)", "tau3 (one tenth torque)", "Pushes + pulls: s1.5", "tau3"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="160 per design (16 per terrain kind)", seeds="one per row as listed (two for the 12-design runs)",
                    protocol="goal task of Table 2 on two designs outside the 11 training designs: s1.5 (robot scaled 1.5 times, ball radius 13.5 cm) and tau3 (wheel torque limit one tenth of nominal); the 12-design runs include tau3 in training, so for them only s1.5 is unseen.")
    inject("generalist.html", "tab_designs_ho", html_table(head, rows) + f'<p class="cap"><b>Table 24.</b> Unseen robot designs. {cap}</p>')
    tex_table("designs_ho", "Generalisation to unseen robot designs, corrected engine. " + tex_escape(cap), "tab:gen_designs_ho", head, rows, colspec="llcccc", star=True)
    LD = {"g18_ret_rs025_s1": (0.71, 0.30, 0.66, 0.15), "g12_dr175_s1": (0.69, 0.15, 0.61, 0.12), "g16_pure_ret100_s1": (0.09, 0.00, 0.15, 0.00), "g21_ret_rs025_tau3_s0": (0.89, 0.29, 0.76, 0.17), "g21_ret_rs025_tau3_s1": (0.75, 0.26, 0.66, 0.21)}
    for run, (a1, a2, b1, b2) in LD.items():
        for q, c, l in (("s1.5 undisturbed", V[f"ho.{run}.u.s1.5"], a1), ("tau3 undisturbed", V[f"ho.{run}.u.tau3"], a2), ("s1.5 disturbed", V[f"ho.{run}.d.s1.5"], b1), ("tau3 disturbed", V[f"ho.{run}.d.tau3"], b2)):
            check("S85-R", "Table 24", f"{run} {q}", c, l, 0.006)


def fig_designs_ho(C, size):
    fig, axs = plt.subplots(1, 2, figsize=size, sharey=True)
    spec = [("then-recommended\n(rs 0.25, retainer)", "g18_ret_rs025_s1", C["hybrid"]), ("rs 0.5,\nno retainer", "g12_dr175_s1", C["model"]), ("pure learning,\nretainer", "g16_pure_ret100_s1", C["learned"]),
            ("rs 0.25, 12 designs\n(s0, s1)", "MEAN12", C["warn"])]
    for ax, key, title in ((axs[0], "u", "(a) undisturbed"), (axs[1], "d", "(b) pushes + pulls")):
        for i, (lab, run, col) in enumerate(spec):
            if run == "MEAN12":
                v1 = np.mean([S["designs_ho"][r][key]["s1.5"] for r in ("g21_ret_rs025_tau3_s0", "g21_ret_rs025_tau3_s1")]); v2 = np.mean([S["designs_ho"][r][key]["tau3"] for r in ("g21_ret_rs025_tau3_s0", "g21_ret_rs025_tau3_s1")])
            else:
                v1, v2 = S["designs_ho"][run][key]["s1.5"], S["designs_ho"][run][key]["tau3"]
            ax.bar(i - 0.2, v1, 0.38, color=col, label="s1.5 (larger robot)" if i == 0 else None)
            ax.bar(i + 0.2, v2, 0.38, color=col, alpha=0.4, hatch="//", edgecolor=C["fg"], linewidth=0, label="tau3 (low torque)" if i == 0 else None)
        ax.set_xticks(range(len(spec))); ax.set_xticklabels([x[0] for x in spec], fontsize=7); ax.set_title(title); ax.set_ylim(0, 1)
    axs[0].set_ylabel("goal success on an unseen design"); axs[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def build_s87():
    D1 = WP / "g18_ret_rs025_s1"
    rows = []
    cols = ("step_up", "bump", "large_drop", "stairs", "flat")
    for sp in ("0.1", "0.2", "0.3"):
        d = json.load(open(D1 / f"speed_long_{sp}.json"))
        rows.append([f"{sp} m/s", f"{d['overall']['fall']:.2f}"] + [f"{d['by_kind'][k]['fall']:.2f}" for k in cols])
        V[f"spd.{sp}.all"] = d["overall"]["fall"]
        for k in cols:
            V[f"spd.{sp}.{k}"] = d["by_kind"][k]["fall"]
    head = ["Commanded speed", "All tiles"] + [KIND_LABEL[k] for k in cols]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="176 per terrain kind (1760 per row)", seeds="one policy (g18_ret_rs025_s1, seed 1)",
                    protocol="velocity task on the tiles, undisturbed, arrestors on, 20 s episodes so that every robot reaches the terrain feature (eval_governor.py with no governor); fall rate per kind.")
    inject("generalist.html", "tab_speedlong", html_table(head, rows) + f'<p class="cap"><b>Table 14.</b> Falls against commanded speed. {cap}</p>')
    tex_table("speedlong", "Falls against commanded speed (20\\,s episodes). " + tex_escape(cap), "tab:gen_speedlong", head, rows, colspec="lcccccc")
    LS = {"0.1": (0.42, 0.64, 0.64, 0.80, 0.55, 0.15), "0.2": (0.41, 0.66, 0.80, 0.73, 0.56, 0.08), "0.3": (0.37, 0.64, 0.74, 0.66, 0.52, 0.10)}
    for sp, vals in LS.items():
        for q, l in zip(("all",) + cols, vals):
            check("S87", "Table 14", f"{sp} m/s {q}", V[f"spd.{sp}.{q}"], l, 0.006)
    sev = json.load(open(D1 / "speed_long_0.3_sev.json"))["by_severity"]
    rows = []
    unit = {"ramp_up": "deg", "ramp_down": "deg"}
    for k in ("step_up", "bump", "stairs", "large_drop", "small_drop", "step_down", "ramp_up", "ramp_down"):
        if k not in sev:
            continue
        e = sev[k]
        m = 1 if k.startswith("ramp") else 100
        u = unit.get(k, "cm")
        rows.append([KIND_LABEL[k] + (" (riser)" if k == "stairs" else ""), f"{e['sev_min'] * m:.1f} to {e['sev_max'] * m:.1f} {u}", f"{e['split'] * m:.1f} {u}", f"{e['small']['fall']:.2f}", f"{e['large']['fall']:.2f}"])
        V[f"sev.{k}.small"], V[f"sev.{k}.large"], V[f"sev.{k}.split"] = e["small"]["fall"], e["large"]["fall"], e["split"] * m
    head = ["Feature", "Severity range on the map", "Split", "Falls, smaller half", "Falls, larger half"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="352 per kind (two tile maps' worth of worlds, split into halves by severity)", seeds="one policy (g18_ret_rs025_s1, seed 1)",
                    protocol="velocity task at 0.3 m/s on the tiles, undisturbed, arrestors on, 20 s episodes; worlds split at the median drawn severity of their tile.")
    inject("generalist.html", "tab_severity", html_table(head, rows) + f'<p class="cap"><b>Table 15.</b> Falls against feature size. {cap}</p>')
    tex_table("severity", "Falls against feature size (velocity task, 0.3\\,m/s, 20\\,s). " + tex_escape(cap), "tab:gen_severity", head, rows, colspec="lcccc")
    for k, (a0, a1) in {"step_up": (0.40, 0.95), "bump": (0.59, 0.90), "stairs": (0.27, 0.70), "large_drop": (0.61, 0.77), "small_drop": (0.23, 0.34), "step_down": (0.12, 0.21)}.items():
        check("S87", "Table 15", f"{k} smaller half", V[f"sev.{k}.small"], a0, 0.006)
        check("S87", "Table 15", f"{k} larger half", V[f"sev.{k}.large"], a1, 0.006)
    g0, g1 = json.load(open(D1 / "gov_vel03_ps0_none.json")), json.load(open(D1 / "gov_vel03_ps0_oracle.json"))
    V["gov.none"], V["gov.oracle"] = g0["overall"]["fall"], g1["overall"]["fall"]
    check("S87", "text", "governor off falls", V["gov.none"], 0.39, 0.006)
    check("S87", "text", "governor on falls", V["gov.oracle"], 0.37, 0.006)


def build_s100():
    """S100: the S87 speed test and the speed governor re-run on the corrected contact model (free-rolling friction 0.01)."""
    D1 = WP / "g18_ret_rs025_s1"
    cols = ("step_up", "bump", "large_drop", "stairs", "flat")
    rows = []
    for sp in ("0.1", "0.2", "0.3", "0.6", "0.9"):
        d = json.load(open(D1 / f"speed_mu01_{sp}.json"))
        assert d["steps"] == 2000 and d["push_scale"] == 0 and d["governor"] is None, sp
        V[f"s100.spd.{sp}.all"] = d["overall"]["fall"]
        for k in cols:
            V[f"s100.spd.{sp}.{k}"] = d["by_kind"][k]["fall"]
        old = f"{V[f'spd.{sp}.all']:.2f}" if f"spd.{sp}.all" in V else "not run"
        rows.append([f"{sp} m/s", f"{d['overall']['fall']:.2f}"] + [f"{d['by_kind'][k]['fall']:.2f}" for k in cols] + [old])
    head = ["Commanded speed", "All tiles"] + [KIND_LABEL[k] for k in cols] + ["All tiles at 0.001 (Table 14, superseded)"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01 (S98), policy trained at 0.001", dr="on at scale 1", noise="on", worlds="176 per terrain kind (1760 per row)", seeds="one policy (g18_ret_rs025_s1, seed 1)",
                    protocol="the Table 14 protocol: velocity task on the tiles, undisturbed, arrestors on, 20 s episodes so that every robot reaches its feature (eval_governor.py with no governor; files speed_mu01_*.json); fall rate per kind; the last column repeats Table 14 at 0.001.")
    inject("generalist.html", "tab_speed_mu01", html_table(head, rows) + f'<p class="cap"><b>Table 35.</b> Falls against commanded speed on the corrected contact model. {cap}</p>')
    tex_table("speed_mu01", "Falls against commanded speed on the corrected contact model (friction 0.01; 20\\,s episodes). " + tex_escape(cap), "tab:gen_speed_mu01", head, rows, colspec="lccccccc", star=True)
    LS = {"0.1": (0.28, 0.46, 0.54, 0.74, 0.43, 0.06), "0.2": (0.34, 0.61, 0.73, 0.72, 0.48, 0.05), "0.3": (0.35, 0.62, 0.74, 0.66, 0.49, 0.06), "0.6": (0.40, 0.54, 0.66, 0.77, 0.51, 0.11), "0.9": (0.51, 0.62, 0.66, 0.83, 0.62, 0.23)}
    for sp, vals in LS.items():
        for q, l in zip(("all",) + cols, vals):
            check("S100", "Table 35", f"{sp} m/s {q}", V[f"s100.spd.{sp}.{q}"], l, 0.006)
    V["s100.spd.rise"] = V["s100.spd.0.9.all"] - V["s100.spd.0.1.all"]
    V["s100.spd.gain_slow"] = V["s100.spd.0.3.all"] - V["s100.spd.0.1.all"]
    # speed layer (S100 addendum, S101 g18 rows, S102 matched pair). S102: eval_governor.py did not apply a run's own pair_mu1,
    # so the S101 g26 rows (gov_vel03_ps0_*.json) were evaluated at 0.001 and are not used; only files with an explicit 0.01 are read.
    gcols = ("step_up", "bump", "stairs", "large_drop", "flat")
    cell = lambda d: " / ".join([f"{d['overall']['fall']:.2f}"] + [f"{d['by_kind'][k]['fall']:.2f}" for k in gcols] + [f"{d['overall']['progress_m']:.2f} m"])
    rows = []
    G18 = ["g18_ret_rs025_s0", "g18_ret_rs025_s1", "g18_ret_rs025_s2"]
    for r in G18:
        dn, dg = json.load(open(WP / r / "gov_mu01_vel03_ps0_none.json")), json.load(open(WP / r / "gov_mu01_vel03_ps0_oracle_v005.json"))
        for g, d in (("none", dn), ("gov", dg)):
            assert d["steps"] == 1500 and d["push_scale"] == 0 and d["speed"] == 0.3, r
            V[f"spl.{r}.{g}.all"], V[f"spl.{r}.{g}.prog"] = d["overall"]["fall"], d["overall"]["progress_m"]
            for k in gcols:
                V[f"spl.{r}.{g}.{k}"] = d["by_kind"][k]["fall"]
        assert dg["governor"]["v_slow"] == 0.05, r
        rows.append([f"g18 seed {r[-1]} (trained at 0.001)<br><small>{r}, gov_mu01_vel03_ps0_*.json</small>", cell(dn), cell(dg)])
    for g in ("none", "gov"):
        vals = [V[f"spl.{r}.{g}.all"] for r in G18]
        V[f"spl.g18.{g}.min"], V[f"spl.g18.{g}.max"] = min(vals), max(vals)
        V[f"spl.g18.{g}.prog_mean"] = float(np.mean([V[f"spl.{r}.{g}.prog"] for r in G18]))
        for k in ("bump", "stairs", "step_up"):
            kv = [V[f"spl.{r}.{g}.{k}"] for r in G18]
            V[f"spl.g18.{g}.{k}.min"], V[f"spl.g18.{g}.{k}.max"] = min(kv), max(kv)
    V["spl.g18.prog_loss"] = 1 - V["spl.g18.gov.prog_mean"] / V["spl.g18.none.prog_mean"]
    # S102 matched pair, both evaluated with PAIR_MU1=0.01 exported (behav_mu01_*.json record pair_mu1)
    for fam, r, lab in (("g18", "g18_ret_rs025_s1", "g18 seed 1 (trained at 0.001), behaviour-comparison run"), ("g26", "g26_ret_rs025_mu01_s1", "g26 seed 1 (trained at 0.01), then-recommended checkpoint")):
        dn, dg = json.load(open(WP / r / "behav_mu01_none.json")), json.load(open(WP / r / "behav_mu01_oracle.json"))
        for g, d in (("none", dn), ("gov", dg)):
            assert d["pair_mu1"] == 0.01 and d["steps"] == 1500 and d["push_scale"] == 0 and d["speed"] == 0.3, r
            V[f"spl.pair.{fam}.{g}.all"], V[f"spl.pair.{fam}.{g}.prog"] = d["overall"]["fall"], d["overall"]["progress_m"]
            for k in gcols:
                V[f"spl.pair.{fam}.{g}.{k}"] = d["by_kind"][k]["fall"]
        assert dg["governor"]["v_slow"] == 0.05, r
        rows.append([f"{lab}<br><small>{r}, behav_mu01_*.json</small>", cell(dn), cell(dg)])
    for r in ("g26_ret_rs025_mu01_s0", "g26_ret_rs025_mu01_s2"):
        rows.append([f"g26 seed {r[-1]} (trained at 0.01)<br><small>{r}</small>", PEND + " re-run at 0.01", PEND + " re-run at 0.01"])
    V["spl.pair.diff.none"] = abs(V["spl.pair.g26.none.all"] - V["spl.pair.g18.none.all"])
    V["spl.pair.diff.gov"] = abs(V["spl.pair.g26.gov.all"] - V["spl.pair.g18.gov.all"])
    allrows = [V[f"spl.{r}.{g}.large_drop"] for r in G18 for g in ("none", "gov")] + [V[f"spl.pair.{f}.{g}.large_drop"] for f in ("g18", "g26") for g in ("none", "gov")]
    V["spl.drop_min"], V["spl.drop_max"] = min(allrows), max(allrows)
    # S103 addendum: the 40 s re-test (g26 s1 at its own friction 0.01). The 15 s rows above stay as values (the S102 family comparison) but are no longer tabulated.
    D26 = WP / "g26_ret_rs025_mu01_s1"
    kcols = ("step_up", "bump", "stairs", "large_drop", "flat")
    rows = []
    for key, f_, lab in (("none", "gov_vel03_ps0_none_40s.json", "no speed layer"), ("oracle", "gov_vel03_ps0_oracle_v005_40s.json", "perfect height map, slow to 0.05 m/s near a discontinuity (upper bound)"),
                         ("tof", "gov_vel03_ps0_tof18_v005_40s.json", "ToF sensor (18 x 14 zones) through a learned height estimator, slow to 0.05 m/s")):
        d = json.load(open(D26 / f_))
        assert d["steps"] == 4000 and d["pair_mu1"] == 0.01 and d["push_scale"] == 0 and d["speed"] == 0.3, f_
        if key != "none":
            assert d["governor"]["v_slow"] == 0.05, f_
        V[f"sl40.{key}.all"], V[f"sl40.{key}.exit"], V[f"sl40.{key}.prog"] = d["overall"]["fall"], d["overall"]["success"], d["overall"]["progress_m"]
        for k in kcols:
            V[f"sl40.{key}.{k}"] = d["by_kind"][k]["fall"]
        rows.append([lab, f"{d['overall']['fall']:.2f}"] + [f"{d['by_kind'][k]['fall']:.2f}" for k in kcols] + [f"{d['overall']['success']:.2f}", f"{d['overall']['progress_m']:.2f} m"])
    r2 = lambda k: round(V[k], 2)      # differences of the rates as printed in Table 36
    V["sl40.gain_min"] = r2("sl40.none.all") - r2("sl40.tof.all")
    V["sl40.gain_max"] = r2("sl40.none.all") - r2("sl40.oracle.all")
    V["sl40.exitcost_min"], V["sl40.exitcost_max"] = r2("sl40.none.exit") - r2("sl40.oracle.exit"), r2("sl40.none.exit") - r2("sl40.tof.exit")
    V["sl40.other_maxdiff"] = max(abs(r2(f"sl40.{k2}.{k}") - r2(f"sl40.none.{k}")) for k2 in ("oracle", "tof") for k in ("step_up", "large_drop", "flat"))
    check("S103 addendum", "text", "speed-layer effect, smallest", V["sl40.gain_min"], 0.03, 0.006)
    check("S103 addendum", "text", "speed-layer effect, largest", V["sl40.gain_max"], 0.05, 0.006)
    check("S103 addendum", "text", "cost in tile exits, smallest", V["sl40.exitcost_min"], 0.04, 0.006)
    check("S103 addendum", "text", "cost in tile exits, largest", V["sl40.exitcost_max"], 0.08, 0.006)
    head = ["Speed layer (g26 seed 1)", "Falls, all tiles"] + [KIND_LABEL[k] for k in kcols] + ["Tile exit reached", "Progress"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01 (the policy's own training value, recorded in each file)", dr="on at scale 1", noise="on", worlds="176 per terrain kind (1760 per row)", seeds="one policy (g26_ret_rs025_mu01_s1, a then-recommended checkpoint), one evaluation seed",
                    protocol="velocity task at 0.3 m/s on the tiles, undisturbed, arrestors on, 40 s episodes so that a slowed robot still reaches its feature, tile seed 0; governor.py slow mode, which lowers the commanded speed to 0.05 m/s within 0.25 m of a rise above 2.4 cm or a drop above 8 cm, read from the perfect height map or from the ToF estimate; files gov_vel03_ps0_{none,oracle_v005,tof18_v005}_40s.json. An earlier 15 s version of this test (S100 to S102) let slowed robots run out of time before their feature and is superseded.")
    inject("generalist.html", "tab_gov_mu01", html_table(head, rows) + f'<p class="cap"><b>Table 36.</b> A terrain-aware speed layer, given time to reach the feature. {cap}</p>')
    tex_table("gov_mu01", "A terrain-aware speed layer with 40\\,s episodes. " + tex_escape(cap), "tab:gen_gov_mu01", head, rows, colspec="p{0.26\\linewidth}cccccccc", star=True)
    for key, vals in {"none": (0.38, 0.62, 0.79, 0.56, 0.72, 0.11, 0.62, 1.35), "oracle": (0.33, 0.61, 0.72, 0.43, 0.74, 0.07, 0.58, 1.31), "tof": (0.35, 0.61, 0.78, 0.47, 0.77, 0.09, 0.54, 1.28)}.items():
        for q, l in zip(("all",) + kcols + ("exit", "prog"), vals):
            check("S103 addendum", "Table 36", f"40 s {key} {q}", V[f"sl40.{key}.{q}"], l, 0.006)
    # goal task, 30 s
    for key, f_ in (("none", "gov_goal_ps0_none_30s.json"), ("tof05", "gov_goal_ps0_tof18_30s.json"), ("tof10", "gov_goal_ps0_tof18_v01_30s.json")):
        d = json.load(open(D26 / f_))
        assert d["steps"] == 3000 and d["task"] == "goal" and d["pair_mu1"] == 0.01, f_
        V[f"sl30.{key}.fall"], V[f"sl30.{key}.succ"] = d["overall"]["fall"], d["overall"]["success"]
    for key, (fa, su) in {"none": (0.24, 0.76), "tof05": (0.23, 0.62), "tof10": (0.25, 0.74)}.items():
        check("S103 addendum", "text", f"goal 30 s {key} falls", V[f"sl30.{key}.fall"], fa, 0.006)
        check("S103 addendum", "text", f"goal 30 s {key} success", V[f"sl30.{key}.succ"], su, 0.006)
    L = {"g18_ret_rs025_s0": ((0.36, 0.62, 0.74, 0.49, 0.73, 0.08, 1.36), (0.19, 0.34, 0.15, 0.18, 0.72, 0.07, 1.06)), "g18_ret_rs025_s1": ((0.36, 0.64, 0.73, 0.48, 0.73, 0.05, 1.35), (0.18, 0.28, 0.18, 0.15, 0.69, 0.06, 1.07)),
         "g18_ret_rs025_s2": ((0.38, 0.62, 0.74, 0.53, 0.74, 0.10, 1.35), (0.18, 0.31, 0.14, 0.10, 0.74, 0.07, 1.03))}
    for r, (ln_, lg_) in L.items():
        for g, vals in (("none", ln_), ("gov", lg_)):
            for q, l in zip(("all", "step_up", "bump", "stairs", "large_drop", "flat", "prog"), vals):
                check("S101 (g18 rows, stand per S102)", "Table 36", f"{r} {g} {q}", V[f"spl.{r}.{g}.{q}"], l, 0.006)
    check("S101 (g18 rows)", "text", "g18 progress loss", V["spl.g18.prog_loss"], 0.22, 0.006)
    for fam, (a_, b_) in {"g18": (0.35, 0.17), "g26": (0.37, 0.16)}.items():
        check("S102", "Table 36", f"{fam} s1 matched pair, no governor", V[f"spl.pair.{fam}.none.all"], a_, 0.006)
        check("S102", "Table 36", f"{fam} s1 matched pair, governor", V[f"spl.pair.{fam}.gov.all"], b_, 0.006)
    # the single-seed 0.1 m/s governor value of the S100 addendum (g18 s1, explicit 0.01)
    d = json.load(open(D1 / "gov_mu01_vel03_ps0_oracle.json"))
    V["s100.gov.oracle.all"], V["s100.gov.oracle.prog"] = d["overall"]["fall"], d["overall"]["progress_m"]
    check("S100 addendum", "text", "g18 s1 governor slow to 0.1 m/s falls", V["s100.gov.oracle.all"], 0.26, 0.006)
    inject("generalist.html", "pending_s100", '<p>The S88 by-size test re-run at 0.01 is unchanged within 0.11 (S99 addendum), and chapter 9 already uses the corrected model.</p>')


def build_s88():
    D1 = WP / "g18_ret_rs025_s1"
    for r in (0.09, 0.135):
        V[f"trac.h.{r}"] = 100 * r * (1 - math.cos(math.radians(45)))
    check("S88", "text", "traction limit 9 cm ball [cm]", V["trac.h.0.09"], 2.6, 0.06)
    check("S88", "text", "traction limit 13.5 cm ball [cm]", V["trac.h.0.135"], 4.0, 0.06)
    # ball-size test
    nom, big = json.load(open(D1 / "stepsize_nom.json"))["by_severity"], json.load(open(D1 / "stepsize_s1.5.json"))["by_severity"]
    rows = []
    for k in ("step_up", "bump", "stairs", "large_drop"):
        m = 100
        e = nom[k]
        rows.append([KIND_LABEL[k] + (" (riser)" if k == "stairs" else ""), f"{e['sev_min'] * m:.1f} to {e['split'] * m:.1f} / {e['split'] * m:.1f} to {e['sev_max'] * m:.1f} cm",
                     f"{nom[k]['small']['fall']:.2f} / {nom[k]['large']['fall']:.2f}", f"{big[k]['small']['fall']:.2f} / {big[k]['large']['fall']:.2f}"])
        V[f"bs.{k}.nom_s"], V[f"bs.{k}.nom_l"], V[f"bs.{k}.big_s"], V[f"bs.{k}.big_l"] = nom[k]["small"]["fall"], nom[k]["large"]["fall"], big[k]["small"]["fall"], big[k]["large"]["fall"]
    head = ["Feature", "Smaller half / larger half of the severity range", "Falls, nominal (9 cm ball): smaller / larger", "Falls, s1.5 (13.5 cm ball, unseen design): smaller / larger"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="640 per terrain kind (64 per kind and design copy, two copies), split into halves by severity", seeds="one policy (g18_ret_rs025_s1, seed 1), trained on 9 cm-ball designs",
                    protocol="velocity task at 0.3 m/s on the tiles, undisturbed, arrestors on, 20 s episodes (eval_governor.py with --eval-designs nom,nom against s1.5,s1.5).")
    inject("generalist.html", "tab_ballsize", html_table(head, rows) + f'<p class="cap"><b>Table 16.</b> Ball size against step height. {cap}</p>')
    tex_table("ballsize", "Ball size against step height (traction-limit test). " + tex_escape(cap), "tab:gen_ballsize", head, rows, colspec="lccc", star=True)
    for k, (a, b, c, d) in {"step_up": (0.40, 0.98, 0.26, 0.80), "bump": (0.52, 0.94, 0.33, 0.62), "stairs": (0.17, 0.67, 0.19, 0.64), "large_drop": (0.64, 0.83, 0.62, 0.72)}.items():
        for q, comp, l in (("nominal smaller", V[f"bs.{k}.nom_s"], a), ("nominal larger", V[f"bs.{k}.nom_l"], b), ("s1.5 smaller", V[f"bs.{k}.big_s"], c), ("s1.5 larger", V[f"bs.{k}.big_l"], d)):
            check("S88", "Table 16", f"{k} {q}", comp, l, 0.006)
    # privileged scan against blind counterparts
    obs = ("step_up", "bump", "stairs", "large_drop")
    pairs = [("g22_ret_rs025_scan_s0", "residual 0.25 + scan"), ("g18_ret_rs025_s0", "residual 0.25, blind"), ("g22_ret_rs05_scan_s0", "residual 0.5 + scan"), ("g16_ret100_s0", "residual 0.5, blind"), ("g16_ret100_s1", "residual 0.5, blind"),
             ("g22_pure_ret_scan_s0", "pure learning + scan"), ("g16_pure_ret100_s0", "pure learning, blind"), ("g16_pure_ret100_s1", "pure learning, blind")]
    rows = []
    for run, lab in pairs:
        bk = json.load(open(WP / run / "eval2_vel_0.3_dr_surf_ps0.json"))["by_kind"]
        g = S["goal"][run]
        rows.append([f"{lab}<br><small>{run}, seed {run[-1]}</small>", f"{g['u']['success']:.2f} / {g['d']['success']:.2f}", f"{S['vel_tiles'][run]['fall_u']:.2f}"] + [f"{bk[k]['fall']:.2f}" for k in obs])
        for k in obs:
            V[f"scan.{run}.{k}"] = bk[k]["fall"]
    head = ["Policy (all with the retainer)", "Goal success undisturbed / disturbed (DR 1)", "Tile velocity falls, undisturbed"] + [f"{KIND_LABEL[k]} falls" for k in obs]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="1760 per goal and tile cell, 176 per terrain kind", seeds="one per row as listed (scan arms one seed each)",
                    protocol="eval_battery.py; the scan arms add the privileged 7 x 5 height scan ahead as a policy input (the upper bound for any terrain sensor); per-kind columns are falls in the 8 s velocity task at 0.3 m/s, undisturbed.")
    inject("generalist.html", "tab_scan", html_table(head, rows) + f'<p class="cap"><b>Table 17.</b> A perfect height map ahead, as policy input. {cap}</p>')
    tex_table("scan", "Privileged height scan as policy input against blind counterparts (retainer on). " + tex_escape(cap), "tab:gen_scan", head, rows, colspec="lcccccc", star=True)
    L = {"g22_ret_rs025_scan_s0": (0.68, 0.41, 0.36, (0.59, 0.78, 0.51, 0.70)), "g22_ret_rs05_scan_s0": (0.74, 0.53, 0.32, (0.55, 0.71, 0.45, 0.63)), "g22_pure_ret_scan_s0": (0.69, 0.50, None, (0.29, 0.24, None, 0.38))}
    for run, (u, d, v, ks) in L.items():
        check("S88", "Table 17", f"{run} goal undisturbed", S["goal"][run]["u"]["success"], u, 0.006)
        check("S88", "Table 17", f"{run} goal disturbed", S["goal"][run]["d"]["success"], d, 0.006)
        if v is not None:
            check("S88", "Table 17", f"{run} vel falls", S["vel_tiles"][run]["fall_u"], v, 0.006)
        for k, l in zip(obs, ks):
            if l is not None:
                check("S88", "Table 17", f"{run} {k}", V[f"scan.{run}.{k}"], l, 0.006)
    # pure learning stalls in front of obstacles
    pu, lq = json.load(open(WP / "g16_pure_ret100_s1" / "stepsize_nom.json"))["by_severity"], nom
    rows = []
    for k in ("step_up", "bump", "stairs", "large_drop"):
        rows.append([KIND_LABEL[k] + (" (riser)" if k == "stairs" else "") + f", larger half ({pu[k]['split'] * 100:.1f} to {pu[k]['sev_max'] * 100:.1f} cm)", f"{pu[k]['large']['fall']:.2f} / {pu[k]['large']['success']:.2f}", f"{lq[k]['large']['fall']:.2f} / {lq[k]['large']['success']:.2f}"])
        V[f"stall.{k}.pf"], V[f"stall.{k}.pc"], V[f"stall.{k}.lf"], V[f"stall.{k}.lc"] = pu[k]["large"]["fall"], pu[k]["large"]["success"], lq[k]["large"]["fall"], lq[k]["large"]["success"]
    head = ["Feature (larger half of the range)", "Pure learning, retainer (g16_pure_ret100_s1): falls / crossed", "LQR + residual 0.25, retainer (g18_ret_rs025_s1): falls / crossed"]
    cap = setup_cap(engine=ENG, dr="on at scale 1", noise="on", worlds="640 per terrain kind, halved by severity", seeds="one policy each (seed 1)",
                    protocol="velocity task at 0.3 m/s on the tiles, undisturbed, arrestors on, 20 s episodes; crossed = left the tile upright past the feature.")
    inject("generalist.html", "tab_stall", html_table(head, rows) + f'<p class="cap"><b>Table 18.</b> Pure learning stalls in front of large obstacles; it does not climb them. {cap}</p>')
    tex_table("stall", "Falls and crossings on the larger half of each obstacle range. " + tex_escape(cap), "tab:gen_stall", head, rows, colspec="lcc")
    for k, (pf, pc, lf, lc) in {"step_up": (0.69, 0.02, 0.98, 0.02), "bump": (0.30, 0.02, 0.94, 0.06), "stairs": (0.38, 0.03, 0.67, 0.33), "large_drop": (0.58, 0.39, 0.83, 0.17)}.items():
        for q, comp, l in (("pure falls", V[f"stall.{k}.pf"], pf), ("pure crossed", V[f"stall.{k}.pc"], pc), ("lqr falls", V[f"stall.{k}.lf"], lf), ("lqr crossed", V[f"stall.{k}.lc"], lc)):
            check("S88", "Table 18", f"{k} {q}", comp, l, 0.006)


def fig_traction(C, size):
    fig, axs = plt.subplots(1, 2, figsize=size, gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axs[0]; ax.set_aspect("equal"); ax.grid(False)
    r, h = 9.0, 3.0
    th = math.acos(1 - h / r)
    cx = -r * math.sin(th)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.fill(cx + r * np.cos(t), r + r * np.sin(t), color=C["mute"], alpha=0.25, lw=0); ax.plot(cx + r * np.cos(t), r + r * np.sin(t), color=C["fg"], lw=1.2)
    ax.plot([-20, 0, 0, 8], [0, 0, h, h], color=C["fg"], lw=1.6)
    ax.plot([cx, 0], [r, h], color=C["hybrid"], lw=1.3); ax.plot([cx, cx], [r, r - 6], color=C["mute"], ls=":", lw=1)
    ax.text(cx + 0.4, r - 3.5, "theta", color=C["hybrid"], fontsize=8)
    ax.annotate("", xy=(3, 0), xytext=(3, h), arrowprops=dict(arrowstyle="<->", color=C["warn"], lw=1)); ax.text(3.5, h / 2, "h", color=C["warn"], fontsize=9)
    ax.text(cx, r + 0.6, "r", fontsize=8, color=C["fg"]); ax.plot([cx], [r], marker="o", ms=3, color=C["fg"])
    ax.set_xlim(-20, 9); ax.set_ylim(-1, 19.5); ax.set_xlabel("cm"); ax.set_title("cos theta = 1 - h/r, climb if tan theta < mu", fontsize=8.5)
    ax = axs[1]
    mu = np.linspace(0.3, 1.3, 100)
    for rr, col, lab in ((9.0, C["hybrid"], "9 cm ball (training designs)"), (13.5, C["warn"], "13.5 cm ball (s1.5)")):
        ax.plot(mu, rr * (1 - np.cos(np.arctan(mu))), color=col, lw=1.8, label=lab)
    ax.axvline(1.0, color=C["mute"], ls="--", lw=0.8)
    ax.axhspan(1.1, 5.9, color=C["bad"], alpha=0.08, lw=0); ax.text(0.32, 5.4, "step-up range on the map (1.1 to 5.9 cm)", fontsize=7, color=C["bad"])
    ax.set_xlabel("ball-floor friction coefficient mu"); ax.set_ylabel("largest climbable step h [cm]"); ax.set_title("h_max = r (1 - cos(arctan mu))", fontsize=8.5); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


NAV_CFG = [("nav_blind", "blind (straight to the goal)"), ("nav_oracle", "privileged height map (upper bound)"), ("nav_tof18_head", "ToF 18 x 14, 55 x 42 deg, head, pitch -25 deg"),
           ("nav_cam32_head", "depth camera 32 x 24, 87 x 58 deg, head, pitch -25 deg"), ("nav_cam32_head_p40", "depth camera 32 x 24, head, pitch -40 deg"), ("nav_tof8_head", "ToF 8 x 8, 45 x 45 deg, head, pitch -25 deg"),
           ("nav_tof18_low", "ToF 18 x 14 at the base origin, pitch -10 deg"), ("nav_tof18_cones", "two ToF 18 x 14 on the body cones")]


def build_nav():
    D1 = WP / "g18_ret_rs025_s1"
    nav = {}
    for f in D1.glob("nav_*.json"):
        nav[f.stem] = json.load(open(f))
    S["nav"] = {k: v["overall"] for k, v in nav.items()}
    def row(stem, lab):
        o = nav[stem]["overall"]
        for q in ("fall", "success", "stuck", "over_block"):
            V[f"nav.{stem}.{q}"] = o[q]
        return [lab, f"{o['fall']:.2f}", f"{o['success']:.2f}", f"{o['stuck']:.2f}", f"{o['over_block']:.2f}"]
    head = ["Hazard source", "Falls", "Reached goal", "Stuck", "Rolled onto a block"]
    rows = [row(st, lab) for st, lab in NAV_CFG]
    cap = setup_cap(engine=ENG_NAV18, dr="on at scale 1 (11 designs, floor classes)", noise="on (sensor ranges: noise 0.5 percent of range, 2 percent dropout)", worlds="528 per row (48 per design, 11 designs)", seeds="one controller seed (g18_ret_rs025_s1, frozen)",
                    protocol="blocks map, undisturbed, 20 s episodes, goal 1.2 m away behind a block; wavefront planner (eval_nav.py) with obstacles inflated by 0.22 m on a 5 cm grid; nothing is retrained.")
    inject("generalist.html", "tab_nav", html_table(head, rows) + f'<p class="cap"><b>Table 19.</b> Navigation around unclimbable blocks, undisturbed. {cap}</p>')
    tex_table("nav", "Navigation around unclimbable blocks (undisturbed), frozen then-recommended controller with a planner. " + tex_escape(cap), "tab:gen_nav", head, rows, colspec="lcccc", star=True)
    rows = [row("nav_blind_pp", "blind"), row("nav_oracle_pp", "privileged height map"), row("nav_tof18_head_pp", "ToF 18 x 14, head, pitch -25 deg"),
            row("nav_tof18_hw0.30", "ToF 18 x 14, margin 0.30 m (undisturbed)"), row("nav_tof18_hw0.38", "ToF 18 x 14, margin 0.38 m (undisturbed)"),
            row("nav_tof18_pp_hw0.30", "ToF 18 x 14, margin 0.30 m, pushes + pulls"), row("nav_tof18_pp_hw0.38", "ToF 18 x 14, margin 0.38 m, pushes + pulls"), row("nav_oracle_pp_hw0.38", "privileged map, margin 0.38 m, pushes + pulls")]
    cap = setup_cap(engine=ENG_NAV18, dr="on at scale 1", noise="on", worlds="528 per row", seeds="one controller seed (g18_ret_rs025_s1, frozen)",
                    protocol="as Table 19, with pushes 5 to 20 N and pulls 3 to 12 N (first three rows), and with the obstacle margin widened from 0.22 m to 0.30 or 0.38 m (last five rows; their disturbance setting is in the label).")
    inject("generalist.html", "tab_navpp", html_table(head, rows) + f'<p class="cap"><b>Table 20.</b> Navigation with pushes and pulls, and a wider planner margin. {cap}</p>')
    tex_table("navpp", "Navigation with pushes and pulls, and a wider planner margin. " + tex_escape(cap), "tab:gen_navpp", head, rows, colspec="lcccc", star=True)
    L = {"nav_blind": (0.91, 0.09, 0.00, 0.74), "nav_oracle": (0.09, 0.91, 0.00, 0.00), "nav_tof18_head": (0.11, 0.85, 0.04, 0.02), "nav_cam32_head": (0.13, 0.83, 0.05, 0.02), "nav_cam32_head_p40": (0.15, 0.51, 0.34, 0.00),
         "nav_tof8_head": (0.14, 0.73, 0.13, 0.02), "nav_tof18_low": (0.11, 0.78, 0.11, 0.02), "nav_tof18_cones": (0.35, 0.58, 0.07, 0.25), "nav_blind_pp": (0.86, 0.14, 0.00, 0.44), "nav_oracle_pp": (0.41, 0.58, 0.01, 0.13), "nav_tof18_head_pp": (0.52, 0.45, 0.03, 0.15)}
    for st, vals in L.items():
        for q, l in zip(("fall", "success", "stuck", "over_block"), vals):
            check("S89", "Tables 19/20", f"{st} {q}", V[f"nav.{st}.{q}"], l, 0.006)
    for st, (su, stk) in {"nav_tof18_hw0.30": (0.11, 0.70), "nav_tof18_hw0.38": (0.00, 0.71)}.items():
        check("S89", "Table 20", f"{st} success", V[f"nav.{st}.success"], su, 0.006)
        check("S89", "Table 20", f"{st} stuck", V[f"nav.{st}.stuck"], stk, 0.006)
    check("S89", "Table 20", "oracle pp 0.38 success", V["nav.nav_oracle_pp_hw0.38.success"], 0.00, 0.006)
    check("S89", "Table 20", "oracle pp 0.38 falls", V["nav.nav_oracle_pp_hw0.38.fall"], 0.57, 0.006)


def build_s90_s91():
    D1 = WP / "g18_ret_rs025_s1"
    J_ = lambda d, n: json.load(open(WP / d / n))
    def hr(o):
        return f"{o['heading_rate_rms']:.1f}" if "heading_rate_rms" in o else f"14.4{DAG}"
    def row(lab, o):
        return [lab, f"{o['success']:.2f}", f"{o['fall']:.2f}", f"{o['stuck']:.2f}", f"{o['over_block']:.2f}", hr(o)]
    # smoothness, privileged map
    o89, oS = J_("g18_ret_rs025_s1", "nav_oracle.json")["overall"], J_("g18_ret_rs025_s1", "navS_oracle.json")["overall"]
    rows = [row("S89 wavefront, best neighbour every step", o89), row("smooth follower (carrot 0.3 m, turn rate at most 2 rad/s)", oS)]
    V["sm.o89.hr"], V["sm.oS.hr"], V["sm.o89.succ"], V["sm.oS.succ"] = o89.get("heading_rate_rms", 14.4), oS["heading_rate_rms"], o89["success"], oS["success"]
    head = ["Planner (privileged height map)", "Reached goal", "Falls", "Stuck", "Rolled onto a block", "Commanded heading-rate RMS [rad/s]"]
    cap = setup_cap(engine=ENG_NAV18, dr="on at scale 1", noise="on", worlds="528 per row (48 per design, 11 designs)", seeds="one controller seed (g18_ret_rs025_s1, frozen)",
                    protocol="blocks map, undisturbed, 20 s, privileged height map; the smooth follower (nav.py) tracks a carrot point 0.3 m down the same distance field, turns the commanded heading at most 2 rad/s, lowers speed with heading error and changes it at most 0.6 m/s^2. A dagger marks a value from the study log (the S89 file has no heading-rate field).")
    inject("generalist.html", "tab_smooth", html_table(head, rows) + f'<p class="cap"><b>Table 21.</b> Smooth following against the jerky wavefront planner. {cap}</p>')
    tex_table("smooth", "Smooth follower against the jerky wavefront planner (privileged map). " + tex_escape(cap), "tab:gen_smooth", head, rows, colspec="lccccc", star=True)
    # ToF variants
    spec = [("navS_tof18", "smooth follower, sensor yawed along the command"), ("navS_tof18_aim_w2.0", "smooth follower, sensor aimed at the target heading, 2 rad/s (adopted)"), ("navS_tof18_aim_w4.0", "sensor aimed at the target heading, 4 rad/s"),
            ("navS_tof18_w4", "rejected: faster turning, 4 rad/s, sensor along the command"), ("navS_tof18_rp4", "rejected: replanning every 0.04 s"), ("navS_tof18_prox0.3", "rejected: slow-down within 0.3 m of seen obstacles"), ("navS_tof18_prox0.5", "rejected: slow-down within 0.5 m"), ("nav_tof18_head", "S89 wavefront (jerky), for reference")]
    rows = []
    for st, lab in spec:
        o = J_("g18_ret_rs025_s1", st + ".json")["overall"]
        rows.append(row(lab, o))
        for q in ("success", "fall", "stuck", "over_block"):
            V[f"sm.{st}.{q}"] = o[q]
        if "heading_rate_rms" in o:
            V[f"sm.{st}.hr"] = o["heading_rate_rms"]
    head[0] = "Smooth follower with the ToF 18 x 14 sensor (head)"
    cap = setup_cap(engine=ENG_NAV18, dr="on at scale 1", noise="on (sensor ranges noisy, with dropout)", worlds="528 per row", seeds="one controller seed (g18_ret_rs025_s1, frozen)",
                    protocol="blocks map, undisturbed, 20 s; the head gimbal is yawed either to the command direction or to the planner's target heading (eval hook; on hardware the gimbal would be commanded by the navigation layer).")
    inject("generalist.html", "tab_smoothtof", html_table(head, rows) + f'<p class="cap"><b>Table 22.</b> Smooth following with a real sensor: aim the sensor where you are going. {cap}</p>')
    tex_table("smoothtof", "Smooth following with the ToF sensor: variants. " + tex_escape(cap), "tab:gen_smoothtof", head, rows, colspec="lccccc", star=True)
    L = {"navS_oracle": (0.86, 0.14, 0.00, 0.04), "navS_tof18": (0.72, 0.22, 0.05, 0.16), "navS_tof18_aim_w2.0": (0.78, 0.16, 0.05, 0.03), "navS_tof18_aim_w4.0": (0.76, 0.17, 0.07, 0.04),
         "navS_tof18_w4": (0.74, None, None, None), "navS_tof18_rp4": (0.69, None, None, None), "navS_tof18_prox0.3": (0.44, None, 0.28, None), "navS_tof18_prox0.5": (0.26, None, 0.46, None)}
    for st, vals in L.items():
        o = J_("g18_ret_rs025_s1", st + ".json")["overall"]
        for q, l in zip(("success", "fall", "stuck", "over_block"), vals):
            if l is not None:
                check("S90/S91", "Tables 21/22", f"{st} {q}", o[q], l, 0.006)
    check("S90", "Table 21", "smooth heading-rate RMS", oS["heading_rate_rms"], 1.7, 0.06)
    check("S90", "Table 22", "aim 4 rad/s heading-rate RMS", V["sm.navS_tof18_aim_w4.0.hr"], 3.3, 0.06)
    # S91: training with the planner in the loop
    conds = [("navS_oracle", "perfect map, undisturbed"), ("navS_oracle_pp", "perfect map, pushes + pulls"), ("navS_tof18", "ToF 18 x 14, undisturbed"), ("navS_tof18_pp", "ToF 18 x 14, pushes + pulls")]
    rows = []
    for st, lab in conds:
        cells = [lab]
        for run in ("g18_ret_rs025_s1", "g23_nav_rs025_s0", "g23_nav_rs025_s1"):
            o = J_(run, st + ".json")["overall"]
            cells.append(f"{o['success']:.2f} / {o['fall']:.2f} / {o['over_block']:.2f}")
            V[f"nt.{run}.{st}.success"], V[f"nt.{run}.{st}.fall"], V[f"nt.{run}.{st}.ob"] = o["success"], o["fall"], o["over_block"]
        rows.append(cells)
    head = ["Hazard source", "frozen then-recommended controller (g18_ret_rs025_s1): reached / fell / on block", "planner-trained, seed 0 (g23_nav_rs025_s0)", "planner-trained, seed 1 (g23_nav_rs025_s1)"]
    cap = setup_cap(engine=ENG_NAV18, dr="on at scale 1", noise="on", worlds="528 per cell", seeds="one frozen controller seed against two planner-trained seeds",
                    protocol="blocks map, smooth follower (sensor along the command), 20 s; the planner-trained runs spent 30 percent of training worlds on blocks tiles following the smooth planner on the privileged map, otherwise the then-recommended recipe.")
    inject("generalist.html", "tab_navtrain", html_table(head, rows) + f'<p class="cap"><b>Table 23.</b> Training with the planner in the loop: a null result. {cap}</p>')
    tex_table("navtrain", "Training with the planner in the loop against the frozen controller (null result). " + tex_escape(cap), "tab:gen_navtrain", head, rows, colspec="lccc", star=True)
    L91 = {"navS_oracle": ((0.86, 0.14, 0.04), (0.85, 0.13, 0.01), (0.85, 0.14, 0.02)), "navS_oracle_pp": ((0.55, 0.44, 0.20), (0.55, 0.43, 0.13), (0.57, 0.42, 0.13)),
           "navS_tof18": ((0.72, 0.22, 0.16), (0.68, 0.25, 0.15), (0.71, 0.23, 0.17)), "navS_tof18_pp": ((0.41, 0.56, 0.22), (0.43, 0.53, 0.18), (0.46, 0.51, 0.21))}
    for st, trip in L91.items():
        for run, (su, fa, ob) in zip(("g18_ret_rs025_s1", "g23_nav_rs025_s0", "g23_nav_rs025_s1"), trip):
            check("S91", "Table 23", f"{run} {st} success", V[f"nt.{run}.{st}.success"], su, 0.006)
            check("S91", "Table 23", f"{run} {st} fall", V[f"nt.{run}.{st}.fall"], fa, 0.006)
            check("S91", "Table 23", f"{run} {st} on block", V[f"nt.{run}.{st}.ob"], ob, 0.006)
    V["nt.ob_gain_pp"] = float(np.mean([V[f"nt.g18_ret_rs025_s1.{c}.ob"] - np.mean([V[f"nt.g23_nav_rs025_s{k}.{c}.ob"] for k in (0, 1)]) for c in ("navS_oracle_pp", "navS_tof18_pp")]))


def build_s104():
    """S104: navigation on the then-recommended checkpoint at its own friction (g26 s1, 0.01); S90 protocol, smooth follower, sensor aimed at the target."""
    D = WP / "g26_ret_rs025_mu01_s1"
    spec = [("blind", "navS_blind", "blind (straight to the goal)"), ("oracle", "navS_oracle", "perfect height map (upper bound)"), ("tof", "navS_tof18", "ToF 18 x 14, head, pitched down 25 deg, aimed at the target heading"),
            ("oracle_pp", "navS_oracle_pp", "perfect height map, pushes and pulls"), ("tof_pp", "navS_tof18_pp", "ToF 18 x 14, pushes and pulls")]
    rows = []
    for key, f_, lab in spec:
        d = json.load(open(D / f"{f_}.json"))
        assert d["run"] == "g26_ret_rs025_mu01_s1" and d["worlds"] == 528 and d["steps"] == 2000 and d["planner"]["kind"] == "smooth", f_
        assert d["push_scale"] == (1.0 if key.endswith("_pp") else 0.0), f_
        if key.startswith("tof"):
            assert d["planner"]["aim"] == "target" and d["sensor"]["res"] == "18x14", f_
        o = d["overall"]
        for q in ("success", "fall", "stuck", "over_block", "median_time_s", "heading_rate_rms"):
            V[f"nav26.{key}.{q}"] = o[q]
        rows.append([lab, f"{o['success']:.2f}", f"{o['fall']:.2f}", f"{o['stuck']:.2f}", f"{o['over_block']:.2f}", f"{o['median_time_s']:.1f} s", f"{o['heading_rate_rms']:.2f}"])
    V["nav26.tof_gap"] = round(V["nav26.oracle.success"], 2) - round(V["nav26.tof.success"], 2)      # as printed in Table 37
    check("S104", "text", "ToF within ... of the perfect map", V["nav26.tof_gap"], 0.05, 0.006)
    head = ["Hazard source (then-recommended checkpoint)", "Reached goal", "Fell", "Stuck", "Rolled onto a block", "Median time to goal", "Commanded heading-rate RMS [rad/s]"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01, the controller's own training value", dr="on at scale 1 (11 designs, floor classes)", noise="on (sensor ranges: noise 0.5 percent of range, 2 percent dropout)",
                    worlds="528 per row (48 per design, 11 designs)", seeds="one controller seed (g26_ret_rs025_mu01_s1, a then-recommended checkpoint, frozen)",
                    protocol="blocks map, 20 s, smooth follower (carrot 0.3 m, turn rate at most 2 rad/s; the blind row heads straight for the goal), sensor gimbal aimed at the planner's target heading; pushes 5 to 20 N and pulls 3 to 12 N in the last two rows; files g26_ret_rs025_mu01_s1/navS_{blind,oracle,tof18}.json and navS_{oracle,tof18}_pp.json.")
    inject("generalist.html", "tab_nav26", html_table(head, rows) + f'<p class="cap"><b>Table 37.</b> Navigation with the then-recommended checkpoint on the corrected contact model. {cap}</p>')
    tex_table("nav26", "Navigation with the then-recommended checkpoint at friction 0.01 (smooth follower). " + tex_escape(cap), "tab:gen_nav26", head, rows, colspec="lcccccc", star=True)
    L = {"blind": (0.09, 0.91, 0.00, 0.75), "oracle": (0.89, 0.07, 0.04, 0.01), "tof": (0.84, 0.07, 0.08, 0.04), "oracle_pp": (0.57, 0.39, 0.04, 0.16), "tof_pp": (0.51, 0.43, 0.06, 0.17)}
    for key, vals in L.items():
        for q, l in zip(("success", "fall", "stuck", "over_block"), vals):
            check("S104", "Table 37", f"{key} {q}", V[f"nav26.{key}.{q}"], l, 0.006)
    for key, (t_, h_) in {"oracle": (9.5, 1.69), "tof": (10.8, 1.74)}.items():
        check("S104", "Table 37", f"{key} median time", V[f"nav26.{key}.median_time_s"], t_, 0.06)
        check("S104", "Table 37", f"{key} heading-rate RMS", V[f"nav26.{key}.heading_rate_rms"], h_, 0.006)


MR_RUNS = [   # S110 to S116: the motion-layer recipe and its variants (run, row label, family key)
    ("g26_ret_rs025_mu01_s1", "previous recommendation, raw commands, no motion layer (g26 s1)", "g26"),
    ("g29_mref_s0", "motion layer, fixed authority 0.25 (g29 s0)", "g29"), ("g29_mref_s1", "motion layer, fixed authority 0.25 (g29 s1)", "g29"), ("g29_mref_s2", "motion layer, fixed authority 0.25 (g29 s2)", "g29"),
    ("g32_mref_rsad_s0", "adaptive authority 0.15 / 0.3 (g32 s0)", "g32"), ("g32_mref_rsad_s1", "adaptive authority 0.15 / 0.3 (g32 s1)", "g32"),
    ("g34_mref_rsad_db_s0", "<b>then-recommended:</b> adaptive authority + goal dead band (g34 s0)", "g34"), ("g34_mref_rsad_db_s1", "<b>then-recommended:</b> adaptive authority + goal dead band (g34 s1)", "g34"),
    ("g34_mref_rsad_db_s2", "<b>then-recommended:</b> adaptive authority + goal dead band (g34 s2)", "g34"),
    ("g33_mref_rsad_trk3_s0", "adaptive + tracking reward 3 (g33 s0)", "g33"), ("g33_mref_rsad_trk3_s1", "adaptive + tracking reward 3 (g33 s1)", "g33"),
    ("g35_mref_rsad_hardw_s0", "adaptive + hard-design weighting (g35 s0)", "g35"), ("g35_mref_rsad_hardw_s1", "adaptive + hard-design weighting (g35 s1)", "g35"),
    ("g35_mref_rsad_hardw_s2", "adaptive + hard-design weighting (g35 s2)", "g35"),
    ("g36_deploy_400m_s0", "then-recommended + weighting, 400M steps (g36 s0)", "g36"), ("g36_deploy_400m_s1", "then-recommended + weighting, 400M steps (g36 s1)", "g36"),
    ("g36_deploy_400m_s2", "then-recommended + weighting, 400M steps (g36 s2)", "g36"),
    ("g37_push15_s0", "then-recommended, trained on 1.5 x pushes and pulls (g37 s0)", "g37"), ("g37_push20_s0", "then-recommended, trained on 2 x pushes and pulls (g37 s0)", "g37")]
HARD = ("kb025", "bal4", "tau6")      # light ball (0.25 x ball mass), heavy body (4 x ballast), weak motors (6 N m torque limit)


def build_s105_s116():
    """S105 to S116: the motion-reference layer, adaptive residual authority and the goal dead band (then-recommended recipe g34), the variants that
    did not help, navigation and the contact escape on the new controller, and the limits (hard designs on flat ground, sustained pulls)."""
    rows, S["motion"] = [], {}
    for run, lab, fam in MR_RUNS:
        D = WP / run
        a = json.load(open(D / "args.json"))
        mf = "motion_goal_raw.json" if fam == "g26" else "motion_goal_ref.json"
        m = json.load(open(D / mf))
        assert m["scenario"] == "goal" and m["worlds"] == 528 and m["steps"] == 1500 and m["push_scale"] == 0.0 and str(m["pair_mu1"]) == "0.01" and not m.get("lqr_only", False), run
        assert m["mode"] == ("raw" if fam == "g26" else "ref"), run
        if fam != "g26":
            assert a.get("motion_ref"), run
        b = json.load(open(D / "battery.json"))
        g, v = b["goal"], b["vel"]
        hard = float(np.mean([m["by_design"][k] for k in HARD]))
        rec = dict(fam=fam, fall=m["fall"], rev=m["cmd_reversals_per_world"], hold=m["hold_speed_rms"], hard=hard, steps=a.get("steps"), rs_adapt=a.get("rs_adapt"), design_w=a.get("design_w"), push_train=a.get("push_train"), rw_track=a.get("rw_track"),
                   goal={f"{sc}.{n}": g[f"drs{sc}_{nn}"]["overall"]["success"] for sc in SCALES for n, nn in (("u", "none"), ("d", "pushes+pulls"))},
                   vel={f"{sc}.{n}": v[f"drs{sc}_{nn}"]["overall"]["fall"] for sc in SCALES for n, nn in (("u", "none"), ("d", "pushes+pulls"))},
                   ach={sp: b["flat"][sp]["achieved"] for sp in SP})
        S["motion"][run] = rec
        V[f"mr.{run}.fall"], V[f"mr.{run}.rev"], V[f"mr.{run}.hold"], V[f"mr.{run}.hard"] = rec["fall"], rec["rev"], rec["hold"], hard
        for k_, x in rec["goal"].items():
            V[f"mr.{run}.goal.{k_}"] = x
        for k_, x in rec["vel"].items():
            V[f"mr.{run}.vel.{k_}"] = x
        gc = rec["goal"]
        rows.append([lab, f"{rec['fall']:.3f}", f"{rec['rev']:.2f}", f"{rec['hold']:.3f}", f"{hard:.2f}",
                     f"{gc['1.u']:.2f} / {gc['1.d']:.2f}", f"{gc['1.75.u']:.2f} / {gc['1.75.d']:.2f}", f"{gc['2.5.u']:.2f} / {gc['2.5.d']:.2f}", f"{rec['vel']['1.u']:.2f} / {rec['vel']['1.d']:.2f}"])
    # family ranges for the prose
    fams = {}
    for run, rec in S["motion"].items():
        fams.setdefault(rec["fam"], []).append(rec)
    for fam, recs in fams.items():
        for q in ("fall", "rev", "hold", "hard"):
            V[f"mrf.{fam}.{q}.min"], V[f"mrf.{fam}.{q}.max"] = min(r[q] for r in recs), max(r[q] for r in recs)
        for k_ in recs[0]["goal"]:
            V[f"mrf.{fam}.goal.{k_}.min"], V[f"mrf.{fam}.goal.{k_}.max"] = min(r["goal"][k_] for r in recs), max(r["goal"][k_] for r in recs)
        for k_ in recs[0]["vel"]:
            V[f"mrf.{fam}.vel.{k_}.min"], V[f"mrf.{fam}.vel.{k_}.max"] = min(r["vel"][k_] for r in recs), max(r["vel"][k_] for r in recs)
        V[f"mrf.{fam}.ach.min"] = min(min(r["ach"].values()) for r in recs)
        V[f"mrf.{fam}.ach.max"] = max(max(r["ach"].values()) for r in recs)
    check("S113", "index.html keynum", "g34 flat achieved / command, largest (log: 1.06 at 0.9 m/s)", V["mrf.g34.ach.max"], 1.06, 0.006)
    check("S113", "index.html keynum", "g34 flat achieved / command, smallest (0.96 on the page)", V["mrf.g34.ach.min"], 0.96, 0.006)
    head = ["Recipe (run)", "Goal test: falls", "Command reversals per world", "Hold speed RMS [m/s]", "Falls of the three hard designs (mean)",
            "Battery goal success DR 1: undisturbed / pushes + pulls", "DR 1.75", "DR 2.5", "Tile velocity falls DR 1: undisturbed / pushes + pulls"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on at scale 1 unless a column says 1.75 or 2.5 (training: ramped to 1.75)",
                    noise="on", worlds="goal test 528 per row (48 per design, 11 designs); battery 1760 per cell (16 per design and terrain kind)",
                    seeds="one per row; three seeds of the then-recommended recipe g34 (s0, s1, s2)",
                    protocol="goal test: eval_motion.py, goal 1.2 m away on flat interior tiles, 15 s, no pushes, commands through the motion layer for every run trained with it (the g26 row uses raw commands, as it was trained); "
                             "hard designs = kb025 (light ball, 0.25 x ball mass), bal4 (heavy body, 4 x ballast) and tau6 (weak motors, 6 N m torque limit against 30). Battery: eval_battery.py, goal task 1.1 m and velocity task 0.3 m/s on the tiled map, 8 s, surfaces on, arrestors on; "
                             "pushes 5 to 20 N for 50 ms every 2 to 4 s and pulls 3 to 12 N for 0.5 to 2 s every 4 to 10 s. All recipes 200M steps except g36 (400M); files outputs/study/warp_ppo/RUN/motion_goal_ref.json (g26: motion_goal_raw.json) and RUN/battery.json.")
    inject("generalist.html", "tab_motion", html_table(head, rows) + f'<p class="cap"><b>Table 38.</b> The motion-layer recipe and its variants. {cap}</p>')
    tex_table("motion", "The motion-layer recipe and its variants (S110 to S116). " + tex_escape(cap), "tab:gen_motion", head, rows, colspec="p{4.2cm}cccccccc", star=True)
    # log values: (goal falls, reversals, hold speed) and battery goal success cells (DR1 u d, 1.75 u d, 2.5 u d), velocity falls DR1 u d
    L = {"g26_ret_rs025_mu01_s1": ("S110", (0.15, 1.19, 0.077), None, None),
         "g32_mref_rsad_s0": ("S113", (0.083, 4.94, 0.096), (0.76, 0.50, 0.73, 0.44, 0.68, 0.39), (0.32, 0.48)),
         "g32_mref_rsad_s1": ("S113", (0.104, 0.57, 0.063), (0.75, 0.52, 0.72, 0.46, 0.67, 0.38), (0.32, 0.48)),
         "g33_mref_rsad_trk3_s0": ("S113", (0.100, 0.99, 0.068), (0.77, 0.53, 0.75, 0.45, 0.70, 0.41), (0.32, 0.48)),
         "g33_mref_rsad_trk3_s1": ("S113 addendum", (0.081, 1.38, 0.060), (0.77, 0.51, 0.73, 0.45, 0.67, 0.38), (0.30, 0.48)),
         "g34_mref_rsad_db_s0": ("S113", (0.104, 0.67, 0.058), (0.79, 0.51, 0.77, 0.47, 0.69, 0.39), (0.32, 0.48)),
         "g34_mref_rsad_db_s1": ("S113", (0.091, 0.86, 0.065), (0.77, 0.51, 0.73, 0.45, 0.68, 0.39), (0.32, 0.48)),
         "g34_mref_rsad_db_s2": ("S113 addendum", (0.110, 0.48, 0.052), (0.78, 0.52, 0.75, 0.45, 0.69, 0.39), (0.31, 0.48)),
         "g35_mref_rsad_hardw_s0": ("S114 addendum", (0.091, None, None), (0.81, 0.53, 0.78, 0.45, 0.72, 0.39), None),
         "g35_mref_rsad_hardw_s1": ("S114 addendum", (0.108, None, None), (0.79, 0.52, 0.77, 0.45, 0.72, 0.39), None),
         "g35_mref_rsad_hardw_s2": ("S114 addendum", (0.081, 0.69, 0.044), (0.77, 0.52, 0.73, 0.49, 0.67, 0.40), (0.31, 0.46)),
         "g36_deploy_400m_s0": ("S115", (0.068, 3.73, 0.076), (0.74, 0.47, 0.71, 0.42, 0.65, 0.35), (0.32, 0.49)),
         "g36_deploy_400m_s1": ("S115", (0.072, 2.52, 0.067), (0.73, 0.48, 0.69, 0.46, 0.65, 0.38), (0.32, 0.49)),
         "g36_deploy_400m_s2": ("S115", (0.064, 3.36, 0.073), (0.72, 0.45, 0.69, 0.41, 0.64, 0.34), (0.32, 0.49)),
         "g37_push15_s0": ("S116", (0.136, None, None), (0.79, 0.49, 0.76, 0.43, 0.70, 0.36), None),
         "g37_push20_s0": ("S116", (0.119, None, None), (0.79, 0.49, 0.76, 0.45, 0.72, 0.36), None)}
    for run, (cid, mt, gl, vl) in L.items():
        r = S["motion"][run]
        for q, l, tol in zip(("fall", "rev", "hold"), mt, (0.006, 0.006, 0.0006)):
            if l is not None:
                check(cid, "Table 38", f"{run} goal test {q}", r[q], l, tol if not (q == "fall" and cid == "S110") else 0.006)
        if gl:
            for k_, l in zip(("1.u", "1.d", "1.75.u", "1.75.d", "2.5.u", "2.5.d"), gl):
                check(cid, "Table 38", f"{run} battery goal {k_}", r["goal"][k_], l, 0.006)
        if vl:
            check(cid, "Table 38", f"{run} velocity falls DR1 undisturbed", r["vel"]["1.u"], vl[0], 0.006)
            check(cid, "Table 38", f"{run} velocity falls DR1 pushes", r["vel"]["1.d"], vl[1], 0.006)
    for run, l in (("g34_mref_rsad_db_s0", 0.32), ("g37_push15_s0", 0.37), ("g37_push20_s0", 0.37), ("g36_deploy_400m_s0", 0.23), ("g36_deploy_400m_s1", 0.25), ("g36_deploy_400m_s2", 0.22)):
        check("S115, S116", "Table 38", f"{run} hard-design falls", S["motion"][run]["hard"], l, 0.006)
    for run, l in (("g35_mref_rsad_hardw_s0", 0.285), ("g35_mref_rsad_hardw_s1", 0.292), ("g35_mref_rsad_hardw_s2", 0.278)):
        check("S114 addendum", "Table 38", f"{run} hard-design falls", S["motion"][run]["hard"], l, 0.0006)

    # navigation with the motion layer (S114 and addenda); the g26 rows are those of Table 37
    spec = [("g26", "g26_ret_rs025_mu01_s1", "", "previous recommendation, no motion layer (g26 s1; Table 37)", False),
            ("g32", "g32_mref_rsad_s1", "", "adaptive authority, with the motion layer (g32 s1)", True),
            ("g34", "g34_mref_rsad_db_s0", "", "<b>then-recommended</b>, with the motion layer (g34 s0)", True),
            ("g34raw", "g34_mref_rsad_db_s0", "_nomref", "then-recommended checkpoint, planner commands raw (g34 s0, NO_MREF=1)", False)]
    rows = []
    S["navmr"] = {}
    for key, run, suf, lab, mref in spec:
        cells = {}
        for src in ("tof18", "tof18_pp", "oracle"):
            f = WP / run / f"navS_{src}{suf}.json"
            if not f.exists():
                cells[src] = None
                continue
            d = json.load(open(f))
            assert d["run"] == run and d["worlds"] == 528 and d["steps"] == 2000 and d["planner"]["kind"] == "smooth" and bool(d.get("motion_ref")) == mref, f
            assert d["push_scale"] == (1.0 if src.endswith("_pp") else 0.0), f
            cells[src] = d["overall"]
            for q in ("success", "fall", "stuck", "heading_rate_rms", "cmd_change_mean", "median_time_s"):
                V[f"nm.{key}.{src}.{q}"] = d["overall"][q]
        S["navmr"][key] = cells
        t, p, o = cells["tof18"], cells["tof18_pp"], cells["oracle"]
        rows.append([lab, f"{t['success']:.3f} / {t['fall']:.3f} / {t['stuck']:.3f}", f"{t['heading_rate_rms']:.2f}", f"{t['cmd_change_mean']:.2f}", f"{t['median_time_s']:.1f} s",
                     (f"{p['success']:.3f} / {p['fall']:.3f}" if p else "not run"), f"{o['success']:.3f} / {o['fall']:.3f} / {o['stuck']:.3f}", f"{o['heading_rate_rms']:.2f}"])
    V["nm.wobble_cut"] = 1 - V["nm.g34.tof18.heading_rate_rms"] / V["nm.g26.tof18.heading_rate_rms"]
    V["nm.layer_cost_s"] = V["nm.g34.tof18.median_time_s"] - V["nm.g34raw.tof18.median_time_s"]
    head = ["Controller (motion layer in the loop or not)", "ToF 18 x 14: reached / fell / stuck", "Commanded heading-rate RMS [rad/s]", "Command change per step", "Median time to goal",
            "ToF, pushes and pulls: reached / fell", "Perfect map: reached / fell / stuck", "Perfect map heading-rate RMS [rad/s]"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on at scale 1 (11 designs, floor classes)", noise="on (sensor ranges: noise 0.5 percent of range, 2 percent dropout)",
                    worlds="528 (48 per design, 11 designs)", seeds="one controller seed per row, frozen",
                    protocol="eval_nav.py on the blocks map (8 cm blocks, goal 1.2 m away behind a block), 20 s, smooth wavefront follower (carrot 0.3 m, turn limit 2 rad/s, v_max 0.3 m/s), ToF on the head pitched down 25 deg and aimed at the planner target; "
                             "for runs trained with the motion layer the planner command passes through motion_ref.py (arrival braking, goal hold) unless NO_MREF=1; pushes 5 to 20 N and pulls 3 to 12 N in the pushes column; files outputs/study/warp_ppo/RUN/navS_{tof18,tof18_pp,oracle}[_nomref].json.")
    inject("generalist.html", "tab_navmref", html_table(head, rows) + f'<p class="cap"><b>Table 39.</b> Navigation with the motion layer. {cap}</p>')
    tex_table("navmref", "Navigation with the motion layer (S114). " + tex_escape(cap), "tab:gen_navmref", head, rows, colspec="p{4cm}ccccccc", star=True)
    for key, src, q, l in [("g26", "tof18", "success", 0.845), ("g26", "tof18", "heading_rate_rms", 1.74), ("g26", "oracle", "success", 0.886),
                           ("g32", "tof18", "success", 0.852), ("g32", "tof18", "fall", 0.078), ("g32", "tof18", "heading_rate_rms", 1.29), ("g32", "tof18_pp", "success", 0.544), ("g32", "oracle", "success", 0.926),
                           ("g34", "tof18", "success", 0.877), ("g34", "tof18", "fall", 0.080), ("g34", "tof18", "stuck", 0.044), ("g34", "tof18", "heading_rate_rms", 1.31), ("g34", "tof18", "cmd_change_mean", 0.40),
                           ("g34", "tof18_pp", "success", 0.557), ("g34", "tof18_pp", "fall", 0.413), ("g34", "oracle", "success", 0.938), ("g34", "oracle", "fall", 0.061), ("g34", "oracle", "stuck", 0.002),
                           ("g34raw", "tof18", "success", 0.881), ("g34raw", "tof18", "fall", 0.070), ("g34raw", "tof18", "heading_rate_rms", 1.79), ("g34raw", "oracle", "success", 0.915), ("g34raw", "oracle", "fall", 0.083)]:
        check("S114", "Table 39", f"{key} {src} {q}", V[f"nm.{key}.{src}.{q}"], l, 0.0006 if q in ("success", "fall", "stuck") else 0.006)
    check("S114 addendum", "Table 39", "median time to goal, g34 with layer", V["nm.g34.tof18.median_time_s"], 9.2, 0.06)
    check("S114 addendum", "Table 39", "median time to goal, g34 without layer", V["nm.g34raw.tof18.median_time_s"], 8.9, 0.06)
    check("S114", "text", "wobble reduction about a quarter", V["nm.wobble_cut"], 0.25, 0.03)
    # faster motion-layer settings (S114 addendum), quoted in prose only
    for nm_, f_ in (("fast", "navS_tof18_mkaacc06abrk08"), ("fast_pp", "navS_tof18_pp_mkaacc06abrk08")):
        o = json.load(open(WP / "g34_mref_rsad_db_s0" / f"{f_}.json"))["overall"]
        V[f"nm.{nm_}.success"], V[f"nm.{nm_}.fall"] = o["success"], o["fall"]
    check("S114 addendum", "text", "faster layer, pushes: reached", V["nm.fast_pp.success"], 0.513, 0.0006)
    check("S114 addendum", "text", "faster layer, pushes: fell", V["nm.fast_pp.fall"], 0.455, 0.0006)

    # blind navigation and the contact escape (S109 and the S115 addendum)
    rows = []
    for key, run in (("g26", "g26_ret_rs025_mu01_s1"), ("g34", "g34_mref_rsad_db_s0")):
        for src, lab in (("blind_40s", "blind, straight at the goal"), ("bump_v15_40s", "blind + contact escape (stall detection, obstacle memory, back up, re-plan), 0.15 m/s")):
            d = json.load(open(WP / run / f"navS_{src}.json"))
            assert d["run"] == run and d["worlds"] == 528 and d["steps"] == 4000 and d["push_scale"] == 0.0, run
            o = d["overall"]
            for q in ("success", "fall", "stuck", "over_block", "median_time_s"):
                V[f"esc.{key}.{src}.{q}"] = o[q]
            rows.append([f"{lab}<br><small>{'previous recommendation g26 s1, no layer' if key == 'g26' else 'then-recommended g34 s0, motion layer'}</small>", f"{o['success']:.3f}", f"{o['fall']:.3f}", f"{o['stuck']:.3f}", f"{o['over_block']:.2f}", f"{o['median_time_s']:.1f} s"])
    head = ["Controller", "Reached goal", "Fell", "Stuck", "Rolled onto a block", "Median time to goal"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on at scale 1", noise="on", worlds="528 per row (48 per design, 11 designs)", seeds="one controller seed per pair of rows, frozen",
                    protocol="blocks map as Table 39 but no terrain sensing at all, 40 s episodes; escape layer experiments/warp_ppo/escape.py through eval_nav.py --source bump: stall detection (under 30 percent of the commanded speed for 1 s, no 0.1 m approach in 4 s, or tilt above 15 deg), a 0.3 m obstacle segment marked 0.2 m ahead, 1 s back-up at 0.15 m/s, re-plan, give up after 6 failures; files outputs/study/warp_ppo/RUN/navS_{blind,bump_v15}_40s.json.")
    inject("generalist.html", "tab_escape", html_table(head, rows) + f'<p class="cap"><b>Table 40.</b> A blind robot that finds obstacles by bumping into them. {cap}</p>')
    tex_table("escape", "Blind navigation with a contact escape layer (S109, S115). " + tex_escape(cap), "tab:gen_escape", head, rows, colspec="p{5cm}ccccc", star=True)
    for key, src, q, l in [("g26", "blind_40s", "success", 0.078), ("g26", "blind_40s", "fall", 0.922), ("g26", "bump_v15_40s", "success", 0.547), ("g26", "bump_v15_40s", "fall", 0.237), ("g26", "bump_v15_40s", "stuck", 0.216),
                           ("g34", "blind_40s", "success", 0.017), ("g34", "blind_40s", "fall", 0.983), ("g34", "bump_v15_40s", "success", 0.634), ("g34", "bump_v15_40s", "fall", 0.261), ("g34", "bump_v15_40s", "stuck", 0.104)]:
        check("S115 addendum" if key == "g34" else "S109", "Table 40", f"{key} {src} {q}", V[f"esc.{key}.{src}.{q}"], l, 0.0006)

    # limits on flat ground: the hard designs and sustained pulls (S114 hard-design addendum, S116 and its addendum), g34 s0, fully flat map
    D = WP / "g34_mref_rsad_db_s0"
    full = json.load(open(D / "motion_goal_ref_allflat.json")); lqr = json.load(open(D / "motion_goal_ref_allflat_lqronly.json"))
    assert full["worlds"] == lqr["worlds"] == 528 and lqr["lqr_only"] and not full["lqr_only"] and full["push_scale"] == lqr["push_scale"] == 0.0
    others = [k for k in full["by_design"] if k not in HARD]
    V["lim.full.all"], V["lim.lqr.all"] = full["fall"], lqr["fall"]
    for k in HARD:
        V[f"lim.full.{k}"], V[f"lim.lqr.{k}"] = full["by_design"][k], lqr["by_design"][k]
    V["lim.full.others_max"] = max(full["by_design"][k] for k in others)
    V["lim.lqr.others_min"], V["lim.lqr.others_max"] = min(lqr["by_design"][k] for k in others), max(lqr["by_design"][k] for k in others)
    rows = [["goal arrival, no disturbances: full controller (LQR + residual + motion layer)", f"{full['fall']:.3f}", f"{full['by_design']['kb025']:.2f}", f"{full['by_design']['bal4']:.2f}", f"{full['by_design']['tau6']:.2f}", f"at most {V['lim.full.others_max']:.2f}"],
            ["goal arrival, no disturbances: LQR alone (residual zeroed), same motion layer", f"{lqr['fall']:.3f}", f"{lqr['by_design']['kb025']:.2f}", f"{lqr['by_design']['bal4']:.2f}", f"{lqr['by_design']['tau6']:.2f}", f"{V['lim.lqr.others_min']:.2f} to {V['lim.lqr.others_max']:.2f}"]]
    sweep = [("none", "eval2_goal_goal_dr_surf_ps0_flat_noexit.json", 0.0, False, "no pushes, no pulls"), ("push", "eval2_goal_goal_dr_surf_flat_noexit.json", 1.0, False, "short pushes only (5 to 20 N, 50 ms)"),
             ("p025", "eval2_goal_goal_dr_surf_pull_ps0.25_flat_noexit.json", 0.25, True, "pushes and pulls at 0.25 x (pulls 0.75 to 3 N)"), ("p05", "eval2_goal_goal_dr_surf_pull_ps0.5_flat_noexit.json", 0.5, True, "0.5 x (pulls 1.5 to 6 N)"),
             ("p075", "eval2_goal_goal_dr_surf_pull_ps0.75_flat_noexit.json", 0.75, True, "0.75 x (pulls 2.25 to 9 N)"), ("p1", "eval2_goal_goal_dr_surf_pull_flat_noexit.json", 1.0, True, "1 x, the standard battery (pulls 3 to 12 N)")]
    for key, f_, ps, pull, lab in sweep:
        d = json.load(open(D / f_))
        es = d["env_settings"]
        assert es["tiles_flat"] and es["no_exit"] and es["push_scale"] == ps and d["args"]["pull"] == pull and d["args"]["dr"] and str(es["pair_mu1"]) == "0.01", f_
        bd = {k: x["fall"] for k, x in d["by_design"].items()}
        V[f"lim.sw.{key}.all"] = d["overall"]["fall"]
        for k_, x in bd.items():
            V[f"lim.sw.{key}.{k_}"] = x
        oth = [bd[k] for k in bd if k not in HARD]
        rows.append([f"goal task 1.1 m, {lab}", f"{d['overall']['fall']:.3f}", f"{bd['kb025']:.2f}", f"{bd['bal4']:.2f}", f"{bd['tau6']:.2f}", f"{min(oth):.2f} to {max(oth):.2f}"])
    V["lim.sw.p1.bal025"] = json.load(open(D / "eval2_goal_goal_dr_surf_pull_flat_noexit.json"))["by_design"]["bal025"]["fall"]
    head = ["Test on a fully flat map (g34 s0)", "Falls, all 11 designs", "kb025 light ball (0.25 x ball mass)", "bal4 heavy body (4 x ballast)", "tau6 weak motors (6 N m)", "the other eight designs"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on at scale 1", noise="on", worlds="528 per row in the first two rows (48 per design); 352 per row in the others (32 per design)",
                    seeds="one controller seed (g34_mref_rsad_db_s0, then-recommended recipe), frozen",
                    protocol="TILES_FLAT=1, so no terrain is involved. First two rows: eval_motion.py goal test (1.2 m, 15 s, motion layer in the loop), files motion_goal_ref_allflat[_lqronly].json. Other rows: eval_tiles2.py goal task (1.1 m, 8 s) with PUSH_SCALE multiplying both the pushes and the sustained pulls, files eval2_goal_goal_dr_surf_*_flat_noexit.json.")
    inject("generalist.html", "tab_limits", html_table(head, rows) + f'<p class="cap"><b>Table 41.</b> What the controller cannot absorb, on flat ground. {cap}</p>')
    tex_table("limits", "Limits on flat ground: hard designs and sustained pulls (S114, S116). " + tex_escape(cap), "tab:gen_limits", head, rows, colspec="p{5.2cm}ccccc", star=True)
    for q, comp, l, tol, cid in [("full controller falls", V["lim.full.all"], 0.068, 0.0006, "S114 addendum"), ("LQR alone falls", V["lim.lqr.all"], 0.324, 0.0006, "S114 addendum"),
                                  ("kb025 full", V["lim.full.kb025"], 0.38, 0.006, "S114 addendum"), ("kb025 LQR", V["lim.lqr.kb025"], 0.96, 0.006, "S114 addendum"),
                                  ("bal4 full", V["lim.full.bal4"], 0.23, 0.006, "S114 addendum"), ("bal4 LQR", V["lim.lqr.bal4"], 0.38, 0.006, "S114 addendum"),
                                  ("tau6 full", V["lim.full.tau6"], 0.12, 0.006, "S114 addendum"), ("tau6 LQR", V["lim.lqr.tau6"], 0.15, 0.006, "S114 addendum"),
                                  ("others full, max", V["lim.full.others_max"], 0.02, 0.006, "S114 addendum"), ("others LQR, min", V["lim.lqr.others_min"], 0.17, 0.006, "S114 addendum"), ("others LQR, max", V["lim.lqr.others_max"], 0.35, 0.006, "S114 addendum"),
                                  ("no disturbances", V["lim.sw.none.all"], 0.074, 0.0006, "S116"), ("short pushes only", V["lim.sw.push.all"], 0.078, 0.0006, "S116"), ("pulls 0.75 to 3 N", V["lim.sw.p025.all"], 0.078, 0.0006, "S116 addendum"),
                                  ("pulls 1.5 to 6 N", V["lim.sw.p05.all"], 0.139, 0.0006, "S116 addendum"), ("pulls 2.25 to 9 N", V["lim.sw.p075.all"], 0.221, 0.0006, "S116 addendum"), ("pulls 3 to 12 N", V["lim.sw.p1.all"], 0.361, 0.0006, "S116"),
                                  ("full pulls, bal4", V["lim.sw.p1.bal4"], 0.18, 0.006, "S116 addendum"),
                                  ("full pulls, tau6", V["lim.sw.p1.tau6"], 0.47, 0.006, "S116 addendum"), ("full pulls, zhigh", V["lim.sw.p1.zhigh"], 0.17, 0.006, "S116 addendum"),
                                  ("full pulls, kb025", V["lim.sw.p1.kb025"], 0.59, 0.006, "S116 addendum"), ("full pulls, bal025", V["lim.sw.p1.bal025"], 0.93, 0.006, "S116 addendum")]:
        check(cid, "Table 41", q, comp, l, tol)


# ----------------------------------------------------------------------------------------------------------- S117 to S122
# (run, short label, family, goal-test file measured at the true ball centre or None, history in ms)
HG_RUNS = [("g34_mref_rsad_db_s0", "g34 s0", "g34", "motion_goal_ref_centre.json", 80), ("g34_mref_rsad_db_s1", "g34 s1", "g34", "motion_goal_ref_centre.json", 80),
           ("g34_mref_rsad_db_s2", "g34 s2", "g34", "motion_goal_ref_centre.json", 80),
           ("g38_hist24_s0", "g38 s0", "g38", "motion_goal_ref_centre.json", 240), ("g38_hist24_s1", "g38 s1", "g38", "motion_goal_ref.json", 240),
           ("g38_hist48_s0", "g38 480 ms s0", "g38h48", None, 480),
           ("g39_goalcentre_s0", "g39 s0", "g39", "motion_goal_ref.json", 80), ("g39_goalcentre_s1", "g39 s1", "g39", "motion_goal_ref.json", 80),
           ("g39_goalcentre_s2", "g39 s2", "g39", "motion_goal_ref.json", 80),
           ("g40_hist24_gc_s0", "g40 s0", "g40", "motion_goal_ref.json", 240), ("g40_hist24_gc_s1", "g40 s1", "g40", "motion_goal_ref.json", 240),
           ("g40_hist24_gc_s2", "g40 s2", "g40", "motion_goal_ref.json", 240)]
HG_LABEL = {"g34": "S113 recipe, goal logic at the ball body origin (g34)", "g38": "g34 + 240 ms history (g38)", "g38h48": "g34 + 480 ms history (g38)",
            "g39": "g34 + goal logic at the true ball centre (g39, <b>recommended</b>)", "g40": "g34 + 240 ms history + goal centre (g40)"}
# S119 design sweep: (spec name, property, value label, unit note)
OTA = [("ota_ballmas_0p042", "ball mass", "0.042 kg"), ("ota_ballmas_0p084", "ball mass", "0.084 kg"), ("ota_ballmas_0p34", "ball mass", "0.34 kg"), ("ota_ballmas_0p67", "ball mass", "0.67 kg"),
       ("ota_bodycom_0p08", "body COM height", "0.08 m"), ("ota_bodycom_0p1", "body COM height", "0.10 m"), ("ota_bodycom_0p17", "body COM height", "0.17 m"), ("ota_bodycom_0p22", "body COM height", "0.22 m"),
       ("ota_bodymas_2", "body mass", "2.0 kg"), ("ota_bodymas_3", "body mass", "3.0 kg"), ("ota_bodymas_5", "body mass", "5.0 kg"), ("ota_bodymas_6p5", "body mass", "6.5 kg"),
       ("ota_motortor_3", "wheel torque limit", "3 N m"), ("ota_motortor_6", "wheel torque limit", "6 N m"), ("ota_motortor_10", "wheel torque limit", "10 N m"), ("ota_motortor_15", "wheel torque limit", "15 N m"),
       ("ota_rotorine_0p001", "rotor inertia", "0.001 kg m^2"), ("ota_rotorine_0p02", "rotor inertia", "0.02 kg m^2"),
       ("ota_contactzen_41", "contact angle", "41 deg"), ("ota_contactzen_61", "contact angle", "61 deg"),
       ("ota_ballrad_0p07", "ball radius", "7 cm"), ("ota_ballrad_0p12", "ball radius", "12 cm")]
OTA_NOM = {"ball mass": "0.168 kg", "body COM height": "0.131 m", "body mass": "3.85 kg", "wheel torque limit": "30 N m", "rotor inertia": "0.005 kg m^2", "contact angle": "51 deg", "ball radius": "9 cm"}
OTA_COLS = [("sim_goal_still_fall", "still"), ("sim_goal_pushes_fall", "pushes"), ("sim_goal_pulls_half_fall", "half pulls"), ("sim_drive_0.3_fall", "drive"), ("sim_goal_terrain_fall", "terrain")]


def build_s117_s122():
    """S117 to S122: a longer observation history (S117), the goal-point bug and its fix (S118), the design sweep (S119), the 240 ms history plus
    goal-centre controller g40 and its trade-off (S120), the recommendation g39 (S121) and the dead band on the ball-centre speed (S122, training)."""
    S["hg"], rows = {}, []
    for run, lab, fam, gfile, hist in HG_RUNS:
        D = WP / run
        a = json.load(open(D / "args.json"))
        assert int(a.get("hist", 8)) * 10 == hist and bool(a.get("goal_centre")) == (fam in ("g39", "g40")) and a.get("motion_ref") and a.get("rs_adapt") == "0.15,0.3,0.05", run
        m = json.load(open(D / (gfile or "motion_goal_ref.json")))
        assert m["scenario"] == "goal" and m["worlds"] == 528 and m["steps"] == 1500 and m["push_scale"] == 0.0 and m["mode"] == "ref" and not m.get("lqr_only"), run
        if gfile:
            assert m.get("goal_metric_point") == "ball_centre", run
        b = json.load(open(D / "battery.json"))
        g, v = b["goal"], b["vel"]
        rec = dict(fam=fam, hist=hist, fall=m["fall"], rev=m["cmd_reversals_per_world"], hold=m["hold_speed_rms"], err=(m["hold_err_rms_m"] * 100 if gfile else None),
                   over=(m["overshoot_median_m"] * 100 if gfile else None), arrived=m["arrived"], hard=float(np.mean([m["by_design"][k] for k in HARD])),
                   goal={f"{sc}.{n}": g[f"drs{sc}_{nn}"]["overall"]["success"] for sc in SCALES for n, nn in (("u", "none"), ("d", "pushes+pulls"))},
                   vel={f"{sc}.{n}": v[f"drs{sc}_{nn}"]["overall"]["fall"] for sc in SCALES for n, nn in (("u", "none"), ("d", "pushes+pulls"))},
                   ach={sp: b["flat"][sp]["achieved"] for sp in SP})
        S["hg"][run] = rec
        for q in ("fall", "rev", "hold", "err", "over", "arrived", "hard"):
            if rec[q] is not None:
                V[f"hg.{run}.{q}"] = rec[q]
        for k_, x in rec["goal"].items():
            V[f"hg.{run}.goal.{k_}"] = x
        for k_, x in rec["vel"].items():
            V[f"hg.{run}.vel.{k_}"] = x
        for sp in SP:
            V[f"hg.{run}.ach.{sp}"] = rec["ach"][sp]
        gc, vc = rec["goal"], rec["vel"]
        rows.append([f"{HG_LABEL[fam]}<br><small>{run}</small>", f"{hist} ms", f"{rec['fall']:.3f}", ("n/a" if rec["err"] is None else f"{rec['err']:.1f}"), f"{rec['hold']:.3f}", f"{rec['rev']:.2f}",
                     ("n/a" if rec["over"] is None else f"{rec['over']:.1f}"), f"{rec['hard']:.2f}",
                     f"{gc['1.u']:.2f} / {gc['1.d']:.2f}", f"{gc['1.75.u']:.2f} / {gc['1.75.d']:.2f}", f"{gc['2.5.u']:.2f} / {gc['2.5.d']:.2f}", f"{vc['1.u']:.2f} / {vc['1.d']:.2f}", f"{rec['ach']['0.9']:.3f}"])
    fams = {}
    for run, rec in S["hg"].items():
        fams.setdefault(rec["fam"], []).append(rec)
    for fam, recs in fams.items():
        for q in ("fall", "rev", "hold", "err", "over", "hard"):
            xs = [r[q] for r in recs if r[q] is not None]
            if xs:
                V[f"hgf.{fam}.{q}.min"], V[f"hgf.{fam}.{q}.max"] = min(xs), max(xs)
        for k_ in recs[0]["goal"]:
            V[f"hgf.{fam}.goal.{k_}.min"], V[f"hgf.{fam}.goal.{k_}.max"] = min(r["goal"][k_] for r in recs), max(r["goal"][k_] for r in recs)
        for k_ in recs[0]["vel"]:
            V[f"hgf.{fam}.vel.{k_}.min"], V[f"hgf.{fam}.vel.{k_}.max"] = min(r["vel"][k_] for r in recs), max(r["vel"][k_] for r in recs)
        V[f"hgf.{fam}.ach.min"] = min(min(r["ach"].values()) for r in recs)
        V[f"hgf.{fam}.ach.max"] = max(max(r["ach"].values()) for r in recs)
    V["hg.err_cut.min"] = V["hgf.g34.err.min"] / V["hgf.g39.err.max"]
    V["hg.err_cut.max"] = V["hgf.g34.err.max"] / V["hgf.g39.err.min"]
    head = ["Recipe (run)", "Observation history", "Goal test: falls", "Hold error RMS at the ball centre [cm]", "Hold speed RMS [m/s]", "Command reversals per world", "Median overshoot [cm]",
            "Falls of the three hard designs (mean)", "Battery goal success DR 1: undisturbed / pushes + pulls", "DR 1.75", "DR 2.5", "Tile velocity falls DR 1: undisturbed / pushes + pulls", "Flat 0.9 m/s: achieved / commanded"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on at scale 1 unless a column says 1.75 or 2.5 (training: ramped to 1.75)", noise="on",
                    worlds="goal test 528 per row (48 per design, 11 designs); battery 1760 per cell (16 per design and terrain kind)", seeds="one per row; three seeds each of g34, g39 and g40, two of g38 at 240 ms, one at 480 ms",
                    protocol="goal test: eval_motion.py, goal 1.2 m away on flat interior tiles, 15 s, no pushes, motion layer in the loop; hold error, overshoot and arrival measured at the true ball centre for every row (files motion_goal_ref.json for runs evaluated after S118, motion_goal_ref_centre.json for the re-measured g34 and g38 s0); "
                             "the 480 ms run was not re-measured at the centre, so its hold error and overshoot are n/a and its other goal-test numbers come from the original file. Battery: eval_battery.py as Table 38; goal success is judged at the true centre for g39 and g40 (their goal rule) and at the body origin for g34 and g38, so g39 and g40 face a slightly stricter test. All runs 200M steps, S113 recipe otherwise unchanged.")
    inject("generalist.html", "tab_histgc", html_table(head, rows) + f'<p class="cap"><b>Table 42.</b> A longer memory and the goal point on the true ball centre. {cap}</p>')
    tex_table("histgc", "A longer observation history and the goal point on the true ball centre (S117 to S121). " + tex_escape(cap), "tab:gen_histgc", head, rows, colspec="p{4.2cm}cccccccccccc", star=True)
    # log values: (falls, hold err cm, hold speed, reversals, overshoot cm), battery goal cells, velocity falls cells, flat 0.9 m/s
    L = {"g34_mref_rsad_db_s0": ("S118", (0.098, 12.6, 0.058, 0.62, 20.5), None, None, None),
         "g34_mref_rsad_db_s1": ("S118", (0.089, 11.4, 0.062, 1.05, 16), None, None, None),
         "g34_mref_rsad_db_s2": ("S118", (0.095, 13.6, 0.060, 0.54, 23), None, None, None),
         "g38_hist24_s0": ("S117", (None, None, 0.041, None, None), (0.81, 0.53, 0.77, 0.47, 0.72, 0.39), (0.28, 0.48, 0.30, 0.51, 0.35, 0.60), 1.048),
         "g38_hist24_s1": ("S118 addendum", (0.059, 8.5, None, 2.10, 11.4), (0.81, 0.51, 0.78, 0.46, 0.73, 0.39), (0.26, 0.50, 0.29, 0.51, 0.33, 0.60), 1.047),
         "g38_hist48_s0": ("S117", (0.061, None, 0.064, 2.30, None), (0.81, 0.51, 0.77, 0.47, 0.74, 0.40), (0.28, 0.51, 0.30, 0.54, 0.36, 0.61), 1.035),
         "g39_goalcentre_s0": ("S118", (0.070, 3.6, 0.025, 0.19, 9.9), (0.75, 0.49, 0.73, 0.44, 0.70, 0.37), (0.32, 0.50, 0.34, 0.53, 0.38, 0.59), 1.063),
         "g39_goalcentre_s1": ("S118", (0.076, 4.6, 0.035, 0.07, None), (0.73, 0.49, 0.70, 0.44, 0.64, 0.38), (0.31, 0.47, 0.33, 0.52, 0.37, 0.57), None),
         "g39_goalcentre_s2": ("S118", (0.093, 4.9, 0.026, 0.15, None), (0.77, 0.50, 0.74, 0.45, 0.69, 0.39), (0.31, 0.48, 0.32, 0.52, 0.36, 0.59), None),
         "g40_hist24_gc_s0": ("S120", (0.062, 4.9, 0.031, 0.01, 10.0), (0.79, 0.55, 0.77, 0.48, 0.74, 0.42), (0.27, 0.47, 0.28, 0.51, 0.34, 0.59), 1.038),
         "g40_hist24_gc_s1": ("S120 addendum", (0.062, 6.2, 0.034, 0.01, None), (0.79, 0.51, 0.76, 0.46, 0.72, 0.39), (0.29, 0.48, 0.30, 0.53, 0.34, 0.60), 1.045),
         "g40_hist24_gc_s2": ("S121", (0.066, 5.5, None, 0.06, None), (0.80, 0.52, 0.77, 0.47, 0.73, 0.40), (0.27, 0.48, 0.30, 0.53, 0.36, 0.60), 1.026)}
    for run, (cid, gt, gl, vl, a09) in L.items():
        r = S["hg"][run]
        for q, l, tol in zip(("fall", "err", "hold", "rev", "over"), gt, (0.0006, 0.06, 0.0006, 0.006, 0.6)):
            if l is not None:
                check(cid, "Table 42", f"{run} goal test {q}", r[q], l, tol)
        if gl:
            for k_, l in zip(("1.u", "1.d", "1.75.u", "1.75.d", "2.5.u", "2.5.d"), gl):
                check(cid, "Table 42", f"{run} battery goal {k_}", r["goal"][k_], l, 0.006)
        if vl:
            for k_, l in zip(("1.u", "1.d", "1.75.u", "1.75.d", "2.5.u", "2.5.d"), vl):
                check(cid, "Table 42", f"{run} velocity falls {k_}", r["vel"][k_], l, 0.006)
        if a09:
            check(cid, "Table 42", f"{run} flat 0.9 m/s achieved", r["ach"]["0.9"], a09, 0.0006)
    # the S117 numbers were read from the origin-basis file of g38 s0 (before S118); check those against the same file
    m0 = json.load(open(WP / "g38_hist24_s0/motion_goal_ref.json"))
    check("S117", "text", "g38_hist24_s0 goal test falls (origin file)", m0["fall"], 0.064, 0.0006)
    check("S117", "text", "g38_hist24_s0 reversals (origin file)", m0["cmd_reversals_per_world"], 1.00, 0.006)
    check("S117", "text", "g38_hist24_s0 hard designs (origin file)", float(np.mean([m0["by_design"][k] for k in HARD])), 0.22, 0.006)
    for run, l, cid in (("g39_goalcentre_s0", 0.26, "S120"), ("g40_hist24_gc_s0", 0.23, "S120"), ("g38_hist24_s1", 0.22, "S120"), ("g40_hist24_gc_s1", 0.23, "S120 addendum"), ("g40_hist24_gc_s2", 0.22, "S121"), ("g38_hist48_s0", 0.22, "S117")):
        check(cid, "Table 42", f"{run} hard-design falls", S["hg"][run]["hard"], l, 0.006)
    check("S118", "text", "g39 s0 arrived", S["hg"]["g39_goalcentre_s0"]["arrived"], 0.94, 0.006)
    check("S118", "text", "g39 s0 flat 0.3 m/s achieved", S["hg"]["g39_goalcentre_s0"]["ach"]["0.3"], 0.996, 0.0006)
    # S117 pull sweep on a flat map (eval_tiles2.py, TILES_FLAT=1, 352 worlds): none / pulls 1.5 to 6 N / pulls 3 to 12 N
    S["hg_pull"] = {}
    for run, logged in (("g34_mref_rsad_db_s0", (0.074, 0.139, 0.361)), ("g38_hist24_s0", (0.055, 0.118, 0.353)), ("g38_hist48_s0", (0.043, 0.116, 0.351))):
        vals = []
        for (key, f_), l in zip((("none", "eval2_goal_goal_dr_surf_ps0_flat_noexit.json"), ("half", "eval2_goal_goal_dr_surf_pull_ps0.5_flat_noexit.json"), ("full", "eval2_goal_goal_dr_surf_pull_flat_noexit.json")), logged):
            d = json.load(open(WP / run / f_))
            assert d["env_settings"]["tiles_flat"] and d["env_settings"]["no_exit"], (run, f_)
            V[f"hgp.{run}.{key}"] = d["overall"]["fall"]
            vals.append(d["overall"]["fall"])
            check("S117", "text", f"{run} flat-map falls, {key}", d["overall"]["fall"], l, 0.0006)
        S["hg_pull"][run] = vals

    # navigation (S120 addendum, S121): eval_nav.py as Table 39, positions from env.ball_xy() (true centre for goal-centre runs)
    NAV2 = [("g34", "g34_mref_rsad_db_s0", "", "g34 s0 (S113, origin logic; Table 39)"), ("g38", "g38_hist24_s0", "", "g38 s0, 240 ms history, origin logic"),
            ("g39s0", "g39_goalcentre_s0", "", "<b>g39 s0, goal centre (recommended)</b>"), ("g39s1", "g39_goalcentre_s1", "", "<b>g39 s1</b>"), ("g39s2", "g39_goalcentre_s2", "", "<b>g39 s2</b>"),
            ("g40", "g40_hist24_gc_s0", "", "g40 s0, 240 ms history + goal centre"), ("g40o", "g40_hist24_gc_s0", "_origin", "g40 s0, judged at the body origin (NAV_ORIGIN=1)")]
    rows, S["nav2"] = [], {}
    for key, run, suf, lab in NAV2:
        cells = {}
        for src in ("tof18", "tof18_pp", "oracle"):
            f = WP / run / f"navS_{src}{suf}.json"
            if not f.exists():
                cells[src] = None
                continue
            d = json.load(open(f))
            assert d["run"] == run and d["worlds"] == 528 and d["steps"] == 2000 and d["planner"]["kind"] == "smooth" and d.get("motion_ref") and d["push_scale"] == (1.0 if src.endswith("_pp") else 0.0), f
            cells[src] = d["overall"]
            for q in ("success", "fall", "stuck", "heading_rate_rms"):
                V[f"n2.{key}.{src}.{q}"] = d["overall"][q]
        S["nav2"][key] = cells
        t, p, o = cells["tof18"], cells["tof18_pp"], cells["oracle"]
        rows.append([lab, f"{t['success']:.3f} / {t['stuck']:.3f} / {t['fall']:.3f}", (f"{p['success']:.3f} / {p['fall']:.3f}" if p else "not run"),
                     (f"{o['success']:.3f} / {o['stuck']:.3f} / {o['fall']:.3f}" if o else "not run")])
    for q in ("success", "fall", "stuck"):
        for src in ("tof18", "tof18_pp", "oracle"):
            xs = [V[f"n2.{k}.{src}.{q}"] for k in ("g39s0", "g39s1", "g39s2")]
            V[f"n2f.g39.{src}.{q}.min"], V[f"n2f.g39.{src}.{q}.max"] = min(xs), max(xs)
    head = ["Controller", "ToF 18 x 14: reached / stuck / fell", "ToF, pushes and pulls: reached / fell", "Perfect map: reached / stuck / fell"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on at scale 1 (11 designs, floor classes)", noise="on (sensor ranges: noise 0.5 percent of range, 2 percent dropout)",
                    worlds="528 (48 per design, 11 designs)", seeds="one controller seed per row, frozen; three seeds of g39",
                    protocol="eval_nav.py as Table 39 (blocks map, goal 1.2 m away behind an 8 cm block, 20 s, smooth follower, motion layer in the loop). Since S120 eval_nav.py takes the robot position from env.ball_xy(): the true ball centre for goal-centre runs (g39, g40), the body origin otherwise, which makes the arrival criterion about 0.02 stricter for the goal-centre runs; the last row forces the origin. Files outputs/study/warp_ppo/RUN/navS_{tof18,tof18_pp,oracle}[_origin].json.")
    inject("generalist.html", "tab_nav2", html_table(head, rows) + f'<p class="cap"><b>Table 43.</b> Navigation with the goal-centre and 240 ms controllers. {cap}</p>')
    tex_table("nav2", "Navigation with the goal-centre and 240 ms controllers (S120, S121). " + tex_escape(cap), "tab:gen_nav2", head, rows, colspec="p{5cm}ccc", star=True)
    NL = [("g39s0", "S121", (0.866, 0.077, 0.057), (0.544, 0.413), (0.953, 0.001, 0.045)), ("g39s1", "S121", (0.867, 0.076, 0.057), (0.545, 0.405), (0.949, 0.006, 0.045)),
          ("g39s2", "S121", (0.892, 0.047, 0.061), (0.487, 0.460), (0.938, 0.008, 0.055)), ("g40", "S121", (0.824, 0.112, 0.064), (0.453, 0.472), (0.907, 0.028, 0.064)),
          ("g40o", "S120 addendum", (0.847, 0.097, 0.057), (0.447, 0.491), None), ("g38", "S120 addendum", (0.854, 0.083, 0.063), (0.485, 0.449), None),
          ("g34", "S121", (0.877, 0.044, 0.080), (0.557, 0.413), (0.938, 0.002, 0.061))]
    for key, cid, t, p, o in NL:      # 3-decimal log values; tolerance one unit in the third decimal (the log rounds half-way cases either way)
        for q, l in zip(("success", "stuck", "fall"), t):
            check(cid, "Table 43", f"{key} ToF {q}", V[f"n2.{key}.tof18.{q}"], l, 0.0011)
        for q, l in zip(("success", "fall"), p):
            check(cid, "Table 43", f"{key} ToF pushes {q}", V[f"n2.{key}.tof18_pp.{q}"], l, 0.0011)
        if o:
            for q, l in zip(("success", "stuck", "fall"), o):
                check(cid, "Table 43", f"{key} perfect map {q}", V[f"n2.{key}.oracle.{q}"], l, 0.0011)

    # S119 design sweep (robot_spec.py, one property at a time around specs/nominal.yaml, g34 s0, 256 worlds per case)
    SPEC = ROOT / "outputs/study/specs"
    sim = {r["name"]: r for r in csv.DictReader(open(SPEC / "ota_sim.csv"))}
    nom = next(csv.DictReader(open(SPEC / "spec_nominal_sim.csv")))
    sim["spec_nominal"] = nom
    S["ota"] = {}
    rows = [["<b>nominal robot</b> (specs/nominal.yaml)", "ball 9 cm, 0.168 kg; body 3.85 kg, COM 0.131 m; 30 N m; rotor 0.005 kg m<sup>2</sup>; contact 51 deg"] + [f"{float(nom[c]):.3f}" for c, _ in OTA_COLS]]
    for c, short in OTA_COLS:
        V[f"ota.nom.{short.replace(' ', '_')}"] = float(nom[c])
    S["ota"]["spec_nominal"] = {short: float(nom[c]) for c, short in OTA_COLS}
    for name, prop, val in OTA:
        r = sim[name]
        S["ota"][name] = dict(prop=prop, value=val, **{short: float(r[c]) for c, short in OTA_COLS}, pull5=float(r["phys_pull_capacity_5deg_N"]))
        for c, short in OTA_COLS:
            V[f"ota.{name}.{short.replace(' ', '_')}"] = float(r[c])
        rows.append([f"{prop} {val.replace('^2', '<sup>2</sup>')}", f"nominal {OTA_NOM[prop].replace('^2', '<sup>2</sup>')}"] + [f"{float(r[c]):.3f}" for c, _ in OTA_COLS])
    V["ota.pull5.nom"] = float(nom["phys_pull_capacity_5deg_N"])
    V["ota.pull5.bodymas_2"] = float(sim["ota_bodymas_2"]["phys_pull_capacity_5deg_N"])
    lq = {}
    for x in ("ballrad_0p07", "ballrad_0p12", "rotorine_0p001", "motortor_3"):
        for suf, k in (("", "full"), ("_lqronly", "lqr")):
            d = json.load(open(WP / "g34_mref_rsad_db_s0" / f"eval2_goal_goal_dr_surf_ps0_des-ota_{x}-ota_{x}_flat_noexit{suf}.json"))
            assert d["env_settings"]["tiles_flat"] and d["env_settings"]["push_scale"] == 0.0, x
            lq[(x, k)] = d["overall"]["fall"]
            V[f"otaq.{x}.{k}"] = d["overall"]["fall"]
    S["ota_lqr"] = {f"{x}.{k}": v for (x, k), v in lq.items()}
    head = ["Variation (one property at a time)", "Nominal value", "Falls: standing at the goal", "with short pushes", "with half-strength pulls (1.5 to 6 N)", "driving at 0.3 m/s", "goals on terrain tiles"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01", dr="on (standard ranges)", noise="on", worlds="256 per case", seeds="one controller checkpoint (g34_mref_rsad_db_s0, design-conditioned; every spec is a design it never trained on)",
                    protocol="scripts/study/robot_spec.py maps a spec in physical units onto the parametric generator of design_space.py and, with --eval, runs the controller; cases on a fully flat map except the last column: goal hold without disturbance, with the short pushes, with half-strength sustained pulls, driving at 0.3 m/s, and goals on the standard terrain tiles. Files outputs/study/specs/ota_sim.csv and spec_nominal_sim.csv.")
    inject("design.html", "tab_ota", html_table(head, rows) + f'<p class="cap"><b>Table 1.</b> The design sweep. {cap}</p>')
    tex_table("ota", "Design sweep: one property at a time around the nominal robot (S119). " + tex_escape(cap), "tab:gen_ota", head, [[re.sub(r"<[^>]+>", " ", str(c)) for c in r] for r in rows], colspec="p{4.4cm}cccccc", star=True)
    OL = {"spec_nominal": (0.007, 0.007, 0.017, 0.001, 0.138),
          "ota_ballmas_0p042": (0.41, 0.41, 0.46, 0.39, 0.53), "ota_ballmas_0p084": (0.052, 0.052, 0.070, 0.006, 0.19), "ota_ballmas_0p34": (0.000, 0.000, 0.012, 0.000, 0.13), "ota_ballmas_0p67": (0.000, 0.000, 0.013, 0.000, 0.14),
          "ota_bodycom_0p08": (0.037, 0.031, 0.051, 0.010, 0.20), "ota_bodycom_0p1": (0.020, 0.021, 0.031, 0.004, 0.17), "ota_bodycom_0p17": (0.004, 0.004, 0.018, 0.000, 0.13), "ota_bodycom_0p22": (0.003, 0.001, 0.013, 0.000, 0.125),
          "ota_bodymas_2": (0.002, 0.003, 0.243, 0.000, 0.14), "ota_bodymas_3": (0.004, 0.007, 0.034, 0.000, 0.14), "ota_bodymas_5": (0.009, 0.013, 0.019, 0.001, 0.15), "ota_bodymas_6p5": (0.021, 0.018, 0.020, 0.001, 0.15),
          "ota_motortor_3": (0.48, 0.48, 0.51, 0.001, 0.54), "ota_motortor_6": (0.13, 0.13, 0.12, 0.000, 0.23), "ota_motortor_10": (0.044, 0.052, 0.067, 0.000, 0.18), "ota_motortor_15": (0.009, 0.011, 0.041, 0.000, 0.145),
          "ota_rotorine_0p001": (0.43, 0.44, 0.47, 0.37, 0.54), "ota_rotorine_0p02": (0.000, 0.000, 0.014, 0.000, 0.109),
          "ota_contactzen_41": (0.015, 0.015, 0.034, 0.004, 0.14), "ota_contactzen_61": (0.025, 0.028, 0.031, 0.006, 0.17),
          "ota_ballrad_0p07": (0.11, 0.11, 0.11, 0.025, 0.26), "ota_ballrad_0p12": (0.70, 0.70, 0.71, 0.76, 0.73)}
    for name, logged in OL.items():
        for (c, short), l in zip(OTA_COLS, logged):
            nd = len(str(l).split(".")[1]) if "." in str(l) else 0
            check("S119", "design.html Table 1", f"{name} {short}", float(sim[name][c]), l, 0.5 * 10 ** (-max(nd, 2)) + 1e-4)
    for (x, k), l in ((("ballrad_0p12", "full"), 0.69), (("ballrad_0p12", "lqr"), 0.022), (("ballrad_0p07", "full"), 0.11), (("ballrad_0p07", "lqr"), 0.28),
                      (("rotorine_0p001", "full"), 0.45), (("rotorine_0p001", "lqr"), 0.49), (("motortor_3", "full"), 0.48), (("motortor_3", "lqr"), 0.46)):
        check("S119", "design.html", f"LQR-only check {x} {k}", lq[(x, k)], l, 0.006 if l >= 0.1 else 0.0006)
    check("S119", "design.html", "pull capacity at 5 deg, nominal [N]", V["ota.pull5.nom"], 3.3, 0.06)
    check("S119", "design.html", "pull capacity at 5 deg, 2 kg body [N]", V["ota.pull5.bodymas_2"], 1.7, 0.06)


def fig_histgc(C, size):
    """Hold precision against disturbed navigation, per seed: the S120/S121 trade-off."""
    fig, ax = plt.subplots(figsize=size)
    pts = [("g34", "g34_mref_rsad_db_s0", "g34", C["model"], "o"), ("g38", "g38_hist24_s0", "g38", C["mute"], "s"),
           ("g39", "g39_goalcentre_s0", "g39s0", C["hybrid"], "D"), ("g39", "g39_goalcentre_s1", "g39s1", C["hybrid"], "D"), ("g39", "g39_goalcentre_s2", "g39s2", C["hybrid"], "D"),
           ("g40", "g40_hist24_gc_s0", "g40", C["learned"], "^")]
    lab = {"g34": "g34: origin goal logic", "g38": "g38: + 240 ms history", "g39": "g39: goal centre (recommended)", "g40": "g40: goal centre + 240 ms"}
    seen = set()
    for fam, run, nk, col, mk in pts:
        x = S["hg"][run]["err"]; y = V[f"n2.{nk}.tof18_pp.success"]
        ax.scatter([x], [y], s=70, color=col, marker=mk, edgecolor=C["fg"], linewidth=0.6, zorder=3, label=None if fam in seen else lab[fam])
        seen.add(fam)
    ax.set_xlabel("goal hold error RMS at the true ball centre [cm] (lower is better)")
    ax.set_ylabel("ToF navigation with pushes and pulls:\nshare of 528 worlds reaching the goal")
    ax.set_xlim(0, 15); ax.set_ylim(0.40, 0.60)
    ax.annotate("better", xy=(1.0, 0.585), xytext=(4.0, 0.585), color=C["mute"], fontsize=8, va="center", arrowprops=dict(arrowstyle="->", color=C["mute"]))
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title("one point per seed: precise holding (left) and robust pushed navigation (top)")
    fig.tight_layout()
    return fig


def fig_ota(C, size):
    """Design sweep small multiples: falls standing at the goal, with half-strength pulls and on terrain goals, per property (categorical x, nominal marked)."""
    props = [("ball mass", "ball mass [kg]", [("ota_ballmas_0p042", "0.042"), ("ota_ballmas_0p084", "0.084"), ("spec_nominal", "0.168"), ("ota_ballmas_0p34", "0.34"), ("ota_ballmas_0p67", "0.67")]),
             ("wheel torque limit", "torque limit [N m]", [("ota_motortor_3", "3"), ("ota_motortor_6", "6"), ("ota_motortor_10", "10"), ("ota_motortor_15", "15"), ("spec_nominal", "30")]),
             ("rotor inertia", "rotor inertia [kg m$^2$]", [("ota_rotorine_0p001", "0.001"), ("spec_nominal", "0.005"), ("ota_rotorine_0p02", "0.02")]),
             ("body COM height", "COM above ball centre [m]", [("ota_bodycom_0p08", "0.08"), ("ota_bodycom_0p1", "0.10"), ("spec_nominal", "0.131"), ("ota_bodycom_0p17", "0.17"), ("ota_bodycom_0p22", "0.22")]),
             ("body mass", "body mass [kg]", [("ota_bodymas_2", "2.0"), ("ota_bodymas_3", "3.0"), ("spec_nominal", "3.85"), ("ota_bodymas_5", "5.0"), ("ota_bodymas_6p5", "6.5")]),
             ("ball radius", "ball radius [cm]", [("ota_ballrad_0p07", "7"), ("spec_nominal", "9"), ("ota_ballrad_0p12", "12")])]
    fig, axs = plt.subplots(2, 3, figsize=size, sharey=True)
    for ax, (prop, xl, pts) in zip(axs.ravel(), props):
        xs = list(range(len(pts)))
        for key, col, lab in (("still", C["hybrid"], "standing at the goal"), ("half pulls", C["warn"], "half-strength pulls"), ("terrain", C["bad"], "goals on terrain")):
            ax.plot(xs, [S["ota"][p[0]][key] for p in pts], "-o", color=col, ms=4, lw=1.6, label=lab)
        ax.axvline([i for i, p in enumerate(pts) if p[0] == "spec_nominal"][0], color=C["mute"], lw=0.8, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([p[1] for p in pts], fontsize=8)
        ax.set_xlabel(xl, fontsize=8); ax.set_title(prop, fontsize=9)
        ax.set_ylim(-0.02, 0.8)
    axs[0, 0].set_ylabel("share of 256 robots that fall"); axs[1, 0].set_ylabel("share of 256 robots that fall")
    axs[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def fig_nav(C, size):
    fig, ax = plt.subplots(figsize=size)
    labs = ["blind", "perfect map", "ToF 18x14\nhead -25", "camera 32x24\nhead -25", "camera 32x24\nhead -40", "ToF 8x8\nhead -25", "ToF 18x14\nlow -10", "2 ToF on\nbody cones"]
    xs = np.arange(len(NAV_CFG))
    su = [S["nav"][st]["success"] for st, _ in NAV_CFG]; fa = [S["nav"][st]["fall"] for st, _ in NAV_CFG]
    ax.bar(xs - 0.2, su, 0.38, color=C["hybrid"], label="reached goal")
    ax.bar(xs + 0.2, fa, 0.38, color=C["bad"], label="fell")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7); ax.set_ylim(0, 1); ax.set_ylabel("share of 528 worlds"); ax.legend(fontsize=7, ncol=2, loc="upper center")
    ax.set_title("same frozen balance controller and planner, different hazard sources (undisturbed)")
    fig.tight_layout()
    return fig


def build_s92_s93():
    D1 = WP / "g18_ret_rs025_s1"
    J_ = lambda n: json.load(open(D1 / n))
    designs = ["nom", "zhigh", "zlow", "J02", "tau6", "bal4", "bal025", "kb4", "kb025", "zeta41", "zeta66"]
    # artefact demonstration
    art = J_("eval2_vel_0.3_dr_surf_ps0_drs1_flat_noexit.json")["overall"]
    V["art.03.rec"], V["art.03.edge"] = art["fall"], art["fall_at_map_edge"]
    V["art.03.real"] = art["fall"] * (1 - art["fall_at_map_edge"])
    f09 = D1 / "eval2_vel_0.9_ps0_flat_noexit.json"
    if f09.exists() and "fall_at_map_edge" in json.load(open(f09))["by_kind"]["flat"]:
        o9 = json.load(open(f09))["by_kind"]["flat"]
        V["art.09.rec"], V["art.09.edge"] = o9["fall"], o9["fall_at_map_edge"]
    else:
        V["art.09.rec"], V["art.09.edge"] = 0.25, 0.97      # S92 (log only)
    for sp, (e, r, fl) in {"0.3": (0.014, 1.00, 0.00), "0.6": (0.017, 1.00, 0.00), "0.9": (None, None, 0.89)}.items():
        c = S["flat"]["LQR"][sp]
        check("S92 addendum", "Table 1 LQR alone", f"falls @{sp}", c["fall"], fl, 0.006, "" if sp != "0.9" else "lqr_flat_interior.log prints 0.91 (error 0.026, achieved 1.00 over survivors); the S92 addendum text says 0.89; the file is used")
        if e is not None and c["err"] is not None:
            check("S92 addendum", "Table 1 LQR alone", f"err @{sp}", c["err"], e, 0.0006)
            check("S92 addendum", "Table 1 LQR alone", f"achieved @{sp}", c["ratio"], r, 0.006)
    check("S92", "methods note", "0.3 m/s recorded falls (border tiles)", V["art.03.rec"], 0.33, 0.006)
    check("S92", "methods note", "share at the map edge", V["art.03.edge"], 0.75, 0.011)
    # (the 20 s base-rate table, the LQR gain sweep and the seed table that followed here were withdrawn: S98, standing blow-up; see build_s98)


SCALES_R = [("s0.5", 4.5, "0.5 (held out, not trained)"), ("s0.75", 6.75, "0.75"), ("nom", 9.0, "1 (nominal)"), ("s1.5", 13.5, "1.5"), ("s2.0", 18.0, "2")]
TRAINED_R = [("s0.75", 6.75), ("nom", 9.0), ("s1.5", 13.5), ("s2.0", 18.0)]
SCALE_RUN = "g24_scale_rs025_s0"


def _bins(pw, kind, edges, nmin=8):
    """Fall probability per height bin from the per-world arrays of a stepsize_*.json (heights in m, edges in m)."""
    k, sv, fl = np.array(pw["kind"]), np.array(pw["severity"], dtype=float), np.array(pw["fell"], dtype=float)
    m = k == kind
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mm = m & (sv >= lo) & (sv < hi)
        out.append((lo, hi, int(mm.sum()), float(fl[mm].mean()) if mm.sum() >= nmin else None))
    return out


def _p50(curve):
    """Height (m) where the binned fall curve first crosses 0.5, by linear interpolation between bin centres; None if it never does."""
    pts = [((lo + hi) / 2, p) for lo, hi, n, p in curve if p is not None]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 < 0.5 <= y1:
            return x0 + (0.5 - y0) / (y1 - y0) * (x1 - x0)
    return pts[0][0] if pts and pts[0][1] >= 0.5 else None


def build_s94_s95():
    # (S94 standing table withdrawn: S98 standing blow-up; see build_s98)
    # ---- S95: ball size against step size (one recipe trained on 14 designs, two training seeds, evaluated at five scales)
    SCALE_SEEDS = [r for r in (SCALE_RUN, "g24_scale_rs025_s1") if (WP / r / "stepsize_mu01_nom.json").exists()]
    SS, SS_OLD = {}, {}      # SS[d][run] = stepsize json at free-rolling friction 0.01 (S98/S99); SS_OLD = the same evaluation at 0.001 (first pass), kept only for the change sentence
    for d, r_cm, lab in SCALES_R:
        SS[d] = {r: json.load(open(WP / r / f"stepsize_mu01_{d}.json")) for r in SCALE_SEEDS if (WP / r / f"stepsize_mu01_{d}.json").exists()}
        SS_OLD[d] = {r: json.load(open(WP / r / f"stepsize_{d}.json")) for r in SCALE_SEEDS if (WP / r / f"stepsize_{d}.json").exists()}
        V[f"sc.{d}.mu01"] = 1.0 if SS[d] else 0.0
        if not SS[d]:      # the held-out 0.5 scale has no 0.01 file yet: shown at 0.001, marked
            SS[d] = SS_OLD[d]
        V[f"sc.{d}.r"] = r_cm
        V[f"sc.{d}.hmax"] = r_cm * (1 - math.cos(math.radians(45)))
        V[f"sc.{d}.nseeds"] = len(SS[d])
        for i, (r, j) in enumerate(SS[d].items()):
            V[f"sc.{d}.all.{i}"], V[f"sc.{d}.flat.{i}"] = j["overall"]["fall"], j["by_kind"]["flat"]["fall"]
            for k in ("step_up", "bump", "stairs", "large_drop"):
                V[f"sc.{d}.{k}.s.{i}"], V[f"sc.{d}.{k}.l.{i}"] = j["by_severity"][k]["small"]["fall"], j["by_severity"][k]["large"]["fall"]
        for key in ["all", "flat"] + [f"{k}.{h}" for k in ("step_up", "bump", "stairs", "large_drop") for h in "sl"]:
            V[f"sc.{d}.{key}"] = float(np.mean([V[f"sc.{d}.{key}.{i}"] for i in range(len(SS[d]))]))      # mean over seeds
        if V[f"sc.{d}.mu01"]:      # change from the first pass at 0.001 (same seeds), mean over seeds
            V[f"sc.{d}.d_all"] = float(np.mean([j["overall"]["fall"] for j in SS_OLD[d].values()])) - V[f"sc.{d}.all"]
            V[f"sc.{d}.d_step"] = float(np.mean([j["by_severity"]["step_up"]["large"]["fall"] for j in SS_OLD[d].values()])) - V[f"sc.{d}.step_up.l"]
            V[f"sc.{d}.d_step_seedmax"] = max(SS_OLD[d][r]["by_severity"]["step_up"]["large"]["fall"] - SS[d][r]["by_severity"]["step_up"]["large"]["fall"] for r in SS[d])      # largest per-seed change
            V[f"sc.{d}.d_bump"] = float(np.mean([j["by_severity"]["bump"]["large"]["fall"] for j in SS_OLD[d].values()])) - V[f"sc.{d}.bump.l"]
    V["sc.d_small_max"] = max(abs(V[f"sc.{d}.d_all"]) for d in ("s0.75", "nom"))
    V["sc.d_large_min"], V["sc.d_large_max"] = min(V[f"sc.{d}.d_all"] for d in ("s1.5", "s2.0")), max(V[f"sc.{d}.d_all"] for d in ("s1.5", "s2.0"))
    V["sc.d_step_max"] = max(V[f"sc.{d}.d_step_seedmax"] for d in ("s1.5", "s2.0"))      # "up to", over seeds and the two large sizes
    bs = SS["nom"][SCALE_RUN]["by_severity"]
    rng = lambda k: f"{bs[k]['sev_min'] * 100:.1f} to {bs[k]['split'] * 100:.1f} / {bs[k]['split'] * 100:.1f} to {bs[k]['sev_max'] * 100:.1f} cm"
    head = ["Robot scale", "Ball radius", "Predicted limit 0.29 r", f"Step up, falls: {rng('step_up')}", f"Bump: {rng('bump')}", f"Stairs riser: below / above {bs['stairs']['split'] * 100:.1f} cm", f"Large drop: {rng('large_drop')}", "Flat", "All ten tile kinds", "All ten tile kinds, per seed"]
    rows = []
    for d, r_cm, lab in SCALES_R:
        per = " | ".join(f"{V[f'sc.{d}.all.{i}']:.2f}" for i in range(len(SS[d]))) + ("" if len(SS[d]) > 1 else " (seed 0 only)")
        rows.append([lab + ("" if V[f"sc.{d}.mu01"] else "; at friction 0.001 only"), f"{r_cm:g} cm", f"{V[f'sc.{d}.hmax']:.1f} cm"] + [f"{V[f'sc.{d}.{k}.s']:.2f} / {V[f'sc.{d}.{k}.l']:.2f}" for k in ("step_up", "bump", "stairs", "large_drop")] + [f"{V[f'sc.{d}.flat']:.2f}", f"{V[f'sc.{d}.all']:.2f}", per])
    n_kind = sum(1 for k in SS["nom"][SCALE_RUN]["per_world"]["kind"] if k == "step_up")
    V["sc.nworlds"] = n_kind
    V["sc.nseeds"] = len(SCALE_SEEDS)
    seedtxt = (f"two training seeds of one recipe ({', '.join(SCALE_SEEDS)}), the then-recommended recipe trained on the 11 designs plus the robot at 0.75, 1.5 and 2 times nominal size (14 designs); cells are the mean of the two seeds (the last column gives each seed); the 0.5 scale was never trained on and was evaluated with seed 0 only, at friction 0.001"
               if len(SCALE_SEEDS) > 1 else f"one policy ({SCALE_RUN}, seed 0), the then-recommended recipe trained on the 11 designs plus the robot at 0.75, 1.5 and 2 times nominal size (14 designs); the 0.5 scale was never trained on")
    cap = setup_cap(engine=ENG, dr="on at scale 1 (floor friction 0.4 to 1.2 by surface class)", noise="on", worlds=f"{n_kind} per terrain kind, robot scale and seed ({len(SS['nom'][SCALE_RUN]['per_world']['kind'])} per scale and seed), split into halves at the midpoint of each feature's drawn range",
                    seeds=seedtxt, protocol="velocity task at 0.3 m/s Froude-scaled (times the square root of the scale) on the tiles, undisturbed, arrestors re-fitted per scale, 20 s episodes, tile seed 0; feature sizes are the same absolute sizes for every robot (eval_governor.py --source none --eval-designs); evaluated at wheel-ball free-rolling friction 0.01 (S98) with policies trained at 0.001 (files stepsize_mu01_*.json).")
    inject("generalist.html", "tab_scale5", html_table(head, rows) + f'<p class="cap"><b>Table 30.</b> Five robot sizes against the same obstacles. {cap}</p>')
    tex_table("scale5", "Five robot sizes against the same obstacles (falls on the smaller / larger half of each feature range, mean of two training seeds; 0.5 scale one seed). " + tex_escape(cap), "tab:gen_scale5", head, rows, colspec="lccccccccc", star=True)
    # S99 addendum (friction 0.01): overall, large-half step up, large-half bump; seed 0 | seed 1
    L = {"s0.75": ((0.46, 0.46), (1.00, 1.00), (1.00, 0.98)), "nom": ((0.31, 0.34), (0.93, 0.95), (0.95, 0.95)), "s1.5": ((0.20, 0.21), (0.56, 0.65), (0.48, 0.52)), "s2.0": ((0.13, 0.13), (0.56, 0.60), (0.42, 0.41))}
    for d, (al, su, bu) in L.items():
        for i in range(2):
            if f"sc.{d}.all.{i}" in V:
                check("S99 addendum", f"Table 30 (seed {i}, friction 0.01)", f"{d} all tiles", V[f"sc.{d}.all.{i}"], al[i], 0.006)
                check("S99 addendum", f"Table 30 (seed {i}, friction 0.01)", f"{d} step up larger", V[f"sc.{d}.step_up.l.{i}"], su[i], 0.006)
                check("S99 addendum", f"Table 30 (seed {i}, friction 0.01)", f"{d} bump larger", V[f"sc.{d}.bump.l.{i}"], bu[i], 0.006)
    # the first-pass (0.001) numbers of the S95 entry and addenda are no longer shown; the 0.5 scale row still is (S95 first table, seed 0)
    for k, (a_, b_) in {"step_up": (0.90, 1.00), "bump": (0.98, 1.00), "stairs": (0.94, 1.00), "large_drop": (0.98, 0.98)}.items():
        check("S95", "Table 30 (0.5 scale, friction 0.001)", f"s0.5 {k} smaller / larger", V[f"sc.s0.5.{k}.s"], a_, 0.006)
        check("S95", "Table 30 (0.5 scale, friction 0.001)", f"s0.5 {k} larger", V[f"sc.s0.5.{k}.l"], b_, 0.006)
    check("S95", "Table 30 (0.5 scale, friction 0.001)", "s0.5 all tiles", V["sc.s0.5.all"], 0.95, 0.006)
    check("S99 addendum", "text", "large robots lose overall, min", V["sc.d_large_min"], 0.05, 0.011)
    check("S99 addendum", "text", "large robots lose overall, max", V["sc.d_large_max"], 0.06, 0.011)
    check("S99 addendum", "text", "large robots lose at large steps, up to", V["sc.d_step_max"], 0.15, 0.011)
    # 1 cm bins (Table 31) and 0.5 cm bins (Figure 10): worlds of both seeds pooled (same tile seed, so the same feature heights per world), seed spread kept for the figure
    def pooled(d):
        pw = {"kind": [], "severity": [], "fell": []}
        for j in SS[d].values():
            for k in pw:
                pw[k] += list(j["per_world"][k])
        return pw
    e1 = np.arange(0.01, 0.0651, 0.01)
    rows, ns = [], []
    for d, r_cm in TRAINED_R:
        pw = pooled(d)
        cs, cb = _bins(pw, "step_up", e1), _bins(pw, "bump", e1[:4])
        ns += [n for lo, hi, n, p_ in cs + cb if p_ is not None]
        for kind, cur in (("step_up", cs), ("bump", cb)):
            for lo, hi, n, p_ in cur:
                if p_ is not None:
                    V[f"scb.{d}.{kind}.{round(lo * 100)}"] = p_
        rows.append([f"{r_cm:g} cm (limit {V[f'sc.{d}.hmax']:.1f} cm)"] + [f"{p_:.2f}" if p_ is not None else "" for lo, hi, n, p_ in cs] + [f"{p_:.2f}" if p_ is not None else "" for lo, hi, n, p_ in cb])
    V["scb.n_min"], V["scb.n_max"] = min(ns), max(ns)
    head = ["Ball radius (predicted limit)"] + [f"step up {lo * 100:.0f} to {hi * 100:.0f} cm" for lo, hi in zip(e1[:-1], e1[1:])] + [f"bump {lo * 100:.0f} to {hi * 100:.0f} cm" for lo, hi in zip(e1[:3], e1[1:4])]
    cap = setup_cap(engine=ENG, dr="on at scale 1 (floor friction 0.4 to 1.2)", noise="on", worlds=f"{min(ns)} to {max(ns)} per bin (the per-world feature heights of Table 30, binned, {'both seeds pooled' if len(SCALE_SEEDS) > 1 else 'one seed'})",
                    seeds=f"{len(SCALE_SEEDS)} training seed{'s' if len(SCALE_SEEDS) > 1 else ''} of the scale recipe ({', '.join(SCALE_SEEDS)})", protocol="as Table 30, evaluated at wheel-ball free-rolling friction 0.01 (S98); fall probability per 1 cm band of feature height.")
    inject("generalist.html", "tab_scalebins", html_table(head, rows) + f'<p class="cap"><b>Table 31.</b> Falls against step and bump height in 1 cm bins, by ball radius. {cap}</p>')
    tex_table("scalebins", "Falls against step and bump height in 1\\,cm bins, by ball radius (both seeds pooled). " + tex_escape(cap), "tab:gen_scalebins", head, rows, colspec="lcccccccc", star=True)
    # (the S95 addendum bins and 18 cm-ball figure values were measured at friction 0.001 and are no longer shown)
    e5s, e5b = np.arange(0.01, 0.0651, 0.005), np.arange(0.01, 0.0551, 0.005)
    S["ballsize_curves"] = {}
    p50s = []
    for d, r_cm in TRAINED_R:
        pw = pooled(d)
        cs, cb = _bins(pw, "step_up", e5s), _bins(pw, "bump", e5b)
        per_seed = {kind: [_bins(j["per_world"], kind, e) for j in SS[d].values()] for kind, e in (("step_up", e5s), ("bump", e5b))}
        S["ballsize_curves"][d] = dict(r_cm=r_cm, seeds=list(SS[d].keys()))
        for kind, cur in (("step_up", cs), ("bump", cb)):
            S["ballsize_curves"][d][kind] = [(round((lo + hi) / 2 * 100, 2), n, p_, [c[i][3] for c in per_seed[kind]]) for i, (lo, hi, n, p_) in enumerate(cur)]
            for lo, hi, n, p_ in cur:
                if p_ is not None:
                    V[f"scf.{d}.{kind}.{lo * 100:.1f}"] = p_
            h50 = _p50(cur)
            if h50 is not None:
                V[f"sc.{d}.p50r.{kind}"] = h50 / (r_cm / 100)
                p50s.append(h50 / (r_cm / 100))
    V["sc.p50r_min"], V["sc.p50r_max"], V["sc.p50r_mean"] = min(p50s), max(p50s), float(np.mean(p50s))
    V["sc.mu_lo_frac"] = 1 - math.cos(math.atan(0.4))      # limit at mu = 0.4, as a fraction of r
    V["sc.rule_lo"], V["sc.rule_hi"] = 1 / (1 - math.cos(math.radians(45))), 1 / V["sc.p50r_mean"]      # ball radius per step height: 3.5 at mu 1, about 5 at the measured 50 percent point (mean over sizes and features)
    note = "the log gives 'roughly 0.18 to 0.22 r' for seed 0 at friction 0.001; here the crossing is found by linear interpolation of the pooled 0.5 cm-bin curves per size and feature (eight curves) at friction 0.01; the pages quote the computed range and mean"
    check("S95 addendum", "text", "50 percent point, smallest (fraction of r)", V["sc.p50r_min"], 0.18, 0.03, note)
    check("S95 addendum", "text", "50 percent point, largest (fraction of r)", V["sc.p50r_max"], 0.22, 0.03, note)
    check("S95 addendum", "text", "50 percent point, mean (fraction of r)", V["sc.p50r_mean"], 0.20, 0.02, note)
    check("S95 addendum", "text", "limit at mu 0.4 (fraction of r)", V["sc.mu_lo_frac"], 0.07, 0.006)
    # the scale policy's standard battery (at friction 0.01 where the file exists) against the recommended checkpoints retrained at 0.01
    rows, mus = [], []
    for i, run in enumerate(SCALE_SEEDS):
        pm = WP / run / "battery_mu0.01.json"
        bsc = json.load(open(pm)) if pm.exists() else battery(run)
        if bsc is None:
            continue
        mus.append(pm.exists())
        V[f"sc.batt.goal_u.{i}"], V[f"sc.batt.goal_d.{i}"] = bsc["goal"]["drs1_none"]["overall"]["success"], bsc["goal"]["drs1_pushes+pulls"]["overall"]["success"]
        V[f"sc.batt.ach_min.{i}"], V[f"sc.batt.ach_max.{i}"] = min(bsc["flat"][sp]["achieved"] for sp in SP), max(bsc["flat"][sp]["achieved"] for sp in SP)
        V[f"sc.batt.falls_max.{i}"] = max(bsc["flat"][sp]["falls"] for sp in SP)
        rows.append([f"scale policy, 14 designs, trained at 0.001<br><small>{run}, seed {run[-1]}, evaluated at {'0.01' if pm.exists() else '0.001'}</small>", f"{V[f'sc.batt.goal_u.{i}']:.2f} / {V[f'sc.batt.goal_d.{i}']:.2f}", " / ".join(f"{bsc['flat'][sp]['achieved']:.2f}" for sp in SP), " / ".join(f"{bsc['flat'][sp]['falls']:.2f}" for sp in SP)])
    nb = len(rows)
    V["sc.batt.mu01"] = 1.0 if mus and all(mus) else 0.0
    for key, fn in (("goal_u", min), ("goal_d", min), ("ach_min", min), ("falls_max", max)):
        V[f"sc.batt.{key}"] = fn(V[f"sc.batt.{key}.{i}"] for i in range(nb))
    V["sc.batt.goal_u_max"], V["sc.batt.goal_d_max"], V["sc.batt.ach_max"] = max(V[f"sc.batt.goal_u.{i}"] for i in range(nb)), max(V[f"sc.batt.goal_d.{i}"] for i in range(nb)), max(V[f"sc.batt.ach_max.{i}"] for i in range(nb))
    for run in ("g26_ret_rs025_mu01_s0", "g26_ret_rs025_mu01_s1", "g26_ret_rs025_mu01_s2"):
        if run in S.get("goal", {}) and run in S.get("flat", {}):
            g, f = S["goal"][run], S["flat"][run]
            rows.append([f"then-recommended checkpoint, 11 designs, trained and evaluated at 0.01<br><small>{run}, seed {run[-1]}</small>", f"{g['u']['success']:.2f} / {g['d']['success']:.2f}", " / ".join(f"{f[sp]['ratio']:.2f}" for sp in SP), " / ".join(f"{f[sp]['fall']:.2f}" for sp in SP)])
    head = ["Policy", "Goal success, undisturbed / pushes and pulls (DR scale 1)", "Flat benchmark, achieved over commanded speed at 0.3 / 0.6 / 0.9 m/s", "Flat benchmark falls at 0.3 / 0.6 / 0.9 m/s"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01 in every row (the scale seeds were trained at 0.001 and evaluated at 0.01 through PAIR_MU1, files battery_mu0.01.json; the then-recommended checkpoints were trained at 0.01)", dr="on at scale 1 for the goal columns (the 11 training designs of the goal task), off for the flat columns (nominal design)", noise="on (goal), off (flat)", worlds="1760 per goal cell, 64 per flat cell", seeds="one row per training seed, as listed",
                    protocol="the standard battery (eval_battery.py): goal task of Table 2 and the clean flat benchmark of Table 1 on interior tiles.")
    inject("generalist.html", "tab_scalebatt", html_table(head, rows) + f'<p class="cap"><b>Table 32.</b> The scale policy keeps the then-recommended controller\'s standard results. {cap}</p>')
    tex_table("scalebatt", "The scale policy on the standard battery at friction 0.01, against the then-recommended checkpoints. " + tex_escape(cap), "tab:gen_scalebatt", head, [[re.sub(r"<[^>]+>", " ", c).replace("  ", " ") for c in r] for r in rows], colspec="p{0.3\\linewidth}ccc", star=True)
    if V["sc.batt.mu01"]:
        for i, (u, d_, a1, a2) in enumerate(((0.74, 0.47, 0.91, 0.98), (0.70, 0.44, 0.89, 0.99))):
            if f"sc.batt.goal_u.{i}" in V:
                check("S99 close-out files", "Table 32", f"scale seed {i} goal undisturbed (battery_mu0.01.json)", V[f"sc.batt.goal_u.{i}"], u, 0.006, "no log row; checked against the file as read on 2026-11-09")
                check("S99 close-out files", "Table 32", f"scale seed {i} goal disturbed", V[f"sc.batt.goal_d.{i}"], d_, 0.006, "as above")
    S.setdefault("pending", {}).pop("g24_scale_rs025_s1", None)
    have_mu = [r for r in SCALE_SEEDS if (WP / r / "battery_mu0.01.json").exists()]
    S.setdefault("pending", {}).pop("scale battery at 0.01", None)
    g27 = "g27_scale_rs025_mu01_s0"
    S["pending"][g27] = "trained; step-size curves being re-run at its own friction 0.01, see S102" if (WP / g27 / "model.pt").exists() else run_progress(g27)
    ptxt = f'<p><span class="pend">PENDING</span> <code>{g27}</code> ({S["pending"][g27]}): the scale recipe retrained on the corrected contact model (friction 0.01); its step-size curves and battery will join this chapter once they are measured at the correct friction. Also pending: the held-out 0.5 scale at 0.01' + ("" if len(have_mu) == len(SCALE_SEEDS) else f", and the 0.01 battery of {len(SCALE_SEEDS) - len(have_mu)} scale seed(s)") + ".</p>"
    inject("generalist.html", "pending_scale", ptxt)


MU_RUNS = [("g18_ret_rs025_s1", "then-recommended controller, seed 1 (trained at 0.001)", "_mu0.01"), ("g18_ret_rs025_s2", "then-recommended controller, seed 2 (trained at 0.001)", "_mu0.01"),
           ("g26_ret_rs025_mu01_s0", "same recipe retrained at 0.01, seed 0", ""), ("g26_ret_rs025_mu01_s1", "same recipe retrained at 0.01, seed 1", ""), ("g26_ret_rs025_mu01_s2", "same recipe retrained at 0.01, seed 2", "")]
MU_DESIGNS = ["all", "kb025", "bal4", "kb4", "bal025", "zeta66", "nom"]


def build_s98():
    """S98/S99: the 20 s standing falls were a float32 blow-up of the 1000:1 anisotropic wheel-ball friction; re-measurement at free-rolling friction 0.01 and the recipe retrained at 0.01."""
    rows, runs = [], []
    for run, lab, suf in MU_RUNS:
        fs, fd = WP / run / f"eval2_vel_0.0_dr_surf_ps0_flat_noexit{suf}.json", WP / run / f"eval2_vel_0.3_dr_surf_ps0_flat_noexit{suf}.json"
        if not (fs.exists() and fd.exists()):
            continue
        st, dr = json.load(open(fs)), json.load(open(fd))
        for j in (st, dr):
            es = j.get("env_settings", {})
            mu = es.get("pair_mu1") if es.get("pair_mu1") is not None else json.load(open(WP / run / "args.json")).get("pair_mu1")
            assert mu == 0.01 and es["tiles_flat"] and es["no_exit"] and j["args"]["steps"] == 2000, run
        runs.append(run)
        g = lambda j, d: j["overall"]["fall"] if d == "all" else j["by_design"][d]["fall"]
        for d in MU_DESIGNS:
            V[f"mu.{run}.stand.{d}"], V[f"mu.{run}.drive.{d}"] = g(st, d), g(dr, d)
        nf = [j["overall"].get("nonfinite") for j in (st, dr)]
        if nf[0] is not None:
            V[f"mu.{run}.nonfinite"] = max(nf)
        V["mu.edge"] = max(V.get("mu.edge", 0), st["overall"]["fall_at_map_edge"], dr["overall"]["fall_at_map_edge"])
        rows.append([f"{lab}<br><small>{run}</small>", " / ".join(f"{g(st, d):.2f}" for d in MU_DESIGNS), " / ".join(f"{g(dr, d):.2f}" for d in MU_DESIGNS), f"{max(nf):.3f}" if nf[0] is not None else "not recorded"])
    for key in ("stand.all", "drive.all", "stand.kb025", "drive.kb025", "drive.bal4", "stand.bal4"):
        vals = [V[f"mu.{r}.{key}"] for r in runs]
        V[f"mu.{key}.min"], V[f"mu.{key}.max"] = min(vals), max(vals)
    for key in ("stand.kb4", "stand.bal025", "drive.kb4", "drive.bal025", "stand.zeta66", "stand.nom"):
        V[f"mu.{key}.max"] = max(V[f"mu.{r}.{key}"] for r in runs)
    g26 = [r for r in runs if r.startswith("g26")]
    V["mu.n_g26"] = len(g26)
    for key in ("stand.all", "drive.all"):
        V[f"mu.g26.{key}.min"], V[f"mu.g26.{key}.max"] = min(V[f"mu.{r}.{key}"] for r in g26), max(V[f"mu.{r}.{key}"] for r in g26)
    V["mu.g26.nonfinite.max"] = max(V.get(f"mu.{r}.nonfinite", 0) for r in g26)
    head = ["Policy", "Standing still for 20 s at friction 0.01, falls: all 11 designs / kb025 / bal4 / kb4 / bal025 / zeta66 / nominal", "Driving at 0.3 m/s for 20 s at 0.01, falls: all / kb025 / bal4 / kb4 / bal025 / zeta66 / nominal", "Non-finite worlds (fraction)"]
    cap = setup_cap(engine=ENG + "; wheel-ball free-rolling friction 0.01 for every row (the first two policies were trained at 0.001 and are evaluated at 0.01 through PAIR_MU1; the retrained ones carry 0.01 in args.json; the value is recorded in each file's env_settings)", dr="on at scale 1", noise="on", worlds="32 per design (352 per cell)",
                    seeds="seeds 1 and 2 of the then-recommended recipe (seed 0 not re-measured) and the seeds of the recipe retrained at 0.01 that are evaluated so far, one evaluation seed each",
                    protocol="flat interior tiles (at least 7.5 m from the map border), 20 s episodes, arrestors on, no pushes or pulls; zero command (standing) or 0.3 m/s at a random heading (driving); no fall was at the map edge; a non-finite state is counted separately where the file records it (new evaluations only) and is 0.000 in every such row.")
    inject("generalist.html", "tab_standing_mu01", html_table(head, rows) + f'<p class="cap"><b>Table 34.</b> Standing and driving for 20 s after the friction correction. {cap}</p>')
    tex_table("standing_mu01", "Standing and driving for 20\\,s on flat interior tiles at wheel-ball free-rolling friction 0.01 (the earlier 20\\,s standing tables are withdrawn, see text). " + tex_escape(cap), "tab:gen_standing_mu01", head,
              [[re.sub(r"<[^>]+>", " ", c).replace("  ", " ") for c in r] for r in rows], colspec="p{0.24\\linewidth}ccc", star=True)
    # S98 addendum table: standing all / kb4 / bal025 / kb025 / bal4; driving all / kb025 / bal4 / nom
    L = {"g18_ret_rs025_s1": ((0.027, 0.00, 0.00, 0.24, 0.05), (0.063, 0.50, 0.18, 0.00)), "g18_ret_rs025_s2": ((0.039, 0.00, 0.00, 0.29, 0.13), (0.075, 0.54, 0.23, 0.02))}
    for run, (a_, b_) in L.items():
        for d, l in zip(("all", "kb4", "bal025", "kb025", "bal4"), a_):
            check("S98 addendum", "Table 34", f"{run} standing {d}", V[f"mu.{run}.stand.{d}"], l, 0.006)
        for d, l in zip(("all", "kb025", "bal4", "nom"), b_):
            check("S98 addendum", "Table 34", f"{run} driving {d}", V[f"mu.{run}.drive.{d}"], l, 0.006)
    # S99 and addendum: g26 s0 standing 0.028 / kb025 0.27 / bal4 0.04, driving 0.062 / 0.48 / 0.20, non-finite 0.000; s1 standing 0.030 / 0.28 / 0.06
    for run, st_, dr_ in (("g26_ret_rs025_mu01_s0", (0.028, 0.27, 0.04), (0.062, 0.48, 0.20)), ("g26_ret_rs025_mu01_s1", (0.030, 0.28, 0.06), None)):
        if run in runs:
            for d, l in zip(("all", "kb025", "bal4"), st_):
                check("S99", "Table 34", f"{run} standing {d}", V[f"mu.{run}.stand.{d}"], l, 0.006)
            if dr_:
                for d, l in zip(("all", "kb025", "bal4"), dr_):
                    check("S99", "Table 34", f"{run} driving {d}", V[f"mu.{run}.drive.{d}"], l, 0.006)
            check("S99", "Table 34", f"{run} non-finite", V.get(f"mu.{run}.nonfinite"), 0.000, 0.0006)
    # the retrained seeds' batteries (S99 addendum and close-out) as ranges, and the checkpoint decision
    bats = {r: battery(r) for r in ("g26_ret_rs025_mu01_s0", "g26_ret_rs025_mu01_s1", "g26_ret_rs025_mu01_s2")}
    bats = {r: b_ for r, b_ in bats.items() if b_}
    V["g26.n_batt"] = len(bats)
    if bats:
        for sc_ in SCALES:
            for n_, k_ in (("none", "u"), ("pushes+pulls", "d")):
                vals = [b_["goal"][f"drs{sc_}_{n_}"]["overall"]["success"] for b_ in bats.values()]
                V[f"g26.goal.{sc_}.{k_}.min"], V[f"g26.goal.{sc_}.{k_}.max"] = min(vals), max(vals)
                vals = [b_["vel"][f"drs{sc_}_{n_}"]["overall"]["fall"] for b_ in bats.values()]
                V[f"g26.vel.{sc_}.{k_}.min"], V[f"g26.vel.{sc_}.{k_}.max"] = min(vals), max(vals)
        for sp in SP:
            vals = [b_["flat"][sp]["achieved"] for b_ in bats.values()]
            V[f"g26.ach.{sp}.min"], V[f"g26.ach.{sp}.max"] = min(vals), max(vals)
        V["g26.ach_min"], V["g26.ach_max"] = min(b_["flat"][sp]["achieved"] for b_ in bats.values() for sp in SP), max(b_["flat"][sp]["achieved"] for b_ in bats.values() for sp in SP)
        V["g26.falls_max"] = max(b_["flat"][sp]["falls"] for b_ in bats.values() for sp in SP)
        V["g26.errmax"] = max(b_["flat"][sp]["err"] for b_ in bats.values() for sp in SP)
        L = {"1": ((0.73, 0.74), (0.45, 0.46), (0.34, 0.34), (0.51, 0.53)), "1.75": ((0.72, 0.72), (0.41, 0.42), (0.34, 0.36), (0.55, 0.56)), "2.5": ((0.68, 0.69), (0.33, 0.36), (0.38, 0.39), (0.60, 0.62))}
        if len(bats) == 3:
            for sc_, (gu, gd, vu, vd) in L.items():
                for k_, (lo, hi), q in (("u", gu, "goal"), ("d", gd, "goal"), ("u", vu, "vel"), ("d", vd, "vel")):
                    check("S99 close-out", "text", f"g26 {q} DR {sc_} {k_} min", V[f"g26.{q}.{sc_}.{k_}.min"], lo, 0.006)
                    check("S99 close-out", "text", f"g26 {q} DR {sc_} {k_} max", V[f"g26.{q}.{sc_}.{k_}.max"], hi, 0.006)
            for sp, (lo, hi) in {"0.3": (0.85, 0.93), "0.6": (0.95, 0.98), "0.9": (0.99, 1.00)}.items():
                check("S99 close-out", "text", f"g26 flat achieved {sp} min", V[f"g26.ach.{sp}.min"], lo, 0.006)
                check("S99 close-out", "text", f"g26 flat achieved {sp} max", V[f"g26.ach.{sp}.max"], hi, 0.006)
            check("S99 close-out", "text", "g26 flat falls max", V["g26.falls_max"], 0.00, 0.006)
            check("S99 close-out", "Table 34", "g26 standing min", V["mu.g26.stand.all.min"], 0.025, 0.0006)
            check("S99 close-out", "Table 34", "g26 standing max", V["mu.g26.stand.all.max"], 0.030, 0.0006)
            check("S99 close-out", "Table 34", "g26 driving min", V["mu.g26.drive.all.min"], 0.062, 0.0006)
            check("S99 close-out", "Table 34", "g26 driving max", V["mu.g26.drive.all.max"], 0.068, 0.0006)
    for run, st_, dr_ in (("g26_ret_rs025_mu01_s2", (0.025, 0.24, 0.04), (0.064, 0.49, 0.21)),):
        if run in runs:
            for d, l in zip(("all", "kb025", "bal4"), st_):
                check("S99 close-out", "Table 34", f"{run} standing {d}", V[f"mu.{run}.stand.{d}"], l, 0.006)
            for d, l in zip(("all", "kb025", "bal4"), dr_):
                check("S99 close-out", "Table 34", f"{run} driving {d}", V[f"mu.{run}.drive.{d}"], l, 0.006)
            check("S99 close-out", "Table 34", f"{run} non-finite", V.get(f"mu.{run}.nonfinite"), 0.000, 0.0006)
    pend = []
    for run in ("g26_ret_rs025_mu01_s0", "g26_ret_rs025_mu01_s1", "g26_ret_rs025_mu01_s2"):
        if run not in runs:
            S.setdefault("pending", {})[run] = run_progress(run)
            pend.append(f"<code>{run}</code> ({S['pending'][run]})")
        else:
            S.setdefault("pending", {}).pop(run, None)
    S["pending"].pop("g25_ret_rs025_stand_s0", None)
    txt = (f'<p><span class="pend">PENDING</span> {", ".join(pend)}: the remaining seed of the recipe retrained at 0.01.</p>' if pend else "")
    if len(bats) == 3:
        txt += ('<div class="callout"><b>Decision: the retrained seeds are the then-recommended checkpoints.</b> <code>g26_ret_rs025_mu01</code> seeds 0, 1 and 2 replace <code>g18_ret_rs025</code> seeds 0, 1 and 2. The recipe is unchanged (LQR plus residual at authority 0.25, arrestors, 11 designs, randomisation ramped to 1.75, 8 s episodes); only the contact model is corrected, and they are the only family with a reproducible standing number. Their standard battery (Tables 1 and 2, rows "corrected contact model"): goal success {{g26.goal.1.u.min:2}} to {{g26.goal.1.u.max:2}} undisturbed and {{g26.goal.1.d.min:2}} to {{g26.goal.1.d.max:2}} with pushes and pulls at randomisation scale 1, {{g26.goal.1.75.u.min:2}} / {{g26.goal.1.75.d.min:2}} to {{g26.goal.1.75.d.max:2}} at 1.75 and {{g26.goal.2.5.u.min:2}} to {{g26.goal.2.5.u.max:2}} / {{g26.goal.2.5.d.min:2}} to {{g26.goal.2.5.d.max:2}} at 2.5; tile velocity falls {{g26.vel.1.u.min:2}} undisturbed and {{g26.vel.1.d.min:2}} to {{g26.vel.1.d.max:2}} disturbed; flat achieved {{g26.ach.0.3.min:2}} to {{g26.ach.0.3.max:2}} at 0.3 m/s, {{g26.ach.0.6.min:2}} to {{g26.ach.0.6.max:2}} at 0.6 and {{g26.ach.0.9.min:2}} to {{g26.ach.0.9.max:2}} at 0.9 with falls {{g26.falls_max:2}}; standing 20 s {{mu.g26.stand.all.min:3}} to {{mu.g26.stand.all.max:3}} and driving {{mu.g26.drive.all.min:3}} to {{mu.g26.drive.all.max:3}} with no non-finite world; every seed within 0.03 of the others, at or above the g18 seeds on every battery cell, and equal to them on the obstacle course with a speed layer on the same contact model (Table 36: seed 1 of each family, {{spl.pair.g26.none.all:2}} against {{spl.pair.g18.none.all:2}} without and {{spl.pair.g26.gov.all:2}} against {{spl.pair.g18.gov.all:2}} with the governor). A gap between the families reported briefly in S101 was an evaluation bug, now fixed: the governor script did not apply a run\'s own free-rolling friction, so the g26 rows had been evaluated at 0.001 (S102). The g18 numbers remain valid comparisons for every 8 s result and for driving; their standing numbers are void. In these tables "then-recommended" means g18_ret_rs025 or, on the corrected contact model, g26_ret_rs025_mu01; each place says which. The current recommendation is g39_goalcentre_s0 (S121).</div>')
    inject("generalist.html", "pending_mu01", txt)


def fig_ballsize(C, size):
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    cols = [C["bad"], C["warn"], C["hybrid"], C["model"]]
    nseeds = 1
    for ax, kind, lab in zip(axes, ("step_up", "bump"), ("step height (cm)", "bump height (cm)")):
        for (d, r_cm), c in zip(TRAINED_R, cols):
            cur = S["ballsize_curves"][d][kind]
            nseeds = max(nseeds, len(S["ballsize_curves"][d]["seeds"]))
            pts = [(x, p_, ps) for x, n, p_, ps in cur if p_ is not None]
            ax.plot([x for x, p_, ps in pts], [p_ for x, p_, ps in pts], "o-", color=c, ms=3.5, lw=1.4, label=f"ball r = {r_cm:g} cm")
            for x, p_, ps in pts:      # seed spread: a bar from the lowest to the highest seed value
                vals = [v for v in ps if v is not None]
                if len(vals) > 1:
                    ax.plot([x, x], [min(vals), max(vals)], color=c, lw=1.0, alpha=0.7, solid_capstyle="butt")
            ax.axvline(0.29 * r_cm, color=c, ls=":", lw=1.1)
        ax.set_xlabel(lab)
        ax.set_ylim(0, 1.04)
        ax.set_title(f"Falls against {lab.split(' (')[0]}")
    axes[0].set_ylabel("fall probability (20 s at 0.3 m/s)")
    axes[0].legend(loc="lower right")
    axes[1].text(0.02, 0.97, "dotted: predicted limit 0.29 r" + ("; bars: spread of the two training seeds" if nseeds > 1 else ""), transform=axes[1].transAxes, va="top", fontsize=8, color=C["mute"])
    fig.tight_layout()
    return fig


def build_cpu_table():
    rows, texrows = [], []
    rows_csv = {r["run"]: r for r in csv.DictReader(open(RES / "tracking_summary.csv"))}
    spec = [("lqr", "LQR on the derived model"), ("trk_f_s0", "PPO, spin cap 1 rad/s, from scratch"), ("trk_f_s1", "same, seed 1"), ("trk_g_s0", "trk_f continued, smoother"),
            ("rob_b_s0", "PPO fine-tuned under DR, pushes 10 to 35 N"), ("robs_b_s0", "PPO from scratch under DR (narrow damping)"), ("robs_b_ki1", "robs_b + outer velocity integral"),
            ("res_c_s0", "LQR + residual (beta 0.1, zero-init)"), ("res_d_s0", "res_c continued, beta 0.2, DR + pushes")]
    S["cpu"] = {}
    for run, label in spec:
        r = rows_csv[run]
        f = lambda k, nd=2: "n/a" if r[k] in ("", "nan") else f"{float(r[k]):.{nd}f}"
        rf = r["rand_falls_30"]
        rf = "n/a" if rf in ("", "nan") else str(int(float(rf)))
        S["cpu"][run] = {k: r[k] for k in ("ratio_0.5", "ratio_1.0", "speed_rmse_le1", "rand_falls_30", "heading_err")}
        rows.append([f"{label}<br><small>{run}</small>", f("ratio_0.5"), f("ratio_1.0"), f("speed_rmse_le1"), f("heading_err"), f"{rf} / 30"])
        texrows.append([f"{label} ({run})", f("ratio_0.5"), f("ratio_1.0"), f("speed_rmse_le1"), f("heading_err"), f"{rf}/30"])
    head = ["Controller", "Achieved/commanded at 0.5 m/s", "at 1.0 m/s", "Velocity RMSE, commands up to 1 m/s [m/s]", "Heading-test error [m/s]", "Falls in 30 random-command episodes"]
    cap = setup_cap(engine="CPU (MuJoCo 3.13, float64, implicit-fast, contact-frame fix)", dr="off (nominal robot)", noise="off", worlds="1 robot per condition, 2 repeats per speed and direction (8 directions); 30 random-command episodes",
                    seeds="1 per row unless two are listed", protocol="scripts/study/eval_control.py and eval_falls.py, deterministic policies. The 1.0 m/s ratio of the LQR averages trials that mostly fall; the LQR falls in every 1 m/s trial of the study log.")
    inject("generalist.html", "tab_cpu", html_table(head, rows) + f'<p class="cap"><b>Table 0.</b> Baseline study on the CPU engine (selected rows). {cap}</p>')
    tex_table("cpu", "Baseline controllers on the CPU engine (selected rows of Table~\\ref{tab:tracking}). " + tex_escape(cap), "tab:gen_cpu", head, texrows, colspec="lccccc", star=True)
    ck = {"lqr": (1.04, 1.06, None), "trk_f_s0": (0.98, 0.91, None), "trk_f_s1": (1.09, 0.95, None), "trk_g_s0": (0.99, 0.87, None), "robs_b_s0": (0.50, 0.51, None), "res_c_s0": (1.03, 1.05, None)}
    for run, (a, b, _) in ck.items():
        check("S6/S7/S10/S11 (paper tab:tracking)", "Table 0", f"{run} ratio 0.5", float(rows_csv[run]["ratio_0.5"]), a, 0.006)
        check("S6/S7/S10/S11 (paper tab:tracking)", "Table 0", f"{run} ratio 1.0", float(rows_csv[run]["ratio_1.0"]), b, 0.006)
    check("S7", "text", "trk_h falls/30", float(rows_csv["trk_h_s0"]["rand_falls_30"]), 3, 0.5)
    check("S10", "text", "res_c falls/30", float(rows_csv["res_c_s0"]["rand_falls_30"]), 3, 0.5)
    check("S10", "text", "res_c heading err", float(rows_csv["res_c_s0"]["heading_err"]), 0.019, 0.001)
    check("S10", "text", "res_d random falls", float(rows_csv["res_d_s0"]["rand_falls_30"]), 15, 0.5)


def build_parity_table():
    """The engine speed benchmark. The old policy-transfer table is not built: those policies were trained with randomisation before the S69 correction."""
    bc, bw = json.load(open(ROOT / "outputs/study/mjx_port/bench_cpu.json")), json.load(open(ROOT / "outputs/study/mjx_port/bench_warp.json"))
    S["bench"] = dict(cpu=bc["control_steps_per_s"], warp4096=bw["4096"]["control_steps_per_s"], ratio=bw["4096"]["control_steps_per_s"] / bc["control_steps_per_s"])
    check("S20", "text", "Warp 4096 worlds / one CPU process", S["bench"]["ratio"], 260, 1.5, "log says about 260x")


def build_tiles28():
    rows, texrows = [], []
    S["tiles28"] = {}
    ds = {s: J("wpgen_none_s0", f"eval_tiles_v{s}.json") for s in ("0.3", "0.6", "1.0")}
    for k in KINDS:
        cells = [f"{ds[s]['kinds'][k]['fall']:.2f} ({ds[s]['kinds'][k]['crossed']:.2f})" for s in ("0.3", "0.6", "1.0")]
        S["tiles28"][k] = cells
        rows.append([KIND_LABEL[k]] + cells)
    head = ["Terrain", "0.3 m/s: fall (crossed)", "0.6 m/s", "1.0 m/s"]
    cap = setup_cap(dr="off", noise="off", worlds="64 per terrain kind", seeds="1 (wpgen_none_s0, blind, 100M steps)", protocol="nominal design, random heading, 8 s, severity mixed within a kind; crossed = left the 2.9 m tile square upright.")
    inject("generalist.html", "tab_tiles28", html_table(head, rows) + f'<p class="cap"><b>Table 5.</b> First blind baseline on the tiled map. {cap}</p>')
    tex_table("tiles28", "First blind policy on the tiled terrain: fall rate (crossed rate). " + tex_escape(cap), "tab:gen_tiles28", head, rows, colspec="lccc")
    for k, (a, b, c) in {"flat": ((0.02, 0.80), (0.06, 0.94), (0.05, 0.95)), "step_up": ((0.56, 0.05), (0.52, 0.48), (0.48, 0.52)), "bump": ((0.64, 0.02), (0.77, 0.22), (0.67, 0.33)),
                         "large_drop": ((0.55, 0.31), (0.59, 0.39), (0.55, 0.45))}.items():
        for s, lv in zip(("0.3", "0.6", "1.0"), (a, b, c)):
            check("S28", "Table 5", f"{k}@{s} fall", ds[s]["kinds"][k]["fall"], lv[0], 0.006)
            check("S28", "Table 5", f"{k}@{s} crossed", ds[s]["kinds"][k]["crossed"], lv[1], 0.006)


def build_perception():
    """Encoder pretraining metrics (depth to height prediction). The ridge and tile policy comparisons are superseded (S69) and not built."""
    enc = json.load(open(WP / "percep/enc_v1.json"))
    S["encoder"] = enc
    rows = []
    for e in enc:
        rows.append([e["model"].replace("_", " "), f"{e['R2_mean']:.2f}", f"{e['MAE_cm']:.2f}", f"{e['rise_ge2cm_F1']:.2f}", f"{e['drop_ge2cm_F1']:.2f}", " ".join(f"{x:.2f}" for x in e["R2_by_forward_row"])])
    head = ["Encoder", "R<sup>2</sup> (7x5 height grid)", "MAE [cm]", "rise &ge; 2 cm F1", "drop &ge; 2 cm F1", "R<sup>2</sup> by forward row (0.1 ... 1.3 m)"]
    cap = setup_cap(dr="on in the data collection (see the caveat)", noise="on (depth zone noise 0.5 % of range, 2 % dropout)", worlds="606k frames from 2048 worlds, 11 designs; validation = 10 % of worlds, split by world", seeds="1",
                    protocol="supervised regression of the height grid ahead and rise/drop hazards from an 18x14 simulated depth image; the driver policy is g2_goal_s0.",
                    extra="Caveat: the frames were collected by a policy running on the engine before the S69 fix (stale subtree mass under mass randomisation). The metrics describe depth-to-height prediction, which does not depend on that artefact, but the frame distribution does.")
    inject("generalist.html", "tab_encoder", html_table(head, rows) + f'<p class="cap"><b>Table 29.</b> What a pretrained depth encoder can predict. {cap}</p>')
    tex_table("encoder", "Supervised depth-to-height encoder. " + tex_escape(cap), "tab:gen_encoder", [re.sub(r"<[^>]+>", "", h) for h in head], rows, colspec="lccccc", star=True)
    for e, (r2, mae, f1r, f1d) in zip(enc, [(0.00, 3.87, 0.0, 0.0), (0.02, 3.97, 0.25, 0.20), (0.55, 2.19, 0.62, 0.67), (0.60, 1.93, 0.70, 0.73)]):
        check("S46", "Table 6", f"{e['model']} R2", e["R2_mean"], r2, 0.006)
        check("S46", "Table 6", f"{e['model']} MAE", e["MAE_cm"], mae, 0.006)
        check("S46", "Table 6", f"{e['model']} rise F1", e["rise_ge2cm_F1"], f1r, 0.006)
        check("S46", "Table 6", f"{e['model']} drop F1", e["drop_ge2cm_F1"], f1d, 0.006)


# ----------------------------------------------------------------------------------------------------------- robustness (S63+): data-driven placeholders
# ----------------------------------------------------------------------------------------------------------- figures
def pm(ax):
    ax.spines["left"].set_linewidth(0.8)


def fig_tracking(C, size):
    fig, axs = plt.subplots(1, 3, figsize=size)
    x = [float(s_) for s_ in SP]
    ser = [("pure learning (g12_pure)", ["g12_pure_s0"], C["learned"], "s", "-"), ("LQR alone", ["LQR"], C["model"], "D", "--"), ("LQR + residual 0.5 (2 seeds)", ["g12_res_s0", "g12_res_s1"], C["hybrid"], "^", "-"),
           ("soft base (2 seeds)", ["g12_soft175_s0", "g12_soft175_s1"], C["warn"], "o", "-")]
    for lbl, runs, col, mk, ls in ser:
        for k, run in enumerate(runs):
            vals = [tuple((np.nan if x is None else x) for x in (S["flat"][run][sp]["err"], S["flat"][run][sp]["ratio"], S["flat"][run][sp]["fall"])) for sp in SP]
            for i_, ax in enumerate(axs):
                ax.plot(x, [v[i_] for v in vals], color=col, marker=mk, ls=ls if k == 0 else ":", lw=1.5, ms=5, mfc=("none" if run == "LQR" else col), label=lbl if k == 0 else None)
    axs[0].set_ylabel("steady-state velocity error [m/s]"); axs[0].set_title("(a) error (lower is better)")
    axs[1].set_ylabel("achieved / commanded speed"); axs[1].set_title("(b) accuracy"); axs[1].axhline(1.0, color=C["mute"], ls="--", lw=0.8); axs[1].set_ylim(0.3, 1.15)
    axs[2].set_ylabel("fall rate"); axs[2].set_title("(c) falls on flat ground"); axs[2].set_ylim(-0.03, 1.05)
    for ax in axs:
        ax.set_xlabel("commanded speed [m/s]"); ax.set_xticks(x)
    axs[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def fig_terrain_goal(C, size):
    fig, axs = plt.subplots(2, 1, figsize=size, sharex=True)
    groups = [("pure learning", ["g12_pure_s0"], C["learned"]), ("LQR + residual 0.5 (2 seeds)", ["g12_res_s0", "g12_res_s1"], C["hybrid"]), ("soft base (2 seeds)", ["g12_soft175_s0", "g12_soft175_s1"], C["warn"])]
    w = 0.26
    for ai, (ax, key) in enumerate(zip(axs, ("ku", "kd"))):
        for gi, (lbl, runs, col) in enumerate(groups):
            ax.bar(np.arange(len(KINDS)) + (gi - 1) * w, [np.mean([S["goal"][r][key][k] for r in runs]) for k in KINDS], w, color=col, label=lbl)
        ax.set_ylabel("goal success" + (" (undisturbed)" if ai == 0 else " (pushes + pulls)")); ax.set_ylim(0, 1)
    axs[0].legend(fontsize=7, ncol=3, loc="upper right")
    axs[1].set_xticks(range(len(KINDS))); axs[1].set_xticklabels([KIND_LABEL[k] for k in KINDS], rotation=25, ha="right")
    fig.tight_layout()
    return fig


def smooth(y, w=7):
    k = np.ones(w) / w
    yy = np.convolve(np.pad(y, (w // 2, w // 2), mode="edge"), k, mode="valid")
    return yy[: len(y)]


def fig_curves(C, size):
    fig, axs = plt.subplots(1, 2, figsize=size)
    spec = [("g2_track_s0", "pure learning x1", C["mute"], "-"), ("g6_trk4_s0", "pure learning x4", C["learned"], "-"), ("g7_lqrres_s0", "LQR + residual 0.5, s0", C["hybrid"], "-"),
            ("g7_lqrres_s1", "LQR + residual 0.5, s1", C["hybrid"], ":"), ("g7_lqrres1_s0", "LQR + residual 1.0", C["bad"], "-")]
    for run, lbl, col, ls in spec:
        lg = read_log(run)
        if lg is None:
            continue
        x = lg["steps"] / 1e6
        axs[0].plot(x, smooth(lg["speed_err"]), color=col, ls=ls, lw=1.4, label=lbl)
        axs[1].plot(x, smooth(lg["fall_rate"]), color=col, ls=ls, lw=1.4)
    axs[0].set_ylabel("speed error during training [m/s]"); axs[1].set_ylabel("falls per environment step"); axs[1].set_yscale("log")
    axs[0].set_title("(a) training speed error"); axs[1].set_title("(b) training fall rate")
    for ax in axs:
        ax.set_xlabel("environment steps [M]")
    axs[0].legend(fontsize=6.5)
    fig.tight_layout()
    return fig


def fig_authority(C, size):
    fig, axs = plt.subplots(1, 3, figsize=size)
    x = [float(s_) for s_ in SP]
    ser = [("soft, authority 0.5 (s0)", "g12_soft175_s0", C["warn"], "o", "-"), ("soft, 0.5 (s1)", "g12_soft175_s1", C["warn"], "o", ":"), ("soft, authority 0.75", "g13_softrs075_s0", C["bad"], "x", "-")]
    for lbl, run, col, mk, ls in ser:
        axs[0].plot(x, [S["flat"][run][sp]["fall"] for sp in SP], color=col, marker=mk, ls=ls, lw=1.5, label=lbl)
        axs[1].plot(x, [S["flat"][run][sp]["err"] for sp in SP], color=col, marker=mk, ls=ls, lw=1.5)
    axs[0].set_ylabel("flat-world fall rate"); axs[0].set_title("(a) falls, flat"); axs[0].set_ylim(-0.03, 0.6)
    axs[1].set_ylabel("steady-state error [m/s]"); axs[1].set_title("(b) error, flat")
    for ax in axs[:2]:
        ax.set_xlabel("commanded speed [m/s]"); ax.set_xticks(x)
    names = [("0.5 s0", "g12_soft175_s0", C["warn"]), ("0.5 s1", "g12_soft175_s1", C["warn"]), ("0.75", "g13_softrs075_s0", C["bad"])]
    for i_, (lbl, run, col) in enumerate(names):
        axs[2].bar(i_ - 0.2, S["goal"][run]["u"]["success"], 0.38, color=col)
        axs[2].bar(i_ + 0.2, S["goal"][run]["d"]["success"], 0.38, color=col, alpha=0.45, hatch="//", edgecolor=C["fg"], linewidth=0)
    axs[2].set_xticks(range(len(names))); axs[2].set_xticklabels([n[0] for n in names]); axs[2].set_xlabel("residual authority (soft base)"); axs[2].set_ylabel("goal success"); axs[2].set_title("(c) goal (solid: undisturbed)")
    axs[2].set_ylim(0, 0.8)
    axs[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def fig_seeds(C, size):
    fig, ax = plt.subplots(figsize=size)
    items = []
    for fam, a, b, col in (("residual 0.5", "g12_res_s0", "g12_res_s1", C["hybrid"]), ("ramped 1.75x", "g12_dr175_s0", "g12_dr175_s1", C["learned"]), ("soft base", "g12_soft175_s0", "g12_soft175_s1", C["warn"])):
        items.append((f"{fam}: goal undisturbed success", [("seed 0", S["goal"][a]["u"]["success"], 1760), ("seed 1", S["goal"][b]["u"]["success"], 1760)], col))
        items.append((f"{fam}: goal disturbed success", [("seed 0", S["goal"][a]["d"]["success"], 1760), ("seed 1", S["goal"][b]["d"]["success"], 1760)], col))
    items.append(("residual 0.5: flat 0.9 m/s falls", [("seed 0", S["flat"]["g12_res_s0"]["0.9"]["fall"], 64), ("seed 1", S["flat"]["g12_res_s1"]["0.9"]["fall"], 64)], C["hybrid"]))
    for i_, (lbl, pts, col) in enumerate(items):
        y = len(items) - 1 - i_
        for j_, (sl, p_, nn) in enumerate(pts):
            lo, hi = wilson(p_, nn)
            ax.errorbar([p_], [y + (0.13 if j_ == 0 else -0.13)], xerr=[[p_ - lo], [hi - p_]], fmt="o" if j_ == 0 else "s", color=col, ms=5, lw=1.2, capsize=2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([it[0] for it in items][::-1]); ax.set_xlabel("value (95 % Wilson interval from the number of worlds; circle seed 0, square seed 1)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def fig_tiles(C, size):
    import tiles as TL
    fig, axs = plt.subplots(2, 5, figsize=size, sharey=False)
    sev = {"flat": 0, "step_up": 0.03, "step_down": 0.03, "small_drop": 0.05, "large_drop": 0.15, "ramp_up": 8.0, "ramp_down": 8.0, "stairs": 0.025, "bump": 0.03, "rough": 0.02}
    t = np.linspace(-1.45, 1.45, 600)
    rng = np.random.default_rng(1)
    for ax, k in zip(axs.ravel(), KINDS):
        if k == "rough":
            nz = TL.smooth_noise(len(t), rng)[0]
            h = np.where(np.abs(t) > TL.START_R, sev[k] * nz[: len(t)], 0.0)
        else:
            h = TL.g_profile(k, sev[k], t)
        ax.fill_between(t, h * 100, -22 if k == "large_drop" else min(h.min() * 100, 0) - 3, color=C["mute"], alpha=0.35, lw=0)
        ax.plot(t, h * 100, color=C["fg"], lw=1.3)
        ax.plot([0], [0], marker="o", ms=5, color=C["hybrid"])
        ax.set_title(f"{KIND_LABEL[k]}\n{TL.RANGE[k][0] * (100 if k not in ('ramp_up','ramp_down') else 1):g} to {TL.RANGE[k][1] * (100 if k not in ('ramp_up','ramp_down') else 1):g} {'deg' if 'ramp' in k else 'cm'}" if k != "flat" else "flat", fontsize=8)
        ax.set_xlim(-1.45, 1.45)
        lo = {"large_drop": -22, "small_drop": -9, "step_down": -8, "ramp_down": -16}.get(k, -3)
        hi = {"large_drop": 3, "ramp_up": 16, "stairs": 12}.get(k, 8)
        ax.set_ylim(lo, hi)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(labelsize=7)
    for i in (0, 5):
        axs.ravel()[i].set_ylabel("height [cm]")
    for ax in axs[1]:
        ax.set_xlabel("distance from tile centre [m]", fontsize=7)
    fig.tight_layout()
    return fig


def fig_arch(C, size):
    fig, ax = plt.subplots(figsize=size)
    ax.set_axis_off(); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.grid(False)
    def box(x, y, w, h, txt, col, fs=7.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="none", ec=col, lw=1.6, joinstyle="round"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=C["fg"])
    def arrow(x0, y0, x1, y1, col=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col or C["mute"], lw=1.3))
    box(0.1, 2.3, 1.7, 1.4, "command c\n(vx, vy, wz)\nor goal", C["mute"])
    box(2.4, 3.9, 2.2, 1.4, "reference governor\na_max = 0.5 m/s^2\n+ integral on velocity", C["model"])
    box(5.1, 3.9, 2.0, 1.4, "LQR (discrete, 10 ms)\nplanar K per design\n+ yaw loop", C["model"])
    box(2.4, 0.7, 2.2, 1.4, "observation o_t\n25 + history 8\n+ design vector", C["mute"])
    box(5.1, 0.7, 2.0, 1.4, "policy MLP\n3 x 512 tanh\nresidual output a_pi", C["learned"])
    box(8.0, 2.3, 1.6, 1.4, "a = clip(a_lqr\n+ rs * a_pi)\nrs = 0.5", C["hybrid"])
    box(10.1, 2.3, 1.8, 1.4, "wheel torques\ntau = kappa tau_d a\n(latency p = 0.2)", C["mute"])
    arrow(1.8, 3.2, 2.4, 4.4); arrow(1.8, 2.8, 2.4, 1.6)
    arrow(4.6, 4.6, 5.1, 4.6); arrow(7.1, 4.6, 8.4, 3.7, C["model"]); arrow(4.6, 1.4, 5.1, 1.4); arrow(7.1, 1.4, 8.4, 2.3, C["learned"])
    arrow(9.6, 3.0, 10.1, 3.0)
    ax.text(11.0, 1.15, "ballbot\n(MuJoCo-Warp)", ha="center", fontsize=7.5, color=C["fg"])
    arrow(11.0, 2.3, 11.0, 1.75)
    ax.text(6.1, 5.6, "model-based base: accurate, fragile", ha="center", fontsize=7.5, color=C["model"])
    ax.text(6.1, 0.2, "learned residual: bounded, trained with PPO", ha="center", fontsize=7.5, color=C["learned"])
    fig.tight_layout()
    return fig


def fig_retainer_geom(C, size):
    fig, ax = plt.subplots(figsize=size)
    ax.set_aspect("equal"); ax.grid(False)
    R, ra = 100 * R_BALL, 100 * R_ARR
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(R * np.cos(th), R + R * np.sin(th), color=C["mute"], alpha=0.25, lw=0)
    ax.plot(R * np.cos(th), R + R * np.sin(th), color=C["fg"], lw=1.2)
    ax.axhline(0, color=C["fg"], lw=1.5)
    z = np.radians(51.2)
    wx, wz = R * np.sin(z), R + R * np.cos(z)
    ax.plot([0, wx], [R, wz], color=C["model"], lw=1, ls="--")
    ax.add_patch(plt.Circle((wx + 1.5 * np.sin(z), wz + 1.5 * np.cos(z)), 1.5, fc="none", ec=C["model"], lw=1.8))
    ax.text(wx + 3.6, wz + 2.2, "drive wheel\n(contact 51 deg)", color=C["model"], fontsize=7.5)
    for phi, col, lab in ((90, C["mute"], "90"), (100, C["hybrid"], "100 (chosen)"), (120, C["warn"], "120"), (135, C["bad"], "135")):
        ph = np.radians(phi)
        cx, cz = -(R + ra) * np.sin(ph), R + (R + ra) * np.cos(ph)
        ax.add_patch(plt.Circle((cx, cz), ra, fc=col if phi == 100 else "none", ec=col, lw=1.6, alpha=0.9))
        ax.plot([0, cx], [R, cz], color=col, lw=0.8, ls=":")
        ax.text(cx - 2.6, cz, f"{lab}", color=col, fontsize=7.5, ha="right", va="center")
        if phi in (100, 135):
            ax.annotate("", xy=(cx, 0), xytext=(cx, cz - ra), arrowprops=dict(arrowstyle="<->", color=col, lw=0.9))
            ax.text(cx - 0.5 if phi == 100 else cx + 0.5, 1.6 if phi == 100 else 0.4, f"{100 * clearance(phi):.1f} cm", color=col, fontsize=7, ha="right" if phi == 100 else "left")
    ax.plot([0], [R], marker="o", ms=3, color=C["fg"])
    ax.text(0.5, R + 0.6, "ball centre", fontsize=7, color=C["fg"])
    ax.set_xlim(-24, 20); ax.set_ylim(-1.5, 23)
    ax.set_xlabel("cm"); ax.set_ylabel("height above floor [cm]")
    ax.set_title("arrestor mounts on the azimuth opposite one wheel (section through the ball centre)")
    fig.tight_layout()
    return fig


def fig_retainer(C, size):
    fig, axs = plt.subplots(1, 2, figsize=size)
    sw = [x for x in S["sweep"] if x["phi"]]
    phis = [x["phi"] for x in sw]
    axs[0].plot(phis, [x["u"] for x in sw], marker="o", color=C["hybrid"], lw=1.6, label="undisturbed")
    axs[0].plot(phis, [x["d"] for x in sw], marker="s", color=C["warn"], lw=1.6, label="pushes + pulls")
    axs[0].axhline(V["sw.none.u"], color=C["hybrid"], ls="--", lw=0.9); axs[0].axhline(V["sw.none.d"], color=C["warn"], ls="--", lw=0.9)
    axs[0].text(136, V["sw.none.u"] + 0.01, "no arrestors", fontsize=7, color=C["mute"], ha="right")
    axs[0].set_xlabel("arrestor mount angle from the top of the ball [deg]"); axs[0].set_ylabel("goal success"); axs[0].set_title("(a) zero-shot mount sweep"); axs[0].set_ylim(0, 0.9); axs[0].legend(fontsize=7)
    groups = [("no arrestors\n(2 seeds)", ["g12_dr175_s0", "g12_dr175_s1"], None), ("zero-shot\n(1 seed)", None, "zs"), ("trained with\n(2 seeds)", ["g16_ret100_s0", "g16_ret100_s1"], None), ("pure learning\n(2 seeds)", ["g12_pure_s0", "g12_pure_s1"], None)]
    for i, (lab, runs, z) in enumerate(groups):
        if z:
            u, d = V["sw.100.u"], V["sw.100.d"]
        else:
            u = np.mean([S["goal"][r]["u"]["success"] for r in runs]); d = np.mean([S["goal"][r]["d"]["success"] for r in runs])
        axs[1].bar(i - 0.2, u, 0.38, color=C["hybrid"], label="undisturbed" if i == 0 else None)
        axs[1].bar(i + 0.2, d, 0.38, color=C["warn"], label="pushes + pulls" if i == 0 else None)
    axs[1].set_xticks(range(len(groups))); axs[1].set_xticklabels([g[0] for g in groups], fontsize=7.5); axs[1].set_ylabel("goal success (DR scale 1)"); axs[1].set_title("(b) arrestors at 100 deg"); axs[1].set_ylim(0, 0.9)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_tradeoff(C, size):
    fig, ax = plt.subplots(figsize=size)
    col = {"ref": C["mute"], "zs": C["model"], "res": C["hybrid"], "pure": C["learned"]}
    mk = {"ref": "s", "zs": "D", "res": "o", "pure": "^"}
    seen = set()
    lab_map = {"ref": "no arrestors (reference)", "zs": "arrestors added at evaluation only", "res": "LQR + residual, trained with arrestors", "pure": "pure learning, trained with arrestors"}
    for pnt in S["tradeoff"]:
        k = pnt["kind"]
        ax.plot(pnt["ach"], pnt["d"], marker=mk[k], ms=7 if "0.25" in pnt["label"] else 6, color=col[k], ls="none", mfc=col[k] if "0.25" not in pnt["label"] else "none", mew=2 if "0.25" in pnt["label"] else 1,
                label=lab_map[k] if k not in seen else None)
        seen.add(k)
        short = pnt["label"].replace(" (then-recommended)", "*").replace("authority ", "rs ").replace(", tracking weight x3", " trk x3").replace("soft LQR base, rs 0.5", "soft").replace("pure learning (no LQR)", "pure").replace("no arrestors (reference)", "").replace("rs 0.5, arrestors added at evaluation only", "zero-shot")
        if short and pnt["run"] not in ("g12_dr175_s1",):
            ax.annotate(short, (pnt["ach"], pnt["d"]), textcoords="offset points", xytext=(5, 3), fontsize=6.5, color=C["fg"])
    ax.set_xlabel("flat-world achieved / commanded speed (mean of 0.3, 0.6, 0.9 m/s)"); ax.set_ylabel("goal success with pushes and pulls (DR scale 1)")
    ax.set_xlim(0.6, 1.08); ax.set_ylim(0.2, 0.56)
    ax.axvline(1.0, color=C["mute"], ls="--", lw=0.8)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("accuracy against robustness, one point per controller (* = then-recommended, rs 0.25)")
    fig.tight_layout()
    return fig


def fig_robust(C, size):
    fig, axs = plt.subplots(1, 2, figsize=size)
    x = [float(sc) for sc in SCALES]
    fams = [("pure learning", ["g12_pure_s0"], C["learned"], "s"), ("LQR + residual 0.5", ["g12_res_s0", "g12_res_s1"], C["hybrid"], "^"), ("ramped to 1.75x", ["g12_dr175_s0", "g12_dr175_s1"], C["model"], "D"),
            ("soft base", ["g12_soft175_s0", "g12_soft175_s1"], C["warn"], "o")]
    for lbl, runs, col, mk in fams:
        for ax, key in zip(axs, ("u", "d")):
            ax.plot(x, [np.mean([S["goalscale"][r][sc][key]["success"] for r in runs]) for sc in SCALES], color=col, marker=mk, lw=1.6, label=lbl)
    axs[0].set_title("(a) goal success, undisturbed"); axs[1].set_title("(b) goal success, pushes + pulls")
    for ax in axs:
        ax.set_xlabel("randomisation scale at evaluation"); ax.set_ylabel("goal success"); ax.set_ylim(0, 0.8); ax.set_xticks(x)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------------------------- videos
CLIPS = [
    ("c1_goal_ramp", "LQR + residual, goal on an 11 degree ramp", "g12_res_s0 reaches the goal on the ramp; no fall in 8 s. Ramp-up goals are the residual's best kind (success 0.74 and 0.69 over 176 worlds each)."),
    ("c1_goal_step", "LQR + residual, goal across a 2.3 cm step (falls)", "g12_res_s0 falls at the step after 3.0 s. Step-up goals succeed in about one world in three (0.33 and 0.28 over 176 worlds each), so most steps end like this one."),
    ("c1_goal_flat", "LQR + residual, goal on a flat tile (falls)", "g12_res_s0 falls after 4.8 s on a flat tile. Flat-tile goal success is about 0.6 to 0.7, so a fall here is not rare; it is shown because failures are part of the record."),
    ("c2_flat_nocage", "Same policy, flat tile, arrestors off (falls)", "g12_dr175_s0 without arrestors: the body derails off the ball and the robot falls after 4.8 s."),
    ("c2_flat_cage100", "Same policy, flat tile, arrestors on", "The same policy and world with the 100 deg arrestors added at evaluation (zero-shot): no fall in 8 s."),
    ("c2_step_up_nocage", "Same policy, 2.3 cm step, arrestors off (falls)", "g12_dr175_s0 without arrestors falls at the step after 2.9 s."),
    ("c2_step_up_cage100", "Same policy, 2.3 cm step, arrestors on (falls later)", "The same policy and world with arrestors falls too, later (5.5 s): the retainer prevents derailing, not every fall."),
    ("c3_step_up_pure_nocage", "Pure learning on a step-up, without arrestors (falls)", "Pure learning trained without arrestors (g12_pure_s0) falls at the 2.3 cm step after 2.8 s."),
    ("c3_step_up_pure_cage", "Pure learning on a step-up, with arrestors", "Pure learning trained with the arrestors (g16_pure_ret100_s1, a different policy from the clip above) crosses toward the goal and does not fall in 8 s."),
    ("c4_step_up_rs025_cage", "Then-recommended controller: step-up toward a goal", "LQR + residual at authority 0.25 trained with the arrestors (g18_ret_rs025_s1) climbs the 2.3 cm step toward the goal and holds for 8 s."),
    ("c4_ramp_vel06_rs025_cage", "Then-recommended controller: 0.6 m/s up an 11 degree ramp (known weakness, falls)", "The same controller following a 0.6 m/s command up an 11 degree ramp lags the command (0.40 m/s at 2.8 s) and falls at 5.6 s. A known weakness, shown as a single clip, not a statistic."),
    ("c6_nav_blind", "Navigation, blind: falls into the block", "The then-recommended balance controller driven straight at the goal with no terrain information rolls onto the 8 cm block and falls."),
    ("c6_nav_oracle", "Navigation with the perfect height map: goes around the block", "The same frozen controller with the wavefront planner on the privileged height map steers around the block end and reaches the goal."),
    ("c7_step4cm_s075", "A 5 cm step, 6.75 cm ball: falls", "The scale policy on the 0.75-scale robot (ball radius 6.75 cm, predicted limit 2.0 cm) drives into a 5 cm up-step at 0.26 m/s, stalls against the edge and falls at 6.6 s."),
    ("c7_step4cm_s2", "The same 5 cm step, 18 cm ball: crosses", "The same policy on the 2-scale robot (ball radius 18 cm, predicted limit 5.2 cm) climbs the same 5 cm step and drives on; no fall in the 8 s clip."),
    ("c6_nav_tof18", "Navigation with the forward ToF sensor (18 x 14): hesitates, then arrives", "The same controller and planner with the head-mounted ToF sensor hesitates while its obstacle memory fills in, then goes around the block and reaches the goal at 16.5 s. A single clip."),
]


CLIP_META = {"c6_nav_blind": dict(run="g18_ret_rs025_s1", kind="blocks", severity=0.08, task="navigation", fell=True, seconds=float("nan")),
             "c6_nav_oracle": dict(run="g18_ret_rs025_s1", kind="blocks", severity=0.08, task="navigation", fell=False, seconds=20.0),
             "c6_nav_tof18": dict(run="g18_ret_rs025_s1", kind="blocks", severity=0.08, task="navigation", fell=False, seconds=20.0)}


def build_videos():
    vsrc = WP / "videos"
    out = []
    for name, title, desc in CLIPS:
        js = json.load(open(vsrc / f"{name}.json")) if (vsrc / f"{name}.json").exists() else CLIP_META[name]
        for ext in ("mp4", "jpg"):
            dst = SITE / "videos" / f"{name}.{ext}"
            if not dst.exists() or dst.stat().st_size != (vsrc / f"{name}.{ext}").stat().st_size:
                shutil.copy2(vsrc / f"{name}.{ext}", dst)
        out.append(dict(name=name, title=title, description=desc, run=js["run"], kind=js["kind"], severity=js["severity"], task=js["task"], fell=js["fell"], seconds=js["seconds"], design=js.get("design", "nom")))
    S["videos"] = out
    cards = []
    for v in out:
        sev = v["severity"]
        sevt = "" if v["kind"] in ("flat", "blocks") else (f", severity {sev:.1f} deg" if v["kind"].startswith("ramp") else f", severity {sev * 100:.1f} cm")
        outcome = ("fell" if v["seconds"] != v["seconds"] else f"fell after {v['seconds']:.1f} s") if v["fell"] else f"no fall in {v['seconds']:.0f} s"
        cards.append(f'<li class="vcard"><video controls preload="none" playsinline data-poster="videos/{v["name"]}.jpg" src="videos/{v["name"]}.mp4" aria-label="{v["title"]}" style="width:100%;aspect-ratio:16/9;background:#000"></video>'
                     f'<div class="vbody"><h3>{v["title"]}</h3><p>{v["description"]}</p>'
                     f'<p class="vmeta">Policy {v["run"]}; {v["task"]} task; terrain {KIND_LABEL[v["kind"]]}{sevt}; {"nominal design" if v["design"] == "nom" else "design " + v["design"] + " (robot scaled " + v["design"][1:] + " times, Froude-scaled command)"}; Warp GPU physics, CPU MuJoCo renderer. Engine: Warp GPU, corrected. DR and noise: not recorded in the clip metadata (the renderer defaults to off). Worlds: 1 (a single clip). Seed: policy seed {v['run'][-1]}. Outcome: <b class="{"bad" if v["fell"] else "good"}">{outcome}</b>. One world is an anecdote, not a statistic; the tables carry the rates.</p></div></li>')
    inject("generalist.html", "videos", '<ul class="vgrid">' + "".join(cards) + "</ul>")
    return out


# Clips shown on videos.html and the home page that are not part of CLIPS above (chapter 6 and 7 of the videos page).
# c11_goal_bump_J02 was withdrawn (study log) and must not be published.
FEATURED = ["c11_goal_hold_nom", "c11_step15_nom", "c11_step3cm_nom", "c11_ramp_zhigh", "c11_stepdown_kb4", "c11_stairs_nom",
            "c12_nav_with_layer", "c12_nav_without_layer", "c13_push_pull_hold", "c14_g40_push_pull_hold"]


def copy_featured_clips():
    vsrc = WP / "videos"
    for name in FEATURED:
        assert "goal_bump" not in name
        for ext in ("mp4", "jpg"):
            src, dst = vsrc / f"{name}.{ext}", SITE / "videos" / f"{name}.{ext}"
            if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
                shutil.copy2(src, dst)
        js = vsrc / f"{name}.json"
        if js.exists():
            S.setdefault("featured_clips", {})[name] = json.load(open(js))


# ----------------------------------------------------------------------------------------------------------- cross-check report
def write_report():
    ok = [c for c in CHECKS if abs(c[3] - c[4]) <= c[5] + 1e-9]
    bad = [c for c in CHECKS if abs(c[3] - c[4]) > c[5] + 1e-9]
    p = ROOT / "docs/research/site_crosscheck.md"
    if SITE_ONLY:
        src_p, p = p, PAPER.parent / "site_crosscheck.md"
        if src_p.exists() and not p.exists():
            shutil.copy2(src_p, p)
    head = (p.read_text(encoding="utf8").split("<!-- AUTO -->")[0] if p.exists() and "<!-- AUTO -->" in p.read_text(encoding="utf8") else "# Site and paper cross-check\n\n")
    L = [head.rstrip() + "\n\n<!-- AUTO -->", "## Automatic cross-check (written by site/build_data.py)\n",
         "Supersession note (S92): the flat-benchmark rows printed in log entries S59 to S91 were measured on border tiles; every battery.json flat row was re-run on interior tiles, and the files are taken as correct. The re-run is a fresh sample, so its steady-state error and achieved speed also differ slightly from the logged values (by 0.001 to 0.01 m/s and up to 0.02 in ratio); the automatic checks therefore no longer compare flat rows with those entries. The LQR-alone row is the interior-tile re-measurement of the S92 addendum (log values, dagger); the zero-shot arrestor flat row was not re-run and is shown with a caveat.\n",
         f"{len(CHECKS)} quantities compared: {len(ok)} match the study log within rounding, {len(bad)} differ.\n"]
    if bad:
        L.append("### Differences\n\n| entry | where | quantity | data file | log | note |\n|---|---|---|---|---|---|")
        for c in bad:
            note = c[6]
            if not note and "g7_lqrres_s0" in c[0] and c[1] == "Table 2":
                note = "eval2_goal_goal_dr_surf[_pull].json of g7_lqrres_s0 was re-written 2026-11-07 07:19 to 07:24 by a re-evaluation (env settings are not recorded in the JSON); the log holds the earlier run"
            if not note and c[0] == "S48:g2_goal_s0" and c[2] == "ramp_up":
                note = "log S51 and S54 print 0.14 for g2_goal_s0; eval_goal.txt and the JSON give 0.18 (0.14 is g3_enc_s0)"
            if not note and c[0] == "S79" and "saturation" in c[2]:
                note = "the log may quote a different statistic (mean over falls here); both values are under 1 percent, so the reading is unchanged"
            if not note and c[0] == "S78" and "g15_edge_depth" in c[1]:
                note = "g15_edge_depth_s0/battery.json was re-written at 20:06 on 2026-11-07, after the S78 table (battery run at 19:52); the site follows the file; the conclusion is unchanged"
            L.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]:.4f} | {c[4]} | {note} |")
        L.append("")
    L.append("### Statements checked against the source code\n\n| statement on the pages | pattern searched | file | found |\n|---|---|---|---|")
    for st, pat, fl, ok_ in SRC_CHECKS:
        L.append(f"| {st} | `{pat}` | {fl} | {'yes' if ok_ else 'NO'} |")
    L.append("")
    L.append("### All matches\n\n| log entry | where used | quantity | value from data file | value in log | tolerance |\n|---|---|---|---|---|---|")
    for c in CHECKS:
        L.append(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]:.4f} | {c[4]} | {c[5]} |")
    p.write_text("\n".join(L) + "\n", encoding="utf8")
    return len(CHECKS), len(bad)


# ----------------------------------------------------------------------------------------------------------- model numbers (math verified numerically)
V = {}      # values used in prose: {{key:decimals*multiplier}} on the pages, \gv{key} in the paper


SRC_CHECKS = []      # (statement printed on the pages, pattern searched in the source file, file, found)


def src_check(statement, pattern, relfile):
    txt = (ROOT / relfile).read_text(encoding="utf8")
    SRC_CHECKS.append((statement, pattern, relfile, pattern in txt))


def build_world_tables():
    import ast
    import tiles as TL
    import yaml
    # terrain tiles
    _, info = TL.build(0)
    cnt = {k: sum(1 for t in info if t["kind"] == k) for k in KINDS}
    desc = {"flat": "no feature", "step_up": "single raised step 0.6 m from the centre on each axis", "step_down": "robot stands on a plateau, floor drops by the severity",
            "small_drop": "drop of 3 to 8 cm", "large_drop": "drop of 10 to 20 cm", "ramp_up": "slope over 1.0 m starting 0.5 m from the centre", "ramp_down": "descending slope over 1.0 m",
            "stairs": "2 to 3 treads of 0.4 m, riser = severity", "bump": "slab 0.3 m wide, 0.6 m from the centre", "rough": "smooth two-octave noise outside a flat 0.5 m start zone"}
    rows = []
    for k in KINDS:
        lo, hi = TL.RANGE[k]
        rng = "none" if k == "flat" else (f"{lo:g} to {hi:g} deg" if "ramp" in k else f"{lo * 100:g} to {hi * 100:g} cm")
        rows.append([KIND_LABEL[k], rng, desc[k], str(cnt[k])])
    S["tiles"] = dict(counts=cnt, map_m=2 * TL.HALF, cell_cm=2 * TL.HALF / (TL.NCELL - 1) * 100, tile_m=TL.TILE, exit_r=TL.EXIT_R)
    head = ["Terrain kind", "Severity range", "Profile", "Tiles of 64"]
    inject("method.html", "tab_tiles", html_table(head, rows, cls="") + f'<p class="cap"><b>Table A.</b> Terrain kinds. Map {2 * TL.HALF:g} m, {S["tiles"]["cell_cm"]:.1f} cm cells, tiles of {TL.TILE:g} m, truncation outside {2 * TL.EXIT_R:.1f} m. Source: experiments/warp_ppo/tiles.py (seed 0). Geometry only: no engine, randomisation, noise or seeds apply.</p>')
    tex_table("tiles", "Terrain tile library (geometry only; map 24\\,m, 3\\,m tiles, truncation outside a 2.9\\,m square).", "tab:gen_tiles", head, rows, colspec="llp{4.2cm}c")
    # surfaces
    src = (ROOT / "experiments/warp_ppo/train_dev.py").read_text(encoding="utf8")
    m = re.search(r"SURFACES = (\{.*?\})\n", src, re.S)
    surf = ast.literal_eval(re.sub(r"\s+", " ", m.group(1)))
    rows = [[k.replace("_", " "), f"{v[0]:.1f}", f"{v[1]:.3f}"] for k, v in surf.items()]
    S["surfaces"] = surf
    inject("method.html", "tab_surfaces", html_table(["Floor class", "Sliding friction mu", "Contact time constant [s]"], rows) + '<p class="cap"><b>Table B.</b> Floor material classes used when floor randomisation is on. Source: experiments/warp_ppo/train_dev.py. Values are plausible, not measured.</p>')
    tex_table("surfaces", "Floor material classes (plausible, not measured).", "tab:gen_surfaces", ["Floor class", "Friction $\\mu$", "Contact time constant [s]"], rows, colspec="lcc")
    # designs
    names = ["nom", "zhigh", "zlow", "kb4", "bal4", "J02", "tau6", "bal025", "kb025", "zeta41", "zeta66", "s0.5", "J0005"]
    what = {"nom": "reference robot", "zhigh": "ballast raised: COM height about 1.9 times nominal", "zlow": "ballast lowered: COM height about 0.67 times nominal", "kb4": "ball mass x4",
            "bal4": "body ballast x4 (body mass 3.5 times nominal, pushes scaled)", "J02": "wheel rotor inertia x4", "tau6": "torque limit 6 N m (one fifth)", "bal025": "ballast x0.25 (body mass 0.38 times nominal)",
            "kb025": "ball mass x0.25", "zeta41": "wheel contact zenith 41 deg (nominal 51)", "zeta66": "wheel contact zenith 66 deg", "s0.5": "half scale (all lengths x0.5, masses x0.125)", "J0005": "rotor inertia x0.1"}
    rows = []
    des = {}
    for n in names:
        P = json.load(open(ROOT / f"ballbot_gym/models/variants/dz_{n}.json"))["params"]
        c = yaml.safe_load(open(ROOT / f"configs/env/dz_{n}.yaml"))["env"]
        used = "no (dropped)" if n in ("s0.5", "J0005") else "yes"
        des[n] = dict(m_k=P["m_k"], m_a=P["m_a"], l=P["l"], J_w=P["J_w"], tau=c["max_wheel_velocity"], zenith=P["zenith_deg"], used=used)
        rows.append([f"<b>{n}</b>", what[n], f"{P['m_a']:.2f}", f"{P['m_k']:.3f}", f"{P['l']:.3f}", f"{P['J_w']:.4g}", f"{c['max_wheel_velocity']:g}", f"{P['zenith_deg']:.0f}", used])
    S["designs"] = des
    head = ["Design", "What it changes", "Body mass [kg]", "Ball mass [kg]", "COM height l [m]", "Rotor inertia [kg m^2]", "Torque limit [N m]", "Contact zenith [deg]", "In training set"]
    inject("method.html", "tab_designs", html_table(head, rows) + '<p class="cap"><b>Table C.</b> The 13 compiled robot designs. Source: ballbot_gym/models/variants/dz_*.json and configs/env/dz_*.yaml. Geometry and mass properties only; no engine or randomisation applies. The eleven designs marked yes form the training set of every main run.</p>')
    tex_table("designs", "The 13 compiled designs (eleven used in training).", "tab:gen_designs", head, [[re.sub(r"<[^>]+>", "", str(c)) for c in r] for r in rows], colspec="llcccccc" + "c", star=True)
    # randomisation table (from the yaml of the nominal design and checks on the source)
    d = yaml.safe_load(open(ROOT / "configs/env/dz_nom.yaml"))["env"]["dr"]
    for stmt, pat in [("pull force 3 to 12 N", "lo=3.0, hi=12.0"), ("pull duration 50 to 199 steps", "torch.randint(50, 200"), ("pull interval 400 to 1000 steps", "lo=400, hi=1000"),
                      ("push 5 control steps", "torch.full_like(self.push_left, 5)"), ("push interval 200 to 400 steps", "lo=200, hi=400"), ("surface friction jitter 0.15", "lo=-0.15, hi=0.15"),
                      ("max latency prob 0.6", "min(self.drc.get('latency_p', 0.2) * self.dr_l, 0.6)"), ("start offset 0.3 m", "lo=-0.3, hi=0.3"),
                      ("master scale touches six ranges only", "self.dr_m = self.dr_f = self.dr_t = self.dr_d = self.dr_l = self.dr_n = float(x)")]:
        src_check(stmt, pat, "experiments/warp_ppo/train_dev.py")
    rows = [
        ["Robot design", "one of 11 designs, uniform", "per episode", "Table C"],
        ["Base mass and ball mass", f"each times (1 plus uniform plus or minus {d['mass']:.2f})", "per episode", "independent draws"],
        ["Wheel-ball transverse friction", f"times (1 plus uniform plus or minus {d['friction']:.2f})", "per episode", "the cross-roller direction of the anisotropic pair"],
        ["Wheel damping", f"uniform {d['damping'][0]} to {d['damping'][1]} N m s times the design's nominal damping over 0.1", "per episode, each wheel", "master scale widens the spread around 0.1"],
        ["Torque scale", f"1 plus uniform plus or minus {d['torque']:.2f}", "per episode", "applied to the commanded torque"],
        ["Action latency", f"with probability {d['latency_p']:.1f} the previous action is applied instead", "per control step", "probability capped at 0.6 when scaled"],
        ["Observation noise", f"Gaussian, sigma {d['obs_noise']:.2f} (orientation 0.25 sigma, angular velocity 2 sigma, linear velocity 1 sigma)", "per control step", "slots 0 to 8 only"],
        ["Pushes", f"horizontal force {d['push_force'][0]:g} to {d['push_force'][1]:g} N (times the design push factor), random direction, 5 control steps ({d['push_dur'] * 1000:g} ms)", f"every {d['push_every'][0]:g} to {d['push_every'][1]:g} s", "factor = push range of the design over 20 N"],
        ["Pulls", "horizontal force 3 to 12 N, random direction, 0.5 to 2 s", "every 4 to 10 s", "only in runs with pulls on; not scaled per design"],
        ["Floor material", "one of 7 classes; friction times (1 plus uniform plus or minus 0.15)", "per episode", "Table B"],
        ["Terrain tile and start", "uniform over 64 tiles; start offset uniform plus or minus 0.3 m; start yaw uniform over a full turn", "per episode", "Table A"],
        ["Master scale (wide runs)", "multiplies mass, friction, torque, damping spread, latency probability and noise ranges; 1 to 1.75 over the first half of training", "per iteration", "does not scale pushes or pulls"],
    ]
    head = ["Quantity", "Range", "When", "Notes"]
    cap = '<p class="cap"><b>Table D.</b> Domain randomisation. Source: configs/env/dz_nom.yaml (the ranges) and experiments/warp_ppo/train_dev.py (how they are applied; each statement was matched against the source by site/build_data.py). Engine: Warp GPU. Randomisation: this table is the randomisation.</p>'
    inject("method.html", "tab_dr", html_table(head, rows, cls="") + cap)
    tex_table("dr", "Domain randomisation ranges (configs/env/dz\\_nom.yaml and train\\_dev.py).", "tab:gen_dr", head, rows, colspec="p{2.6cm}p{6cm}p{2.4cm}p{3.6cm}", star=True)
    for q, comp, lg in [("mass range", d["mass"], 0.15), ("friction range", d["friction"], 0.2), ("damping lo", d["damping"][0], 0.07), ("damping hi", d["damping"][1], 0.15), ("torque range", d["torque"], 0.15),
                        ("latency p", d["latency_p"], 0.2), ("obs noise", d["obs_noise"], 0.01), ("push min", d["push_force"][0], 5.0), ("push max", d["push_force"][1], 20.0),
                        ("push every lo", d["push_every"][0], 2.0), ("push every hi", d["push_every"][1], 4.0), ("push dur", d["push_dur"], 0.05)]:
        check("config", "Table D", q, comp, lg, 1e-9)
    cfg = yaml.safe_load(open(ROOT / "configs/env/dz_nom.yaml"))["env"]
    for q, comp, lg in [("cmd_vmax", cfg["cmd_vmax"], 1.2), ("cmd_wmax", cfg["cmd_wmax"], 1.0), ("action_repeat", cfg["action_repeat"], 5), ("max episode steps", cfg["max_ep_steps"], 800),
                        ("tilt limit", cfg["max_allowed_tilt"], 70), ("ball offset limit", cfg["max_ball_offset_deg"], 30), ("cmd resample steps", cfg["cmd_resample_steps"], 300),
                        ("cmd zero prob", cfg["cmd_zero_prob"], 0.1), ("torque limit", cfg["max_wheel_velocity"], 30.0)]:
        check("config", "Method text", q, comp, lg, 1e-9)


def build_model():
    """Re-derive the structural numbers of the planar model from the exported parameters and compare with the numbers printed on the pages and in the paper."""
    import scipy.linalg as la
    from scipy.signal import cont2discrete
    M = json.load(open(ROOT / "ballbot_gym/models/ballbot_planar_model.json"))
    P = M["params"]
    A, B = np.array(M["A"]), np.array(M["B"]).ravel()
    g = 9.81
    r, mk, Ik, ma, l, Ia, Jw = P["r"], P["m_k"], P["I_k"], P["m_a"], P["l"], P["I_a"], P["J_w"]
    n = np.array(P["W"])[:, 1]
    Jeff = Jw * float(n @ n)
    M11, M12, M22 = Ik + Jeff + (ma + mk) * r * r, ma * l * r - Jeff, Ia + Jeff + ma * l * l
    Dl = M11 * M22 - M12 ** 2
    lam0 = math.sqrt(ma * g * l * M11 / Dl)
    z = math.sqrt(ma * g * l / (Ia + ma * l * l + ma * l * r))
    kappa = ((ma + mk) * r * r + Ik + ma * l * r) / (ma * g * l * r)
    ev = np.linalg.eigvals(A)
    Q = np.diag([1e-6, 50.0, 1.0, 1.0])
    X = la.solve_continuous_are(A, B[:, None], Q, np.array([[1.0]]))
    Kc = (B[None] @ X).ravel()
    cl = np.linalg.eigvals(A - B[:, None] @ Kc[None])
    Ad, Bd, *_ = cont2discrete((A, B[:, None], np.eye(4), np.zeros((4, 1))), 0.01)
    Xd = la.solve_discrete_are(Ad, Bd, Q * 0.01, np.array([[0.01]]))
    Kd = np.linalg.solve(np.array([[0.01]]) + Bd.T @ Xd @ Bd, Bd.T @ Xd @ Ad).ravel()
    pd = np.abs(np.linalg.eigvals(Ad - Bd @ Kd[None]))
    vals = dict(Jeff=Jeff, n2=float(n @ n), lam0=lam0, p=float(ev.real.max()), z=z, kappa=kappa, kappa_deg=math.degrees(math.asin(kappa * 1.0)) if kappa < 1 else float("nan"),
                Kc1=Kc[1], Kc2=Kc[2], Kc3=Kc[3], Kd1=Kd[1], Kd2=Kd[2], Kd3=Kd[3], clmin=float(cl.real.min()), dpoles_max=float(pd.max()), dpole_fast=float(pd.min()),
                ratio_J=Jeff / Ik, M11=M11, M12=M12, M22=M22, Delta=Dl, a_lean30=math.sin(math.radians(30)) / kappa)
    S["model"] = vals
    for k, v in vals.items():
        V[f"model.{k}"] = v
    S["model_matrices"] = dict(A=A.tolist(), B=B.tolist(), poles=[float(x.real) for x in ev], params={k: P[k] for k in ("r", "m_k", "I_k", "m_a", "l", "I_a", "J_w", "d_w", "zenith_deg")})
    for q, comp, lg, tol, src in [("unstable pole with damping", vals["p"], 4.79, 0.006, "S-model, paper Prop. 1"), ("unstable pole without damping", lam0, 5.22, 0.006, "paper Prop. 1"),
                                  ("NMP zero", z, 5.78, 0.006, "S-model"), ("kappa [rad per m/s^2]", kappa, 0.176, 0.001, "S-model"), ("J_eff", Jeff, 0.1025, 0.0006, "S-model"),
                                  ("J_eff / ball inertia", vals["ratio_J"], 190, 1.5, "paper"), ("continuous K theta", -Kc[1], 76.1, 0.06, "S-model (log gives the magnitude)"),
                                  ("continuous K phidot", -Kc[2], 4.33, 0.006, "S-model"), ("continuous K thetadot", -Kc[3], 11.6, 0.06, "S-model"),
                                  ("closed-loop fast pole", vals["clmin"], -20.3, 0.06, "paper")]:
        check(src, "Setup page / paper", q, comp, lg, tol)
    V["model.lean30"] = vals["a_lean30"]


def build_diag():
    """Derived comparisons from the corrected-engine batteries and the LQR calibration."""
    acc = []
    for sp in SP:
        le = S["flat"]["LQR"][sp]["err"]
        if le:
            V[f"cmp.lqr_acc.{sp}"] = S["flat"]["g12_pure_s0"][sp]["err"] / le
            acc.append(V[f"cmp.lqr_acc.{sp}"])
    V["cmp.lqr_acc_min"], V["cmp.lqr_acc_max"] = min(acc), max(acc)
    mean = lambda f, runs: float(np.mean([f(r) for r in runs]))
    FAM = {"res": ["g12_res_s0", "g12_res_s1"], "dr175": ["g12_dr175_s0", "g12_dr175_s1"], "soft": ["g12_soft175_s0", "g12_soft175_s1"], "pure": ["g12_pure_s0", "g12_pure_s1"], "soft075": ["g13_softrs075_s0"], "ret": ["g16_ret100_s0", "g16_ret100_s1"], "pureret": ["g16_pure_ret100_s0", "g16_pure_ret100_s1"], "ret025": ["g18_ret_rs025_s0", "g18_ret_rs025_s1", "g18_ret_rs025_s2"], "ret025mu": ["g26_ret_rs025_mu01_s0", "g26_ret_rs025_mu01_s1", "g26_ret_rs025_mu01_s2"], "ret025_400": ["g19_ret_rs025_400m_s0"], "ret025depth": ["g20_ret_rs025_depth_s0"], "ret025_12": ["g21_ret_rs025_tau3_s0", "g21_ret_rs025_tau3_s1"], "ret015": ["g18_ret_rs015_s0"], "ret035": ["g18_ret_rs035_s0"], "retsoft": ["g18_ret_soft_s0"], "trk3": ["g17_ret100_trk3_s0"], "dr175f": ["g12_dr175_s0", "g12_dr175_s1"]}
    for fam, runs in FAM.items():
        for sc in SCALES:
            V[f"fam.{fam}.u.{sc}"] = mean(lambda r: S["goalscale"][r][sc]["u"]["success"], runs)
            V[f"fam.{fam}.d.{sc}"] = mean(lambda r: S["goalscale"][r][sc]["d"]["success"], runs)
        V[f"fam.{fam}.time"] = mean(lambda r: S["goal"][r]["u"]["median_time_s"], runs)
        V[f"fam.{fam}.velu"] = mean(lambda r: S["vel_tiles"][r]["fall_u"], runs)
        V[f"fam.{fam}.veld"] = mean(lambda r: S["vel_tiles"][r]["fall_d"], runs)
        V[f"fam.{fam}.flat09"] = mean(lambda r: S["flat"][r]["0.9"]["err"], runs)
        V[f"fam.{fam}.ratio09"] = mean(lambda r: S["flat"][r]["0.9"]["ratio"], runs)
        V[f"fam.{fam}.ratio03"] = mean(lambda r: S["flat"][r]["0.3"]["ratio"], runs)
        V[f"fam.{fam}.ach"] = mean(lambda r: np.mean([S["flat"][r][sp]["ratio"] for sp in SP]), runs)
        V[f"fam.{fam}.achmin"] = min(S["flat"][r][sp]["ratio"] for r in runs for sp in SP)
        V[f"fam.{fam}.achmax"] = max(S["flat"][r][sp]["ratio"] for r in runs for sp in SP)
        V[f"fam.{fam}.errmax"] = max(S["flat"][r][sp]["err"] for r in runs for sp in SP)
        for k in ("step_up", "stairs", "bump", "ramp_up", "flat"):
            V[f"fam.{fam}.k.{k}"] = mean(lambda r: S["goal"][r]["ku"][k], runs)
    pairs = [("res", "g12_res_s0", "g12_res_s1"), ("dr175", "g12_dr175_s0", "g12_dr175_s1"), ("soft", "g12_soft175_s0", "g12_soft175_s1")]
    dif = []
    for fam, a, b in pairs:
        for sc in SCALES:
            dif.append(abs(S["goalscale"][a][sc]["u"]["success"] - S["goalscale"][b][sc]["u"]["success"]))
            dif.append(abs(S["goalscale"][a][sc]["d"]["success"] - S["goalscale"][b][sc]["d"]["success"]))
    V["seeds.goal_maxdiff"] = max(dif)
    V["seeds.flat09_diff"] = abs(S["flat"]["g12_res_s0"]["0.9"]["fall"] - S["flat"]["g12_res_s1"]["0.9"]["fall"])
    V["cmp.relerr_min"] = min(S["flat"][r][sp]["err"] / float(sp) for r in ("g12_res_s0", "g12_res_s1") for sp in SP)
    V["cmp.relerr_max"] = max(S["flat"][r][sp]["err"] / float(sp) for r in ("g12_res_s0", "g12_res_s1") for sp in SP)
    V["cmp.soft_gain_u"] = V["fam.soft.u.1"] - V["fam.res.u.1"]
    V["cmp.soft_gain_d"] = V["fam.soft.d.1"] - V["fam.res.d.1"]
    for fam in ("ret025", "ret025mu"):      # per-seed ranges of the then-recommended families, for the headline sentences
        runs = FAM[fam]
        for k_, f_ in (("u", lambda r: S["goalscale"][r]["1"]["u"]["success"]), ("d", lambda r: S["goalscale"][r]["1"]["d"]["success"]), ("velu", lambda r: S["vel_tiles"][r]["fall_u"]), ("veld", lambda r: S["vel_tiles"][r]["fall_d"])):
            vals = [f_(r) for r in runs]
            V[f"fam.{fam}.{k_}.min"], V[f"fam.{fam}.{k_}.max"] = min(vals), max(vals)
    V["cmp.dr_gain_u"] = max(abs(V[f"fam.dr175.u.{sc}"] - V[f"fam.res.u.{sc}"]) for sc in SCALES)
    V["cmp.dr_gain_d"] = max(abs(V[f"fam.dr175.d.{sc}"] - V[f"fam.res.d.{sc}"]) for sc in SCALES)
    check("S74", "text", "soft gain undisturbed (about +0.03 to +0.05)", V["cmp.soft_gain_u"], 0.04, 0.02)
    check("S74", "text", "seed spread of goal success about 0.02", V["seeds.goal_maxdiff"], 0.02, 0.011)


def collect_values():
    for run, d in S.get("flat", {}).items():
        for sp, c in d.items():
            for k in ("err", "ratio", "fall"):
                if c.get(k) is not None:
                    V[f"flat.{run}.{sp}.{k}"] = c[k]
    for run, g in S.get("goal", {}).items():
        for k in ("fall", "success", "median_time_s"):
            V[f"goal.{run}.u.{k}"] = g["u"][k]
            if g["d"]:
                V[f"goal.{run}.d.{k}"] = g["d"][k]
        for kd in KINDS:
            V[f"goal.{run}.u.k.{kd}"] = g["ku"][kd]
            if g["kd"]:
                V[f"goal.{run}.d.k.{kd}"] = g["kd"][kd]
    for run, v in S.get("vel_tiles", {}).items():
        for k in ("fall_u", "fall_d", "err_u", "err_d"):
            if v[k] is not None:
                V[f"vel.{run}.{k}"] = v[k]
    for p in S.get("load_curve", []):
        V[f"curve.{p['scale']}.fall"] = p["fall"]
        V[f"curve.{p['scale']}.err"] = p["err"]
    po = S.get("pushes_only")
    if po:
        V["pushes.only"], V["pushes.none"], V["pushes.both"] = po["fall"], po["none"], po["both"]
        V["pushes.cost_push"], V["pushes.cost_pull"] = po["fall"] - po["none"], po["both"] - po["fall"]
    if S.get("seeds"):
        V["seeds.maxdiff"] = max(x["diff"] for x in S["seeds"])
        V["seeds.vel_u"], V["seeds.vel_d"] = S["seeds"][0]["diff"], S["seeds"][1]["diff"]
    for i, e in enumerate(S.get("encoder", [])):
        V[f"enc.{e['model']}.R2"], V[f"enc.{e['model']}.mae"], V[f"enc.{e['model']}.rise"], V[f"enc.{e['model']}.drop"] = e["R2_mean"], e["MAE_cm"], e["rise_ge2cm_F1"], e["drop_ge2cm_F1"]
    for k, v in S.get("bench", {}).items():
        V[f"bench.{k}"] = v
    if "bench" in S:
        V["bench.warp4096_k"], V["bench.cpu_k"] = S["bench"]["warp4096"] / 1000, S["bench"]["cpu"] / 1000
    # derived comparisons used in the text
    try:
        V["cmp.err_ratio_09"] = V["flat.g6_trk4_s0.0.9.err"] / V["flat.g7_lqrres_s0.0.9.err"]
        V["cmp.err_ratio_03"] = V["flat.g6_trk4_s0.0.3.err"] / V["flat.g7_lqrres_s0.0.3.err"]
        V["cmp.goal_speedup"] = 1 - np.mean([V["goal.g7_lqrres_s0.u.median_time_s"], V["goal.g7_lqrres_s1.u.median_time_s"]]) / np.mean([V["goal.g2_goal_s0.u.median_time_s"], V["goal.g2_goal_s1.u.median_time_s"]])
        V["cmp.blind_goal_mean"] = np.mean([V["goal.g2_goal_s0.u.success"], V["goal.g2_goal_s1.u.success"]])
        V["cmp.res_goal_mean"] = np.mean([V["goal.g7_lqrres_s0.u.success"], V["goal.g7_lqrres_s1.u.success"]])
        V["cmp.res_goal_d_mean"] = np.mean([V["goal.g7_lqrres_s0.d.success"], V["goal.g7_lqrres_s1.d.success"]])
        V["cmp.blind_goal_d_mean"] = np.mean([V["goal.g2_goal_s0.d.success"], V["goal.g2_goal_s1.d.success"]])
        V["cmp.res_goal_dfall_mean"] = np.mean([V["goal.g7_lqrres_s0.d.fall"], V["goal.g7_lqrres_s1.d.fall"]])
    except KeyError as e:
        pass


TOK = re.compile(r"\{\{([A-Za-z0-9_.\-]+)(?::(\d+))?(?:\*(\d+))?\}\}")
SPAN = re.compile(r'<span class="v" data-k="([^"]+)" data-f="(\d+)" data-m="(\d+)">[^<]*</span>')
MISSING = set()


def fmt_v(k, f, m):
    if k not in V:
        MISSING.add(k)
        return "?"
    return f"{V[k] * m:.{f}f}"


def fill(page):
    p = SITE / page
    s = p.read_text(encoding="utf8")
    s = TOK.sub(lambda m: f'<span class="v" data-k="{m.group(1)}" data-f="{m.group(2) or 2}" data-m="{m.group(3) or 1}">?</span>', s)
    s = SPAN.sub(lambda m: f'<span class="v" data-k="{m.group(1)}" data-f="{m.group(2)}" data-m="{m.group(3)}">{fmt_v(m.group(1), int(m.group(2)), int(m.group(3)))}</span>', s)
    p.write_text(s, encoding="utf8")


def write_values_tex():
    """paper_v2/gen_values.tex: \\gv{key} (2 decimals) and \\gvt{key} (3 decimals) for every value, keys as in V."""
    lines = ["% generated by site/build_data.py; do not edit", r"\makeatletter"]
    for k, v in sorted(V.items()):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        kk = k
        lines.append(r"\expandafter\gdef\csname gv@%s\endcsname{%.2f}" % (kk, v))
        lines.append(r"\expandafter\gdef\csname gvt@%s\endcsname{%.3f}" % (kk, v))
        lines.append(r"\expandafter\gdef\csname gvp@%s\endcsname{%.0f}" % (kk, v * 100))
        lines.append(r"\expandafter\gdef\csname gvo@%s\endcsname{%.1f}" % (kk, v))
    lines += [r"\newcommand{\gv}[1]{\csname gv@\detokenize{#1}\endcsname}", r"\newcommand{\gvt}[1]{\csname gvt@\detokenize{#1}\endcsname}", r"\newcommand{\gvp}[1]{\csname gvp@\detokenize{#1}\endcsname}", r"\newcommand{\gvo}[1]{\csname gvo@\detokenize{#1}\endcsname}", r"\makeatother"]
    (PAPER / "gen_values.tex").write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    if REFRESH:
        refresh_snapshot()
    S["superseded"] = "Only corrected-engine (post S69) results are built; all earlier randomised results are excluded."
    build_model()
    build_world_tables()
    build_flat_table()
    build_goal_vel_tables()
    build_g14_heldout()
    build_perception_arm()
    build_retainer()
    build_tradeoff()
    build_s84_s85()
    build_s87()
    build_s100()
    build_s88()
    build_nav()
    build_s90_s91()
    build_s104()
    build_s105_s116()
    build_s117_s122()
    build_s92_s93()
    build_s94_s95()
    build_s98()
    build_cpu_table()
    build_parity_table()
    build_perception()
    build_diag()
    render("tracking", fig_tracking, (9.2, 3.2), (7.2, 2.6))
    render("terrain_goal", fig_terrain_goal, (8.6, 5.2), (7.0, 4.4))
    render("authority", fig_authority, (9.4, 3.3), (7.2, 2.6))
    render("seeds", fig_seeds, (8.0, 3.8), (6.5, 3.2))
    render("tiles", fig_tiles, (9.4, 4.0), (7.2, 3.2))
    render("arch", fig_arch, (9.4, 3.8), (7.2, 3.0))
    render("robust", fig_robust, (8.4, 3.0), (7.0, 2.5))
    render("retainer_geom", fig_retainer_geom, (6.4, 3.8), (5.0, 3.0))
    render("retainer", fig_retainer, (9.0, 3.3), (7.0, 2.7))
    render("tradeoff", fig_tradeoff, (7.6, 4.4), (6.4, 3.8))
    render("designs_ho", fig_designs_ho, (9.0, 3.4), (7.0, 2.8))
    render("traction", fig_traction, (9.0, 3.4), (7.0, 2.8))
    render("nav", fig_nav, (9.0, 3.4), (7.2, 2.8))
    render("ballsize", fig_ballsize, (9.4, 3.6), (7.2, 3.0))
    render("histgc", fig_histgc, (8.0, 4.0), (6.6, 3.4))
    render("ota", fig_ota, (9.6, 5.4), (7.2, 4.4))
    build_videos()
    copy_featured_clips()
    collect_values()
    write_values_tex()
    for pg in ("generalist.html", "method.html", "index.html", "design.html", "deploy.html"):
        if (SITE / pg).exists():
            fill(pg)
    if MISSING:
        print("MISSING VALUE KEYS:", sorted(MISSING))
    S["checks"] = dict(n=len(CHECKS))
    def _clean(o):      # JSON has no NaN: other pages fetch this file, so non-finite numbers become null
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(float(o)) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    (SITE / "data/study.json").write_text(json.dumps(_clean(S), default=str, indent=1, allow_nan=False), encoding="utf8")
    n, nb = write_report()
    print(f"figures done; {n} checks, {nb} differ")
    for c in CHECKS:
        if abs(c[3] - c[4]) > c[5] + 1e-9:
            print(f"  DIFF {c[0]} | {c[1]} | {c[2]} | data {c[3]:.4f} | log {c[4]}")


if __name__ == "__main__":
    main()
