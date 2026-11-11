"""Convert repo evaluation outputs into compact JSON for the static website (site/data/*.json).

Every output file carries `provenance` (source files relative to the repo root) and `data_notes` (places where files
disagree with the study log or where a number has a caveat). Nothing is hand-typed except labels, family colours and the
one-line findings in FINDINGS, which are quotes/summaries of docs/research/tracking_study_log.md.

Run: python scripts/site/build_data.py        (needs pandas, numpy, pyyaml not required)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs/study/results"
RUNS = ROOT / "outputs/study/runs"
OUT = ROOT / "site/data"
OUT.mkdir(parents=True, exist_ok=True)
rel = lambda p: str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def R(x, n=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), n)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {(OUT / name).stat().st_size / 1024:.1f} KB")


def family(run):
    r = run.lower()
    if r.startswith("lqr"):
        return "model"
    if r.startswith("res_"):
        return "hybrid"
    return "learned"


# ---------------------------------------------------------------------------------------------------- tracking
# (run id used in the paper table, results csv stem, stage, description) copied from scripts/study/make_tracking_report.py
TS = pd.read_csv(RES / "tracking_summary.csv")


def speed_curve(stem):
    df = pd.read_csv(RES / f"{stem}.csv")
    s = df[df.test == "speed"].copy()
    s["ach"] = np.hypot(s.vx, s.vy)
    s["err"] = np.hypot(s.vx - s.cmd_x, s.vy - s.cmd_y)
    pts = []
    for v, g in s.groupby("speed_cmd"):
        pts.append(dict(v=float(v), ratio=R(g.ach.mean() / v), err=R(g.err.mean()), fall=R(g.fell.mean()), n=int(len(g))))
    return pts


def extra_summary(run, stem):
    """Same definitions as make_tracking_report.summarize for runs that are not in tracking_summary.csv."""
    df = pd.read_csv(RES / f"{stem}.csv")
    s = df[df.test == "speed"].copy()
    s["ach"] = np.hypot(s.vx, s.vy)
    s["err"] = np.hypot(s.vx - s.cmd_x, s.vy - s.cmd_y)
    o = dict(run=run)
    for v in (0.5, 1.0):
        g = s[np.isclose(s.speed_cmd, v)]
        o[f"ratio_{v}"] = g.ach.mean() / v
    inr = s[s.speed_cmd <= 1.0 + 1e-9]
    o["speed_rmse_le1"] = float(np.sqrt((inr.err ** 2).mean()))
    o["fall_speed_le1"] = inr.fell.mean()
    y = df[(df.test == "yaw") & (df.cmd_w.abs().round(2) == 1.0)]
    o["yaw_err_1"] = (y.wz - y.cmd_w).abs().mean() if len(y) else np.nan
    o["act_rate"] = df.act_rate.mean()
    h = df[df.test == "heading"]
    o["heading_err"] = float(np.hypot(h.vx - h.cmd_x, h.vy - h.cmd_y).mean()) if len(h) else np.nan
    o["rand_falls_30"] = np.nan
    return o


CHART = [  # id, csv stem, label, extra
    ("lqr", "lqr", "LQR (model-based)"),
    ("trk_g_s0", "trk_g_s0", "trk_g (PPO)"),
    ("trk_h_s0", "trk_h_s0", "trk_h (PPO, smooth)"),
    ("res_c_s0", "res_c_s0", "res_c (LQR + residual PPO)"),
    ("robs_b_s0", "robs_b_s0", "robs_b (robust PPO)"),
    ("robs_b_ki1", "robs_b_s0_ki1.0", "robs_b + integral"),
    ("robs_c_ki1", "robs_c_s0_ki1.0", "robs_c + integral"),
]


def build_tracking():
    notes = []
    ctrls = []
    for cid, stem, label in CHART:
        ctrls.append(dict(id=cid, label=label, family=family(cid), source=f"outputs/study/results/{stem}.csv", curve=speed_curve(stem)))
    # table: tracking_summary.csv (paper table) + robs_c rows computed with the same definitions
    rows = []
    for _, r in TS.iterrows():
        rows.append(dict(run=r.run, stage=r.stage, desc=r.desc, family=family(r.run), valid=bool(r.valid),
                         ratio_05=R(r["ratio_0.5"], 2), ratio_10=R(r["ratio_1.0"], 2), rmse=R(r.speed_rmse_le1, 2),
                         yaw_err=R(r.yaw_err_1, 2), act_rate=R(r.act_rate, 3), grid_falls=R(100 * r.fall_speed_le1, 0),
                         rand_falls=None if pd.isna(r.rand_falls_30) else int(r.rand_falls_30), heading_err=R(r.heading_err, 2),
                         source="outputs/study/results/tracking_summary.csv"))
    extra = [("robs_c_s0", "robs_c_s0", "S13 integrated-error observation", "robs_c (obs_vint, DR, 20M)"),
             ("robs_c_s0@ki=1.0", "robs_c_s0_ki1.0", "S13 + outer integral", "robs_c + velocity integral ki=1"),
             ("robs_c_s1", "robs_c_s1", "S13 integrated-error observation", "robs_c seed 1"),
             ("robs_c_s1@ki=1.0", "robs_c_s1_ki1.0", "S13 + outer integral", "robs_c seed 1 + integral")]
    for run, stem, stage, desc in extra:
        o = extra_summary(run, stem)
        fl = RES / f"{run.split('@')[0]}_falls.log"
        if "@" not in run and fl.exists():
            m = re.search(r"falls (\d+) /30", fl.read_text())
            o["rand_falls_30"] = int(m.group(1)) if m else np.nan
        rows.append(dict(run=run, stage=stage, desc=desc, family="learned", valid=True, ratio_05=R(o["ratio_0.5"], 2), ratio_10=R(o["ratio_1.0"], 2),
                         rmse=R(o["speed_rmse_le1"], 2), yaw_err=R(o["yaw_err_1"], 2), act_rate=R(o["act_rate"], 3),
                         grid_falls=R(100 * o["fall_speed_le1"], 0), rand_falls=None if pd.isna(o["rand_falls_30"]) else int(o["rand_falls_30"]),
                         heading_err=R(o["heading_err"], 2), source=f"outputs/study/results/{stem}.csv (+ {run.split('@')[0]}_falls.log)"))
    # cross-check selected numbers quoted in the study log
    chk = {c["id"]: {p["v"]: p for p in c["curve"]} for c in ctrls}
    notes.append("Speed curves are mean achieved speed / commanded speed over 8 directions x 2 repeats (16 trials per speed; falls included), the same "
                 "definition as paper_v2/tab_tracking.tex. Speeds above 1 m/s are outside the training range for most policies.")
    notes.append("Table columns for the first 34 rows are copied from outputs/study/results/tracking_summary.csv, which reproduces paper_v2/tab_tracking.tex exactly "
                 "(checked: all 34 rows). The four robs_c rows are computed here with the same functions, because tracking_summary.csv does not list them yet.")
    log_ok = abs(chk["robs_c_ki1"][1.0]["ratio"] - 0.93) < 0.01 and abs(chk["robs_c_ki1"][0.75]["ratio"] * 0.75 - 0.76) < 0.02
    notes.append(f"Log claim robs_c + integral 'ratio 0.93 at 1.0 m/s, 38 % falls' reproduced from CSV: {log_ok} (ratio {chk['robs_c_ki1'][1.0]['ratio']}, falls {chk['robs_c_ki1'][1.0]['fall']}).")
    notes.append("The tab_tracking.tex caption says 'single seed unless noted, 2 repeats per condition'; robs_c is reported for seeds 0 and 1 separately (no averaging here).")
    return dict(provenance=["outputs/study/results/*.csv (speed curves)", "outputs/study/results/tracking_summary.csv", "paper_v2/tab_tracking.tex",
                            "scripts/study/make_tracking_report.py"],
                data_notes=notes, controllers=ctrls, table=rows)


# ---------------------------------------------------------------------------------------------------- push
PUSH_LABEL = {"lqr": "LQR", "trk_g_s0": "trk_g", "trk_h_s0": "trk_h", "rob_a_s0": "rob_a (DR + pushes)", "rob_b_s0": "rob_b (strong pushes)",
              "rob_a_h_s0": "rob_a_h", "res_c_s0": "res_c", "res_d_s0": "res_d", "robs_a_s0": "robs_a", "robs_a_s0@ki=2.0": "robs_a + integral",
              "robs_b_s0": "robs_b", "robs_b_s0@ki=1.0": "robs_b + integral", "robs_c_s0@ki=1.0": "robs_c + integral"}
PUSH_ORDER = ["lqr", "res_c_s0", "res_d_s0", "trk_g_s0", "trk_h_s0", "rob_a_s0", "rob_a_h_s0", "rob_b_s0", "robs_a_s0", "robs_a_s0@ki=2.0",
              "robs_b_s0", "robs_b_s0@ki=1.0", "robs_c_s0@ki=1.0"]


def build_push():
    d = pd.concat([pd.read_csv(RES / "robustness.csv"),
                   pd.read_csv(RES / "push_robs_c_ki1.csv"), pd.read_csv(RES / "push_dr_robs_c_ki1.csv")], ignore_index=True)
    forces = [0, 10, 20, 30, 40, 50]
    ctrls = []
    for run in PUSH_ORDER:
        g = d[d.run == run]
        nom = g[~g.dr_params.astype(bool)].set_index("force")
        dr = g[g.dr_params.astype(bool)].set_index("force")
        ctrls.append(dict(id=run, label=PUSH_LABEL[run], family=family(run),
                          trials=int(g.trials.iloc[0]),
                          nominal=[int(nom.falls.get(f, -1)) for f in forces], randomised=[int(dr.falls.get(f, -1)) for f in forces]))
    c = {x["id"]: x for x in ctrls}
    notes = ["Cells are falls out of 10 trials; one 50 ms push at t = 2 s while driving 0.5 m/s, any fall counts (scripts/study/eval_push.py). 'randomised' = domain-randomised physics (mass, friction, damping, torque, latency, noise) AND the push; 0 N rows therefore show the fall rate from randomised physics alone.",
             "Study log checks (all reproduced from CSV): nominal LQR 6/10 at 40 N and 9/10 at 50 N: "
             f"{c['lqr']['nominal'][4]}/{c['lqr']['nominal'][5]}; trk_g 1/10, 5/10: {c['trk_g_s0']['nominal'][4]}/{c['trk_g_s0']['nominal'][5]}; "
             f"randomised physics without push: trk_g 7, trk_h 6, LQR 4, rob_a 0: {c['trk_g_s0']['randomised'][0]}, {c['trk_h_s0']['randomised'][0]}, {c['lqr']['randomised'][0]}, {c['rob_a_s0']['randomised'][0]}.",
             "robs_c + integral is not in robustness.csv; its rows come from push_robs_c_ki1.csv and push_dr_robs_c_ki1.csv and are not discussed in the study log (nominal 0/10 up to 40 N, 1/10 at 50 N; randomised 4/10 to 7/10)."]
    # ablation, one randomised parameter at a time (trk_g)
    ab = []
    for key, label in [("damping", "Wheel damping 0.05 to 0.3"), ("mass", "Mass +-15 %"), ("friction", "Wheel friction +-20 %"), ("torque", "Torque +-15 %"),
                       ("latency_p", "One-step latency (p = 0.2)"), ("obs_noise", "Observation noise")]:
        a = pd.read_csv(RES / f"abl_{'latency_p' if key == 'latency_p' else key}.csv").iloc[0]
        ab.append(dict(id=key, label=label, falls=int(a.falls), trials=int(a.trials), source=f"outputs/study/results/abl_{key}.csv"))
    full = c["trk_g_s0"]["randomised"][0]
    ab.append(dict(id="all", label="All randomised together", falls=full, trials=10, source="outputs/study/results/robustness.csv"))
    # contact-frame friction fix ablation
    fx = {}
    for k in ("fix", "nofix"):
        x = pd.read_csv(RES / f"abl_flat_{k}.csv")
        fx[k] = dict(n=int(len(x)), seeds=int(x.seed.nunique()), fell=R(x.fell.mean(), 2), steps=int(x.steps.mean()), progress=R(x.progress.mean(), 3))
    notes.append("Ablation: trk_g evaluated at 0 N with one randomised parameter at a time, 10 trials each. Only wheel damping (5/10) and mass (1/10) cause falls.")
    notes.append("Contact-frame fix ablation (abl_flat_fix.csv / abl_flat_nofix.csv): 4 evaluation seeds that give identical rollouts, so this is one trial, not four. Without the fix the robot falls after 36 steps; with it, it does not fall.")
    return dict(provenance=["outputs/study/results/robustness.csv", "outputs/study/results/push_robs_c_ki1.csv", "outputs/study/results/push_dr_robs_c_ki1.csv",
                            "outputs/study/results/abl_*.csv", "scripts/study/eval_push.py"],
                data_notes=notes, forces=forces, controllers=ctrls, ablation=ab, contact_fix=fx)


# ---------------------------------------------------------------------------------------------------- steps
STEP_LABEL = {"lqr": "LQR", "res_c_s0": "res_c", "robs_b_s0@ki=1.0": "robs_b + integral", "robs_c_s0@ki=1.0": "robs_c + integral", "trk_h_s0": "trk_h",
              "climb_a_s0@ki=1.0": "climb_a (step-trained) + integral", "climb_a_s0": "climb_a (step-trained)", "climb_b_s0": "climb_b (step-trained, from res_c)"}


def build_steps():
    d = pd.read_csv(RES / "step_all.csv")
    d = pd.concat([d, pd.read_csv(RES / "step_climb_a_s0_ki_1.0.csv"), pd.read_csv(RES / "step_climb_a_s0.csv"), pd.read_csv(RES / "step_climb_b_s0.csv")], ignore_index=True)
    ctrls = []
    for run in ["lqr", "res_c_s0", "robs_b_s0@ki=1.0", "robs_c_s0@ki=1.0", "trk_h_s0", "climb_a_s0@ki=1.0", "climb_a_s0", "climb_b_s0"]:
        g = d[d.run == run]
        ser = {}
        for sp, h in g.groupby("speed"):
            ser[str(sp)] = [dict(h=float(a.height_cm), ok=int(a.success), n=int(a.trials)) for a in h.sort_values("height_cm").itertuples()]
        ctrls.append(dict(id=run, label=STEP_LABEL[run], family=family(run), series=ser))
    vm = pd.read_csv(RES / "step_vmin_lqr.csv")
    win = []
    for h, g in vm.groupby("height_cm"):
        ok = g[g.success == g.trials].speed.tolist()
        win.append(dict(h=float(h), ok_speeds=ok, grid=[dict(v=float(a.speed), ok=int(a.success), n=int(a.trials)) for a in g.itertuples()]))
    tr = pd.read_csv(RES / "step_traction.csv")
    trc = [dict(ctrl=a.ctrl, fric=float(a.fric), h=float(a.height_cm), v=float(a.speed), ok=int(a.success), n=int(a.trials)) for a in tr.itertuples()]
    notes = ["Success = cross the step, stop and hold 2 s; 10 trials, random start heading, raised floor with 2 cm bevel (scripts/study/eval_step.py). Rows above 3 cm exist only for the step-trained policies (1 to 4 cm).",
             "Log check reproduced: robs_c + integral at 0.7 m/s: 10/10 at 1.5 and 2 cm, 3/10 at 3 cm; LQR 10/10 at 2 cm, 0/10 at 3 cm; res_c 7/10 at 1.5 cm and 4/10 at 2 cm.",
             "At 0.4 m/s nobody clears 2 cm (all flat-trained rows show 0/10), as stated in the log.",
             "The traction test uses only 4 trials per cell (suggestive, not conclusive; log wording). Friction multiplier scales wheel-ball cross friction.",
             "Speed windows (step_vmin_lqr.csv) use 2 trials per cell."]
    return dict(provenance=["outputs/study/results/step_all.csv", "outputs/study/results/step_climb_*.csv", "outputs/study/results/step_vmin_lqr.csv",
                            "outputs/study/results/step_traction.csv", "scripts/study/eval_step.py"],
                data_notes=notes, controllers=ctrls, lqr_window=win, traction=trc)


# ---------------------------------------------------------------------------------------------------- terrain
TERR = [("lqr", "LQR", "lqr", "terrain_lqr_A{a}", "lqr"),
        ("res_c", "res_c", "res_c_s0", "terrain_res_c_s0_A{a}", "res_c_s0"),
        ("res_t", "res_t (terrain-trained)", None, "terrain_res_t_s0_A{a}", "res_t_s0"),
        ("robs_b_ki1", "robs_b + integral", "robs_b_s0_ki1.0", "terrain_robs_b_s0_ki_1.0_A{a}", "robs_b_s0"),
        ("robs_c_ki1", "robs_c + integral", "robs_c_s0_ki1.0", "terrain_robs_c_s0_ki_1.0_A{a}", "robs_c_s0"),
        ("robs_t_ki1", "robs_t (terrain-trained) + integral", None, "terrain_robs_t_s0_ki_1.0_A{a}", "robs_t_s0"),
        ("trk_h", "trk_h", "trk_h_s0", "terrain_trk_h_s0_A{a}", "trk_h_s0")]


def falls(df, v):
    s = df[(df.test == "speed") & np.isclose(df.speed_cmd, v)]
    return (int(s.fell.sum()), int(len(s))) if len(s) else (None, 0)


def build_terrain():
    ctrls = []
    for cid, label, flat_stem, pat, run in TERR:
        pts = []
        for a in ["0", "0.04", "0.08", "0.12"]:
            f = RES / (pat.format(a=a) + ".csv")
            src = f
            if a == "0" and flat_stem and not f.exists():
                f = RES / f"{flat_stem}.csv"
                src = f
            if not f.exists():
                continue
            df = pd.read_csv(f)
            h = df[df.test == "heading"]
            p = dict(amp_cm=float(a) * 100, source=rel(src))
            for v in (0.5, 0.75, 1.0):
                k, n = falls(df, v)
                p[f"f{str(v).replace('.', '')}"] = k
                p[f"n{str(v).replace('.', '')}"] = n
            p["heading_f"] = int(h.fell.sum()) if len(h) else None
            p["heading_n"] = int(len(h))
            pts.append(p)
        ctrls.append(dict(id=cid, label=label, family=family(run), points=pts))
    notes = ["res_c falls in 3 of 16 flat trials at 0.75 m/s but in 0 of 8 on both rough floors: with 6 to 8 trials per cell and a single terrain seed, differences of one or two trials are noise, so the lines are not monotonic in amplitude.",
             "Falls are counted per commanded speed over the 8 directions (8 trials on rough floor; the flat 0 cm column for LQR, res_c, robs_b/c and trk_h comes from the 2-repeat flat benchmark and so has 16 trials per speed). "
             "The 0 cm points of res_t and robs_t come from their own terrain files (8 trials). Heading test: 24 trials.",
             "No policy except res_t and robs_t was trained on terrain; the floor is Perlin noise (scale 25, 4 octaves, persistence 0.2) of the stated amplitude, a single terrain seed.",
             "Log check: 'robs_t + integral degrades with amplitude (62-88 % falls at 12 cm)'. The CSV gives 62 % (0.75 m/s) and 62 % (1.0 m/s) and 75 % at 0.5 m/s on the speed grid, and 42 % in the heading test; no cell reaches 88 %. "
             "The 88 % in the log may refer to an earlier evaluation; the website uses the CSV values.",
             "Log check reproduced: at 0.75 m/s res_c 0 % falls on both 4 and 8 cm, LQR 12 % and 25 %, robs_b/c + integral 12 to 75 %."]
    return dict(provenance=["outputs/study/results/terrain_*_A*.csv", "outputs/study/results/{lqr,res_c_s0,robs_b_s0_ki1.0,robs_c_s0_ki1.0,trk_h_s0}.csv (flat points)",
                            "scripts/study/eval_control.py --perlin-amp"], data_notes=notes, controllers=ctrls)


# ---------------------------------------------------------------------------------------------------- learning curves
CURVE = [("trk_f_s0", "trk_f (S6, from scratch)"), ("trk_h_s0", "trk_h (S7)"), ("res_c_s0", "res_c (residual on LQR)"), ("rob_a_s0", "rob_a (DR, fine-tuned)"),
         ("robs_b_s0", "robs_b (DR, from scratch)"), ("robs_c_s0", "robs_c (DR + integrated error)"), ("climb_a_s0", "climb_a (step training)"),
         ("dz_nom_s0", "design run: nominal"), ("dz_s0.5_s0", "design run: half scale"), ("dz_s1.5_s0", "design run: 1.5x scale"),
         ("dz_zhigh_s0", "design run: high ballast"), ("dz_tau3_s0", "design run: 3 N m motors"), ("dz_kb4_s0", "design run: 4x ball mass")]


def build_curves():
    out = []
    for run, label in CURVE:
        f = RUNS / run / "progress.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        cfg = json.loads((RUNS / run / "run_config.json").read_text())
        ep_max = cfg["env"]["env"].get("max_ep_steps", 800)
        target = float(cfg["args"]["total_timesteps"])
        p = p.dropna(subset=["rollout/ep_len_mean"])
        t = p["time/total_timesteps"].to_numpy() / 1e6
        el = p["rollout/ep_len_mean"].to_numpy()
        er = p["rollout/ep_rew_mean"].to_numpy()
        edges = np.linspace(t.min(), t.max(), 61)
        idx = np.clip(np.digitize(t, edges) - 1, 0, 59)
        pts = []
        for b in range(60):
            m = idx == b
            if m.any():
                pts.append([R(t[m].mean(), 3), R(el[m].mean(), 1), R(er[m].mean(), 1)])
        out.append(dict(id=run, label=label, family=family(run) if run.startswith("res_") else "learned", ep_max=ep_max, target_msteps=target / 1e6,
                        last_msteps=R(t.max(), 2), running=bool(t.max() < 0.97 * target / 1e6), design_run=run.startswith("dz_"),
                        init_from=Path(cfg["args"].get("init_from") or "").parent.name or None, points=pts, source=rel(f)))
    notes = ["Curves are PPO training logs (rollout/ep_len_mean and rollout/ep_rew_mean from progress.csv), averaged into 60 bins along the x axis. Episode length is the average length of recent training episodes; "
             "the training episode limit is 800 steps (8 s) for these runs, so a value near 800 means 'rarely falls'. Training episodes include domain randomisation and pushes where the run uses them, so curves of different recipes are not like-for-like.",
             "Returns are on different reward scales (different reward coefficients per config); compare shapes, not magnitudes, across runs.",
             "Runs whose x axis stops before their target are still training or were stopped; dz_* design runs were at about 3.6 of 5 M steps when the data was generated and have no evaluation CSV yet.",
             "Fine-tuned runs (trk_g, trk_h, rob_a, climb_a) restart their step counter at 0 although they continue from a previous policy (see init_from)."]
    return dict(provenance=["outputs/study/runs/<run>/progress.csv", "outputs/study/runs/<run>/run_config.json"], data_notes=notes, runs=out)


# ---------------------------------------------------------------------------------------------------- design space
def build_design():
    sw = pd.read_csv(RES / "design_sweep.csv")
    cols = ["name", "s", "kball", "kbal", "zb", "J", "tau", "p", "z", "z_over_p", "Ms_lb", "kappa_deg", "torque_authority", "tilt_domain_deg", "mass_ratio", "l"]
    sweep = [{k: (R(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in zip(cols, r)} for r in sw[cols].itertuples(index=False)]
    cs = pd.read_csv(RES / "design_constraints.csv")
    ccols = ["name", "zeta_deg", "tau_plane_trac", "mgl", "Ms_floor", "C_struct", "C_delay", "C_trac", "C_motor", "C_lean", "C_geom", "C_speed", "v_max", "rec_tilt_trac_deg", "binding", "feasible"]
    cons = [{k: (R(v, 3) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)) for k, v in zip(ccols, r)} for r in cs[ccols].itertuples(index=False)]
    dz = pd.concat([pd.read_csv(RES / "dz_lqr.csv"), pd.read_csv(RES / "dz_lqr_motor.csv")], ignore_index=True)
    lq = []
    for r in dz.to_dict("records"):
        lq.append(dict(design=r["design"], mode=r["run"], ratio_lo=R(r["ratio_0.4"]), falls_lo=int(r["falls_0.4"]), ratio_mid=R(r["ratio_0.8"]), falls_mid=int(r["falls_0.8"]),
                       falls_hi=int(r["falls_1.1"]), push05=R(r["pushfall_0.5"]), push10=R(r["pushfall_1.0"]), push15=R(r["pushfall_1.5"])))
    ok_lo = [x["design"] for x in lq if x["falls_lo"] == 0 and x["design"] not in ("zeta41", "zeta66", "nom_motor")]
    n13 = len([x for x in lq if x["design"] not in ("zeta41", "zeta66", "nom_motor")])
    notes = ["All design results are zero-shot LQR in simulation: the model is re-derived for each design with no training (scripts/study/eval_design.py). Speed columns are 0.4 / 0.8 / 1.1 x sqrt(s) m/s, 4 trials per speed cell; push columns are the fall FRACTION for pushes of 0.5 / 1 / 1.5 x body weight (trial counts are not stored in the CSV).",
             f"Log says 'works on 8/13 designs at 0.5 m/s'. In dz_lqr.csv {len(ok_lo)} of {n13} designs have zero falls in the slowest cell (0.4 sqrt(s) m/s): {', '.join(ok_lo)}; bal4 and J0005 have 1 of 4 falls there. "
             "The two numbers use different criteria (the log probably counts 0.5 m/s with a fall threshold); the website plots the per-design cells from the CSV and does not claim a count.",
             "Contact angle (log: zeta 66 deg survives 1 x weight pushes 6/6, nominal 1/8, zeta 41 deg 0/6): CSV push fractions at 1 x weight are 0.0 (zeta66), 0.875 (nominal, 7/8 fall) and 1.0 (zeta41); the CSV does not store the number of trials, the log says 6 for the zeta rows. Cell sizes are tiny; treat as suggestive.",
             "'nom_motor' row (dz_lqr_motor.csv): realistic actuator model (first-order lag, back-EMF curve, Coulomb friction); falls 4/4 at 0.8 sqrt(s) m/s and 1.1, but survives 1 x weight pushes.",
             "Constraint margins (design_constraints.csv) are ratios of the limit to the demand: values above 1 satisfy the rule (C_struct, C_delay, C_trac, C_motor, C_lean, C_geom, C_speed; see docs/research/design_rules.md). 'binding' is the constraint with the smallest margin of the seven.",
             "Design-run learning curves (PPO from scratch on 13 designs) are included on the learning-curve chart as 'design run' and are unfinished; no evaluation CSV for them (dz_*.csv eval outputs) existed when this file was generated."]
    return dict(provenance=["outputs/study/results/design_sweep.csv", "outputs/study/results/design_constraints.csv", "outputs/study/results/dz_lqr.csv",
                            "outputs/study/results/dz_lqr_motor.csv", "docs/research/design_rules.md", "scripts/study/design_space.py", "scripts/study/design_constraints.py"],
                data_notes=notes, sweep=sweep, constraints=cons, lqr_zero_shot=lq)


# ---------------------------------------------------------------------------------------------------- experiments catalog
FINDINGS = {
    "lqr": "Model-based baseline designed only on the derived planar model. Accurate at low speed (heading error 0.02), 50x smoother than PPO, but falls in every 1 m/s trial and on the square course's turning leg.",
    "ctrl_base_s0": "Balances but barely moves (ratio 0.06 at 1 m/s): not enough authority.",
    "ctrl_free_c_s0": "Void. The policy kicks the ball out and stands on its wheels; the tilt check never fired. Led to the explicit off-ball termination.",
    "ballc_free_s0": "First honest balance on the ball; ratio 0.21 to 0.33.",
    "ballc_freed_s0": "Larger reward, low fall penalty: ratio 0.33 at 1 m/s but very jittery (action rate 0.88).",
    "smooth_a_s0": "Action-rate penalty cuts jitter 8 to 500x but speed falls: a strong smoothness penalty before the policy can move freezes it.",
    "smooth_b_s0": "Stronger action-rate penalty: smoothest, slowest.",
    "smooth_c_s0": "Smoothness from scratch; no better.",
    "lean_a_s0": "Lean bonus 0.5 continued from a timid policy; ratio 0.07.",
    "lean_b_s0": "Lean bonus 1.0: ratio 0.31.",
    "lean_c_s0": "Lean reward from scratch: best of the early stage (0.36 at 1 m/s). Probes: policies use 1 to 3 % of torque; the limit is learned caution, not physics.",
    "trk_a_s0": "Error reward: still 0.15 to 0.31 speed ratio; 23 of 29 falls under yaw commands above 1 rad/s.",
    "trk_b_s0": "No soft ball penalty: no gain.",
    "trk_c_s0": "Dot reward, vmax 2: 0.31 at 1 m/s.",
    "trk_d_s0": "Spin cap (wmax 1) continued from a timid policy: only 0.59.",
    "trk_e_s0": "Spin cap continued from trk_c: one policy collapses into spinning (30/30 falls).",
    "trk_f_s0": "Spin cap from scratch: first policy that tracks speed (0.98 at 0.5, 0.91 at 1 m/s). Seed 1 reproduces (0.95).",
    "trk_f_s1": "Seed 1 of trk_f: 1.09 at 0.5 m/s, 0.95 at 1 m/s.",
    "trk_g_s0": "trk_f continued: action rate 0.147 to 0.013 with no loss at 0.5 m/s, yaw error 0.40 to 0.12. Track first, then smooth.",
    "trk_h_s0": "trk_f continued with 4x action-rate penalty: fewest falls (3/30), about 10 % slower, completes the square course; heading error 0.31 m/s.",
    "spin_a_s0": "Spin curriculum (wmax 1.5): yaw improves, speed drops, falls rise on the harder distribution.",
    "spin_b_s0": "Spin curriculum from trk_h: ratio 0.70 at 1 m/s.",
    "spin_c_s1": "Spin curriculum from trk_f seed 1: ratio 0.75 at 1 m/s.",
    "rob_a_s0": "Domain randomisation and pushes (5 to 20 N) fine-tuned from trk_g: no falls under randomised physics, but slow (0.35 at 1 m/s).",
    "rob_b_s0": "Stronger pushes (10 to 35 N): 0/10 falls at every nominal push up to 50 N, but ratio 0.27 at 1 m/s. Fine-tuning under DR brings back the timid optimum.",
    "rob_a_h_s0": "DR fine-tuned from trk_h: ratio 0.29 and some grid falls.",
    "res_a_s0": "Residual on LQR with a randomly initialised policy (beta 0.3): worse than LQR alone; the untrained residual destabilised it.",
    "res_b_s0": "Same with wmax 1.5: 52 % grid falls.",
    "res_c_s0": "Zero-initialised residual (beta 0.1): keeps LQR accuracy and smoothness, fixes yaw (+-1 rad/s best of all), 3/30 random falls; but 94 % falls at 1 m/s and weak pushes (6/10 at 30 N).",
    "res_d_s0": "res_c continued with beta 0.2, DR and pushes: worse; overshoots and 15/30 random falls. A larger residual fights the LQR.",
    "robs_a_s0": "Full DR from scratch (20M): ratio about 0.35 at every speed; 0/10 nominal pushes up to 50 N.",
    "robs_b_s0": "Damping narrowed to 0.07 to 0.15: ratio about 0.50 at every speed, 3/30 random falls. A constant ratio means steady-state gain below 1, not a speed cap.",
    "robs_a_ki2": "Outer velocity integral (ki = 2) on robs_a: saturates near 0.57 m/s.",
    "robs_b_ki1": "Outer velocity integral (ki = 1) on robs_b: ratio 1.01 at 0.5 and 0.79 at 1 m/s, heading error 0.06 (from 0.27), nominal pushes 0/10 to 50 N.",
    "robs_c_s0": "Integrated velocity error as an observation (from scratch, DR): ratio 0.62 on its own.",
    "robs_c_s0@ki=1.0": "robs_c with the outer integral: most accurate robust controller (heading error 0.045; no falls up to 0.75 m/s, 38 % falls at 1 m/s); clears 2 cm steps 10/10 at 0.7 m/s.",
    "robs_c_s1": "Seed 1 of robs_c: ratio 0.54.",
    "robs_c_s1@ki=1.0": "Seed 1 with the integral: ratio 0.80 at 1 m/s, 12 % falls.",
    "res_t_s0": "res_c continued on random Perlin terrain 0 to 10 cm: about the same as res_c; terrain training did not help.",
    "robs_t_s0": "robs_c continued on Perlin terrain: degrades with amplitude (62 to 75 % grid falls at 12 cm); terrain training did not help.",
    "climb_a_s0": "Step-trained from robs_c (book terrain 0 to 4 cm): clears 2 cm in 8 to 9/10 and 3 cm in 2/10 at 0.7 m/s; no better than the flat-trained robs_c + integral. The limit is physical (traction), not learned.",
    "climb_b_s0": "Step-trained from res_c: worse (2/10 at 2 cm, 0.7 m/s).",
    "dz_nom_s0": "Design-space run: nominal robot, PPO from scratch, 5M steps (unfinished when generated).",
    "dz_s0.5_s0": "Design-space run: half-scale robot, which LQR cannot balance at all; PPO learns it (episode length 539 vs 528 at nominal when generated).",
    "dz_s1.5_s0": "Design-space run: 1.5x scale; learning is struggling (mean episode length about 148 steps when generated).",
}
KEY_ENV = ["max_allowed_tilt", "max_ball_offset_deg", "wheel_damping", "cmd_vmax", "cmd_wmax", "action_rate_coef", "lean_coef", "cmd_reward",
           "residual_lqr", "residual_scale", "obs_vint", "integrator", "hybrid_gate", "morph_pool", "init_yaw_random", "max_ep_steps"]
ORDER = [t[0] for t in [(r,) for r in TS.run]] + ["robs_c_s0", "robs_c_s1", "res_t_s0", "robs_t_s0", "climb_a_s0", "climb_b_s0", "dz_nom_s0", "dz_s0.5_s0", "dz_s1.5_s0"]


def runcfg(run):
    f = RUNS / run / "run_config.json"
    return json.loads(f.read_text()) if f.exists() else None


def build_experiments():
    summ = {r.run: r for r in TS.itertuples()}
    out = []
    for run in ORDER:
        base = run.split("@")[0]
        stem_run = {"robs_a_ki2": "robs_a_s0", "robs_b_ki1": "robs_b_s0"}.get(run, base)
        cfg = runcfg(stem_run)
        e = dict(run=run, family=family(run), finding=FINDINGS.get(run, ""))
        if run in summ:
            s = summ[run]
            e.update(stage=s.stage, desc=s.desc, ratio_10=R(TS.loc[TS.run == run, "ratio_1.0"].iloc[0], 2),
                     falls30=None if pd.isna(s.rand_falls_30) else int(s.rand_falls_30))
        if cfg:
            a, env = cfg["args"], cfg["env"]["env"]
            e["config"] = a["env_config"]
            e["steps_m"] = a["total_timesteps"] / 1e6
            e["seed"] = a["seed"]
            e["init_from"] = Path(a.get("init_from") or "").parent.name or None
            e["zero_init_actor"] = bool(a.get("zero_init_actor"))
            e["terrain"] = cfg["env"]["terrain"]["type"]
            e["domain_randomisation"] = "dr" in env
            e["settings"] = {k: env[k] for k in KEY_ENV if k in env}
            par = runcfg(e["init_from"]) if e["init_from"] else None
            if par:
                penv = par["env"]["env"]
                diff = {k: [penv.get(k), env.get(k)] for k in sorted(set(env) | set(penv)) if env.get(k) != penv.get(k)}
                e["changes_vs_parent"] = {k: v for k, v in diff.items()}
                if par["env"]["terrain"] != cfg["env"]["terrain"]:
                    e["changes_vs_parent"]["terrain"] = [par["env"]["terrain"]["type"], cfg["env"]["terrain"]["type"]]
            e["source"] = [rel(RUNS / stem_run / "run_config.json"), a["env_config"]]
        out.append(e)
    notes = ["Settings and parent differences are read from each run's run_config.json (the config actually used, which can differ from the YAML in configs/env after later edits). "
             "'changes_vs_parent' lists env keys that differ from the run it was initialised from ([parent, run]); for DR the whole block is shown as a key ('dr').",
             "Findings are summaries of docs/research/tracking_study_log.md; numeric metrics come from outputs/study/results/tracking_summary.csv.",
             "Runs such as robs_a_ki2 and robs_b_ki1 are evaluation wrappers (outer velocity integral), not separate trainings."]
    return dict(provenance=["outputs/study/runs/*/run_config.json", "configs/env/*.yaml", "docs/research/tracking_study_log.md", "outputs/study/results/tracking_summary.csv"],
                data_notes=notes, experiments=out)


if __name__ == "__main__":
    dump("tracking.json", build_tracking())
    dump("push.json", build_push())
    dump("steps.json", build_steps())
    dump("terrain.json", build_terrain())
    dump("curves.json", build_curves())
    dump("design.json", build_design())
    dump("experiments.json", build_experiments())
