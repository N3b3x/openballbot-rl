"""Aggregate every velocity-tracking benchmark into one table + paper figures.

Reads outputs/study/results/<run>.csv (from eval_control.py) and <run>_falls.log (random-command
fall probe), writes:
  outputs/study/results/tracking_summary.csv
  paper_v2/tab_tracking.tex
  paper_v2/figures/fig_speed_tracking.pdf, fig_progress.pdf, fig_smooth_vs_track.pdf
Usage: python scripts/study/make_tracking_report.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs/study/results"
PAPER = ROOT / "paper_v2"

# (run, results csv stem, stage, short description) in the order the study happened
RUNS = [
    ("lqr", "lqr", "Model-based", "LQR on the derived planar model"),
    ("ctrl_base_s0", "control_base", "S0 baseline limits", "torque 10, tilt 20 deg"),
    ("ctrl_free_c_s0", "control_free_c", "S1 free limits (exploit)", "torque 30, tilt 70, no ball check"),
    ("ballc_free_s0", "ballc_free_s0", "S2 on-ball", "+ off-ball termination 30 deg"),
    ("ballc_freed_s0", "ballc_freed_s0", "S2 on-ball", "+ larger reward, low fall penalty"),
    ("smooth_a_s0", "smooth_a_s0", "S3 smoothness", "+ action-rate -0.05"),
    ("smooth_b_s0", "smooth_b_s0", "S3 smoothness", "+ action-rate -0.2, 25 deg"),
    ("smooth_c_s0", "smooth_c_s0", "S3 smoothness", "scratch, action-rate -0.05"),
    ("lean_a_s0", "lean_a_s0", "S4 lean reward", "lean bonus 0.5"),
    ("lean_b_s0", "lean_b_s0", "S4 lean reward", "lean bonus 1.0"),
    ("lean_c_s0", "lean_c_s0", "S4 lean reward", "scratch, lean 0.5"),
    ("trk_a_s0", "trk_a_s0", "S5 error reward", "err reward, vmax 1.5, wmax 3"),
    ("trk_b_s0", "trk_b_s0", "S5 error reward", "no soft ball penalty"),
    ("trk_c_s0", "trk_c_s0", "S5 error reward", "dot reward, vmax 2"),
    ("trk_d_s0", "trk_d_s0", "S6 spin cap", "wmax 1, from trk_a"),
    ("trk_e_s0", "trk_e_s0", "S6 spin cap", "wmax 1, from trk_c"),
    ("trk_f_s0", "trk_f_s0", "S6 spin cap", "wmax 1, scratch"),
    ("trk_f_s1", "trk_f_s1", "S6 spin cap", "wmax 1, scratch, seed 1"),
    ("trk_g_s0", "trk_g_s0", "S7 refine", "trk_f continued"),
    ("trk_h_s0", "trk_h_s0", "S7 refine", "trk_f + action-rate -0.08"),
    ("spin_a_s0", "spin_a_s0", "S8 spin curriculum", "wmax 1.5, from trk_g"),
    ("spin_b_s0", "spin_b_s0", "S8 spin curriculum", "wmax 1.5, from trk_h"),
    ("spin_c_s1", "spin_c_s1", "S8 spin curriculum", "wmax 1.5, from trk_f_s1"),
    ("rob_a_s0", "rob_a_s0", "S9 robustness", "DR + pushes 5-20 N, from trk_g"),
    ("rob_b_s0", "rob_b_s0", "S9 robustness", "DR + pushes 10-35 N, from trk_g"),
    ("rob_a_h_s0", "rob_a_h_s0", "S9 robustness", "DR + pushes 5-20 N, from trk_h"),
    ("res_a_s0", "res_a_s0", "S10 residual", "LQR + 0.3 PPO, vmax 1.5"),
    ("res_b_s0", "res_b_s0", "S10 residual", "LQR + 0.3 PPO, vmax 1.5, wmax 1.5"),
    ("res_c_s0", "res_c_s0", "S10 residual", "LQR + 0.1 PPO, zero-init, vmax 1"),
    ("res_d_s0", "res_d_s0", "S10 residual", "res_c + beta 0.2 + DR + pushes"),
    ("robs_a_s0", "robs_a_s0", "S11 DR from scratch", "full DR, 20M"),
    ("robs_b_s0", "robs_b_s0", "S11 DR from scratch", "damping 0.07-0.15, 20M"),
    ("robs_a_ki2", "robs_a_s0_ki2.0", "S12 + outer integral", "robs_a + velocity integral ki=2"),
    ("robs_b_ki1", "robs_b_s0_ki1.0", "S12 + outer integral", "robs_b + velocity integral ki=1"),
]
# runs whose policies left the ball (no off-ball check at training time): flagged, not ranked
INVALID = {"ctrl_free_c_s0"}


def summarize(run, stem):
    f = RES / f"{stem}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    s = df[df.test == "speed"].copy()
    s["ach"] = np.hypot(s.vx, s.vy)
    s["err"] = np.hypot(s.vx - s.cmd_x, s.vy - s.cmd_y)
    out = dict(run=run)
    for v in (0.5, 1.0):
        g = s[np.isclose(s.speed_cmd, v)]
        out[f"ratio_{v}"] = g.ach.mean() / v if len(g) else np.nan
        out[f"err_{v}"] = g.err.mean() if len(g) else np.nan
    inr = s[s.speed_cmd <= 1.0 + 1e-9]
    out["speed_rmse_le1"] = float(np.sqrt((inr.err ** 2).mean()))
    out["fall_speed_le1"] = inr.fell.mean()
    y = df[(df.test == "yaw") & (df.cmd_w.abs().round(2) == 1.0)]
    out["yaw_err_1"] = (y.wz - y.cmd_w).abs().mean() if len(y) else np.nan
    out["act_rate"] = df.act_rate.mean() if "act_rate" in df else np.nan
    out["off_ball_max"] = df.off_ball.max() if "off_ball" in df else np.nan
    h = df[df.test == "heading"]
    out["heading_err"] = float(np.hypot(h.vx - h.cmd_x, h.vy - h.cmd_y).mean()) if len(h) else np.nan
    st = df[df.test == "step"]
    out["rise_s"] = st.rise_s.mean() if "rise_s" in st and len(st) else np.nan
    fl = RES / f"{run}_falls.log"
    out["rand_falls_30"] = np.nan
    if fl.exists():
        m = re.search(r"falls (\d+) /30", fl.read_text())
        if m:
            out["rand_falls_30"] = int(m.group(1))
    return out


def main():
    rows = []
    for run, stem, stage, desc in RUNS:
        r = summarize(run, stem)
        if r:
            r.update(stage=stage, desc=desc, valid=run not in INVALID)
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(RES / "tracking_summary.csv", index=False)
    print(T[["run", "ratio_0.5", "ratio_1.0", "speed_rmse_le1", "yaw_err_1", "act_rate", "rand_falls_30"]].round(3).to_string())

    # LaTeX table
    PAPER.mkdir(exist_ok=True)
    (PAPER / "figures").mkdir(exist_ok=True)
    fmt = lambda x, d=2: "--" if pd.isna(x) else f"{x:.{d}f}"
    lines = [r"\begin{table*}[t]\centering\small",
             r"\caption{Flat-ground velocity-tracking benchmark for every policy, in study order (single seed unless noted, deterministic policy, 2 repeats per condition). "
             r"Ratio: achieved/commanded speed averaged over 8 directions. RMSE: planar velocity error over all commands $\le$1\,m/s. "
             r"Yaw err: $|\omega_z-\omega_{cmd}|$ at $\pm$1\,rad/s. Act.\ rate: mean $\lVert a_t-a_{t-1}\rVert^2$ (lower is smoother). "
             r"Grid falls: share of speed-grid trials ($\le$1\,m/s) that fell. Falls/30: falls in 30 episodes of random training-distribution commands. Heading err: 0.5\,m/s commands from random start headings. $^\dagger$Policy leaves the ball (invalid).}",
             r"\label{tab:tracking}",
             r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llcccccccc}\toprule",
             r"Stage & Run (change) & Ratio@0.5 & Ratio@1.0 & RMSE [m/s] & Yaw err [rad/s] & Act.\ rate & Grid falls & Falls/30 & Heading err [m/s]\\\midrule"]
    for _, r in T.iterrows():
        name = r.run.replace("_", r"\_") + ("$^\\dagger$" if not r.valid else "")
        lines.append(f"{r.stage} & {name} ({r.desc.replace('_', chr(92) + '_')}) & {fmt(r['ratio_0.5'])} & {fmt(r['ratio_1.0'])} & {fmt(r.speed_rmse_le1)} & "
                     f"{fmt(r.yaw_err_1)} & {fmt(r.act_rate, 3)} & {fmt(100 * r.fall_speed_le1, 0)}{chr(92)}% &{'--' if pd.isna(r.rand_falls_30) else int(r.rand_falls_30)} & {fmt(r.get('heading_err', np.nan))}\\\\")
    lines += [r"\bottomrule\end{tabular}}\end{table*}"]
    (PAPER / "tab_tracking.tex").write_text("\n".join(lines).replace("deg", r"$^\circ$"), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]

    # Fig 1: achieved vs commanded speed for representative stages
    pick = [("ctrl_base_s0", "S0 baseline"), ("ballc_freed_s0", "S2 on-ball"), ("lean_c_s0", "S4 lean reward"),
            ("trk_f_s0", "S6 spin cap"), ("trk_g_s0", "S7 refined"), ("trk_h_s0", "S7 smooth")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot([0, 1.5], [0, 1.5], color="#9a9a9a", lw=1, ls="--", label="ideal")
    for (run, lab), c in zip(pick, C):
        stem = dict((a, b) for a, b, *_ in RUNS)[run]
        f = RES / f"{stem}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s = s[s.test == "speed"]
        g = s.assign(ach=np.hypot(s.vx, s.vy)).groupby("speed_cmd").ach.mean()
        g = g[g.index <= 1.5]
        ax.plot(g.index, g.values, marker="o", ms=3, lw=1.6, color=c, label=lab)
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("achieved speed [m/s]")
    ax.set_xlim(0, 1.55); ax.set_ylim(0, 1.55)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(PAPER / "figures/fig_speed_tracking.pdf"); fig.savefig(RES / "fig_speed_tracking.png", dpi=200)
    plt.close(fig)

    # Fig 2: progression of speed ratio at 1 m/s across runs (valid only)
    V = T[T.valid].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    stages = list(dict.fromkeys(V.stage))
    cols = {st: C[i % len(C)] for i, st in enumerate(stages)}
    ax.bar(range(len(V)), V["ratio_1.0"], color=[cols[s] for s in V.stage], width=0.7)
    ax.axhline(1.0, color="#9a9a9a", lw=1, ls="--")
    ax.set_xticks(range(len(V)))
    ax.set_xticklabels(V.run.str.replace("_s0", "").str.replace("_s1", " (s1)"), rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("achieved / commanded\nat 1.0 m/s")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[s], label=s) for s in stages], frameon=False, fontsize=6, ncol=4, loc="upper left")
    ax.set_ylim(0, 1.25)
    fig.tight_layout(); fig.savefig(PAPER / "figures/fig_progress.pdf"); fig.savefig(RES / "fig_progress.png", dpi=200)
    plt.close(fig)

    # Fig 3: smoothness vs tracking trade-off
    W = V.dropna(subset=["act_rate"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for st in stages:
        g = W[W.stage == st]
        ax.scatter(g.act_rate, g.speed_rmse_le1, s=22, color=cols[st], label=st, edgecolor="white", linewidth=0.8, zorder=3)
    for _, r in W.iterrows():
        if r.run in ("trk_g_s0", "trk_h_s0", "trk_f_s0", "lean_c_s0", "ballc_freed_s0"):
            ax.annotate(r.run.replace("_s0", ""), (r.act_rate, r.speed_rmse_le1), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("action rate (log, lower = smoother)")
    ax.set_ylabel("speed RMSE, cmd $\\leq$ 1 m/s [m/s]")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(); fig.savefig(PAPER / "figures/fig_smooth_vs_track.pdf"); fig.savefig(RES / "fig_smooth_vs_track.png", dpi=200)
    plt.close(fig)
    # Fig 4: top-down path following on the square course (planned vs driven)
    cases = [("trk_g_square_guide0", "trk_g, velocity only"), ("trk_h_square_guide0", "trk_h, velocity only"),
             ("trk_h_square_guide1.5", "trk_h + position guidance k=1.5")]
    cases = [(c, l) for c, l in cases if (ROOT / f"outputs/study/preview/{c}_traj.npz").exists()]
    if cases:
        fig, axs = plt.subplots(1, len(cases), figsize=(2.35 * len(cases), 2.5), sharex=True, sharey=True)
        axs = np.atleast_1d(axs)
        for ax, (c, lab) in zip(axs, cases):
            z = np.load(ROOT / f"outputs/study/preview/{c}_traj.npz")
            ax.plot(z["plan"][:, 0], z["plan"][:, 1], color="#9a9a9a", lw=1.2, ls="--", label="planned")
            ax.plot(z["real"][:, 0], z["real"][:, 1], color=C[0], lw=1.6, label="driven")
            ax.plot(*z["real"][-1], marker="x" if bool(z["fell"]) else "o", color=C[1] if bool(z["fell"]) else C[0], ms=6)
            ax.set_title(f"{lab}\nmean err {z['path_err'].mean():.2f} m" + (" (fell)" if bool(z["fell"]) else ""), fontsize=7)
            ax.set_aspect("equal"); ax.set_xlabel("x [m]")
        axs[0].set_ylabel("y [m]"); axs[0].legend(frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(PAPER / "figures/fig_paths.pdf"); fig.savefig(RES / "fig_paths.png", dpi=200)
        plt.close(fig)
    # robustness: push recovery (nominal and randomised physics)
    import glob
    P = [pd.read_csv(f) for f in glob.glob(str(RES / "push_*.csv"))]
    if P:
        P = pd.concat(P).drop_duplicates(subset=["run", "force", "dr_params"], keep="last")
        P.to_csv(RES / "robustness.csv", index=False)
        names = {"lqr": "LQR (derived)", "trk_g_s0": r"PPO trk\_g", "trk_h_s0": r"PPO trk\_h",
                 "rob_a_s0": r"PPO rob\_a (DR-trained)", "rob_b_s0": r"PPO rob\_b (strong DR)",
                 "robs_a_s0": r"PPO robs\_a (DR, scratch)", "robs_b_s0": r"PPO robs\_b (narrow DR, scratch)",
                 "res_c_s0": r"Residual res\_c", "res_d_s0": r"Residual res\_d",
                 "robs_a_s0@ki=2.0": r"robs\_a + integral", "robs_b_s0@ki=1.0": r"robs\_b + integral"}
        forces = sorted(P.force.unique())
        L = [r"\begin{table}[t]\centering\small",
             r"\caption{Push recovery while driving at 0.5\,m/s: falls out of 10 trials with a single 50\,ms push from a random "
             r"direction at $t=2$\,s (any fall counts). Top: nominal robot. Bottom: physical parameters randomised as in training.}",
             r"\label{tab:push}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * len(forces) + r"}\toprule",
             "Controller & " + " & ".join(f"{f:.0f}" + r"\,N" for f in forces) + r"\\\midrule"]
        for drp in (False, True):
            if drp:
                L.append(r"\midrule\multicolumn{" + str(len(forces) + 1) + r"}{l}{\emph{randomised physics}}\\")
            for run, nm in names.items():
                g = P[(P.run == run) & (P.dr_params == drp)].set_index("force")
                if len(g):
                    L.append(nm + " & " + " & ".join(str(int(g.falls[f])) if f in g.index else "--" for f in forces) + r"\\")
        L.append(r"\bottomrule\end{tabular}}\end{table}")
        (PAPER / "tab_push.tex").write_text("\n".join(L), encoding="utf-8")

    # step climbing: LQR climb window (speed x height) and controller comparison at 0.7 m/s
    fw = RES / "step_vmin_lqr.csv"
    fa = RES / "step_all.csv"
    if fw.exists() and fa.exists():
        W = pd.read_csv(fw)
        A_ = pd.read_csv(fa)
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
        piv = W.pivot_table(index="height_cm", columns="speed", values="success") / W.trials.max()
        im = ax[0].imshow(piv.values, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1,
                          extent=[piv.columns.min() - 0.05, piv.columns.max() + 0.05, piv.index.min() - 0.25, piv.index.max() + 0.25])
        hs = np.linspace(0.8, 3.6, 30)
        ax[0].plot(np.sqrt(2 * 9.81 * hs / 100), hs, color="#eb6834", lw=1.4, ls="--", label=r"$v=\sqrt{2gh}$")
        ax[0].set_xlabel("approach speed [m/s]"); ax[0].set_ylabel("step height [cm]")
        ax[0].set_title("LQR: climb-and-hold success", fontsize=7); ax[0].legend(frameon=False, fontsize=6, loc="upper right")
        ax[0].grid(False)
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
        lab = {"lqr": "LQR", "res_c_s0": "residual res_c", "robs_c_s0@ki=1.0": "robs_c + integral",
               "robs_b_s0@ki=1.0": "robs_b + integral", "trk_h_s0": "PPO trk_h"}
        for i, (run, nm) in enumerate(lab.items()):
            g = A_[(A_.run == run) & (A_.speed == 0.7)].sort_values("height_cm")
            if len(g):
                ax[1].plot(g.height_cm, g.success / g.trials, marker="o", ms=3, lw=1.5, color=C[i % len(C)], label=nm)
        ax[1].set_xlabel("step height [cm]"); ax[1].set_ylabel("success rate at 0.7 m/s")
        ax[1].set_ylim(-0.05, 1.05); ax[1].legend(frameon=False, fontsize=6, loc="lower left")
        fig.tight_layout(); fig.savefig(PAPER / "figures/fig_step.pdf"); fig.savefig(RES / "fig_step.png", dpi=200)
        plt.close(fig)
    print("wrote", RES / "tracking_summary.csv", PAPER / "tab_tracking.tex")


if __name__ == "__main__":
    main()
