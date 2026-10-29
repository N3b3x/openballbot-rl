"""Aggregate evaluation CSVs into tables, paired statistics and figures for the paper."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)
COLORS = {"proprio": "#4c72b0", "depth_frozen": "#dd8452", "depth_e2e": "#55a868"}
LABELS = {"proprio": "Proprio-only", "depth_frozen": "Proprio + depth (frozen enc.)",
          "depth_e2e": "Proprio + depth (end-to-end)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(x, n_boot=5000):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def summarize(df):
    rows = []
    for (variant, cond, amp), g in df.groupby(["variant", "condition", "amplitude"]):
        n = len(g)
        lo, hi = wilson(g.success.sum(), n)
        flo, fhi = wilson(g.fell.sum(), n)
        ttg = g.loc[g.success, "time_to_goal"] * 0.01  # seconds (100 Hz decisions)
        rows.append(dict(variant=variant, condition=cond, amplitude=amp, n=n,
                         success=g.success.mean(), success_lo=lo, success_hi=hi,
                         fall=g.fell.mean(), fall_lo=flo, fall_hi=fhi,
                         ret=g.ret.mean(), ret_sd=g.ret.std(),
                         ttg_s=ttg.mean() if len(ttg) else np.nan,
                         speed=g.speed.mean()))
    return pd.DataFrame(rows)


def per_terrain(df, variant, cond="clean", col="success"):
    g = df[(df.variant == variant) & (df.condition == cond)]
    return g.groupby(["amplitude", "seed"])[col].mean()  # mean over training seeds (runs)


def paired_test(a, b, name):
    """a, b: Series indexed by terrain. Returns dict with mean diff, bootstrap CI, Wilcoxon p."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    d = (j.iloc[:, 0] - j.iloc[:, 1]).values
    lo, hi = boot_ci(d)
    try:
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(test=name, n_terrains=len(d), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", required=True)
    ap.add_argument("--h2", default=None)
    ap.add_argument("--h3", default=None, help="CSV from evaluating goal_relative / goal_direction runs")
    ap.add_argument("--out", default="outputs/study/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    h1 = pd.read_csv(args.h1)
    tab = summarize(h1[h1.condition == "clean"])
    tab.to_csv(out / "h1_summary.csv", index=False)
    print(tab.round(3).to_string())

    tests = []
    variants = sorted(h1.variant.unique())
    if "proprio" in variants:
        for v in variants:
            if v == "proprio":
                continue
            for col in ("success", "fell"):
                t = paired_test(per_terrain(h1, v, col=col), per_terrain(h1, "proprio", col=col),
                                f"H1 {col}: {v} - proprio")
                tests.append(t)
            t = paired_test(per_terrain(h1, v, col="ret"), per_terrain(h1, "proprio", col="ret"),
                            f"H1 return: {v} - proprio")
            tests.append(t)

    # H1 figure: success / fall vs terrain difficulty
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for v in variants:
        s = tab[tab.variant == v].sort_values("amplitude")
        for a, col, ttl in ((ax[0], "success", "Success rate"), (ax[1], "fall", "Fall rate")):
            a.errorbar(s.amplitude, s[col], yerr=[np.clip(s[col] - s[col + "_lo"], 0, None), np.clip(s[col + "_hi"] - s[col], 0, None)],
                       marker="o", capsize=3, color=COLORS.get(v), label=LABELS.get(v, v))
            a.set_title(ttl)
            a.set_xlabel("Terrain amplitude (held-out seeds)")
            a.set_ylim(-0.02, 1.02)
            a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "h1_success_fall.png", dpi=180)
    plt.close(fig)

    if args.h2:
        h2 = pd.read_csv(args.h2)
        s2 = summarize(h2)
        s2.to_csv(out / "h2_summary.csv", index=False)
        agg = h2.groupby(["variant", "condition"]).agg(success=("success", "mean"), fall=("fell", "mean"),
                                                       ret=("ret", "mean"), n=("success", "size")).reset_index()
        agg.to_csv(out / "h2_by_condition.csv", index=False)
        print(agg.round(3).to_string())
        for v in sorted(h2.variant.unique()):
            if v == "proprio":
                continue
            base = per_terrain(h2, v, "clean", "success")
            for cond in sorted(h2.condition.unique()):
                if cond == "clean":
                    continue
                tests.append(paired_test(per_terrain(h2, v, cond, "success"), base,
                                         f"H2 success: {v} {cond} - clean"))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        sub = agg[agg.variant != "proprio"]
        order = ["clean"] + [c for c in sub.condition.unique() if c != "clean"]
        x = np.arange(len(order))
        for i, v in enumerate(sorted(sub.variant.unique())):
            vals = [sub[(sub.variant == v) & (sub.condition == c)].success.mean() for c in order]
            ax.bar(x + i * 0.38, vals, 0.38, color=COLORS.get(v), label=LABELS.get(v, v))
        ref = h2[h2.variant == "proprio"]
        if len(ref):
            ax.axhline(ref.success.mean(), color=COLORS["proprio"], ls="--", label="Proprio-only (clean)")
        ax.set_xticks(x + 0.19 * (sub.variant.nunique() - 1))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out / "figures" / "h2_robustness.png", dpi=180)
        plt.close(fig)

    if args.h3:
        h3 = pd.read_csv(args.h3)
        g = h3.groupby("run").agg(n=("success", "size"), success=("success", "mean"), fall=("fell", "mean"),
                                  ret=("ret", "mean"), ttg_steps=("time_to_goal", "mean")).reset_index()
        g["ttg_s"] = g.ttg_steps * 0.01
        g.to_csv(out / "h3_summary.csv", index=False)
        print(g.round(3).to_string())
        runs = sorted(h3.run.unique())
        if len(runs) == 2:
            a, b = runs
            pa = h3[h3.run == a].set_index(["amplitude", "seed"])
            pb = h3[h3.run == b].set_index(["amplitude", "seed"])
            tests.append(paired_test(pa.success.astype(float), pb.success.astype(float), f"H3 success: {a} - {b}"))
            both = pa.success & pb.success
            if both.sum() > 1:
                tests.append(paired_test(pa.loc[both.index[both], "time_to_goal"] * 0.01,
                                         pb.loc[both.index[both], "time_to_goal"] * 0.01,
                                         f"H3 time-to-goal [s] (both succeed): {a} - {b}"))

    pd.DataFrame(tests).to_csv(out / "paired_tests.csv", index=False)
    print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()


def plot_training(run_dirs, out_png, title=None, smooth=5):
    """Learning curves (mean episode return and length) from SB3 progress.csv files."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for rd in run_dirs:
        rd = Path(rd)
        df = pd.read_csv(rd / "progress.csv")
        x = df["time/total_timesteps"]
        label = rd.name
        v = label.rsplit("_s", 1)[0]
        for a, col, ttl in ((ax[0], "rollout/ep_rew_mean", "Mean episode return"),
                            (ax[1], "rollout/ep_len_mean", "Mean episode length (decisions)")):
            y = df[col].rolling(smooth, min_periods=1).mean()
            a.plot(x, y, color=COLORS.get(v), alpha=0.9, label=label)
            a.set_title(ttl)
            a.set_xlabel("Training decisions")
            a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
