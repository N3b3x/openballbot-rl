"""Turn analysis CSVs into LaTeX tables for the paper."""
import sys
from pathlib import Path

import pandas as pd

A = Path("outputs/study/analysis")
P = Path("paper")
NAMES = {"proprio": "Proprio", "depth_frozen": "Prop.+depth", "depth_e2e": "Prop.+depth (e2e)"}


def pct(x):
    return f"{max(0.0, 100 * x):.0f}"


def h1_table():
    t = pd.read_csv(A / "h1_summary.csv")
    rows = []
    for _, r in t.sort_values(["amplitude", "variant"]).iterrows():
        rows.append(f"{r.amplitude:.2f} & {NAMES.get(r.variant, r.variant)} & {int(r.n)} & "
                    f"{pct(r.success)} [{pct(r.success_lo)}, {pct(r.success_hi)}] & "
                    f"{pct(r.fall)} [{pct(r.fall_lo)}, {pct(r.fall_hi)}] & {r.ret:.1f} & "
                    + ("--" if pd.isna(r.ttg_s) else f"{r.ttg_s:.1f}") + " \\\\")
    body = "\n".join(rows)
    (P / "tab_h1.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\caption{H1: held-out terrains (clean depth). Success and fall rates in \\% with 95\\% "
        "Wilson intervals; pooled over training seeds. Time-to-goal is the mean over successful episodes.}\\label{tab:h1}\n"
        "\\small\\begin{tabular}{clcccrc}\\toprule\n"
        "Amplitude & Policy & $n$ & Success [\\%] & Fall [\\%] & Return & TTG [s]\\\\\\midrule\n"
        + body + "\n\\bottomrule\\end{tabular}\\end{table*}\n")


def tests_table():
    t = pd.read_csv(A / "paired_tests.csv")
    rows = []
    for _, r in t.iterrows():
        p = r.wilcoxon_p
        ps = "<0.001" if p < 0.001 else f"{p:.3f}"
        rows.append(f"{r.test.replace('_', ' ')} & {int(r.n_terrains)} & {r.mean_diff:+.3f} & "
                    f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {ps} \\\\")
    (P / "tab_tests.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\caption{Paired comparisons over identical held-out terrains (mean difference, bootstrap "
        "95\\% CI, Wilcoxon signed-rank $p$).}\\label{tab:tests}\n\\scriptsize\\begin{tabular}{lcccc}\\toprule\n"
        "Test & $n$ & $\\Delta$ & 95\\% CI & $p$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def h2_table():
    f = A / "h2_by_condition.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    rows = []
    for _, r in t.iterrows():
        rows.append(f"{NAMES.get(r.variant, r.variant)} & {r.condition.replace('_', ' ')} & {int(r.n)} & "
                    f"{pct(r.success)} & {pct(r.fall)} & {r.ret:.1f} \\\\")
    (P / "tab_h2.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\caption{H2: depth corruption at test time (all amplitudes pooled).}\\label{tab:h2}\n"
        "\\small\\begin{tabular}{llcccc}\\toprule\nPolicy & Condition & $n$ & Success [\\%] & Fall [\\%] & Return\\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


if __name__ == "__main__":
    h1_table()
    tests_table()
    h2_table()
    print("tables written to", P)
