"""Figures/tables for the morphology (design-space) study.
  python scripts/study/make_design_report.py            # analytic figure from design_sweep.csv (+ sim results if present)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "outputs/study/results"
FIG = ROOT / "paper_v2/figures"
FIG.mkdir(exist_ok=True)


def analytic_fig():
    df = pd.read_csv(RES / "design_sweep.csv")
    nom = df[df.name == "d_nominal"].iloc[0]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4))
    # (a) COM height: pole, zero, sensitivity bound
    z = df[(df.s == 1) & (df.kball == 1) & (df.kbal == 1) & (df.J == 0.005) & (df.tau == 30)].sort_values("l")
    ax[0, 0].plot(z.l * 100, z.p, "o-", label="unstable pole $p$")
    ax[0, 0].plot(z.l * 100, z.z, "s-", label="NMP zero $z$")
    ax[0, 0].set_xlabel("body COM height $l$ [cm]"), ax[0, 0].set_ylabel("rate [1/s]"), ax[0, 0].legend(fontsize=7)
    t = ax[0, 0].twinx()
    t.plot(z.l * 100, z.Ms_lb, "k^--", ms=4)
    t.set_ylabel("$\\|S\\|_\\infty \\geq (z+p)/|z-p|$")
    ax[0, 0].set_title("(a) COM height", fontsize=9)
    # (b) scale
    s = df[(df.kball == 1) & (df.kbal == 1) & (df.zb == 0.002) & (df.J == 0.005) & (df.tau == 30)].sort_values("s")
    ax[0, 1].loglog(s.s, s.p, "o-", label="$p$ (model)")
    ax[0, 1].loglog(s.s, nom.p * s.s ** -0.5, "k:", label="$s^{-1/2}$")
    ax[0, 1].set_xlabel("scale $s$"), ax[0, 1].set_ylabel("unstable pole $p$ [1/s]"), ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("(b) geometric scale (Froude)", fontsize=9)
    # (c) lean per acceleration vs ball/body mass ratio
    g = df[(df.s == 1) & (df.kbal == 1) & (df.zb == 0.002) & (df.J == 0.005) & (df.tau == 30)].sort_values("m_k")
    ax[1, 0].semilogx(g.m_k / g.m_a, g.kappa_deg, "o-")
    ax[1, 0].set_xlabel("ball / body mass ratio"), ax[1, 0].set_ylabel("lean per accel. [deg per m/s$^2$]")
    ax[1, 0].set_title("(c) ball mass", fontsize=9)
    # (d) torque authority vs linear recoverable tilt
    ta = df[(df.s == 1) & (df.kball == 1) & (df.kbal == 1) & (df.zb == 0.002) & (df.J == 0.005)].sort_values("tau")
    ax[1, 1].loglog(ta.torque_authority, ta.tilt_domain_deg, "o-")
    ax[1, 1].axhline(20, color="r", ls=":", lw=1)
    ax[1, 1].set_xlabel("torque authority $\\tau_{max}/(m_a g l)$"), ax[1, 1].set_ylabel("LQR recoverable tilt [deg]")
    ax[1, 1].set_title("(d) motor strength", fontsize=9)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_design_analytic.pdf")
    print("saved fig_design_analytic.pdf")


if __name__ == "__main__":
    analytic_fig()
