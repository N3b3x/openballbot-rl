"""Merge per-engine sweeps into one comparison table and compute trace errors (PhysX vs MuJoCo reference).

  python compare.py --px px            # reads sweep_mj.csv, sweep_<px>.csv, traces/*.npz
"""
import argparse

import numpy as np
import pandas as pd

from common import ROOT

OUT = ROOT / "outputs/study/isaac_port"


def trace_err(mj, px, key):
    a, b = np.load(mj, allow_pickle=True)[key], np.load(px, allow_pickle=True)[key]
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return a[:n], b[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", default="px")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    mj = pd.read_csv(OUT / f"sweep_mj{a.tag}.csv")
    px = pd.read_csv(OUT / f"sweep_{a.px}{a.tag}.csv")
    m = mj.merge(px, on="design", suffixes=("_mj", "_px"))
    rows = []
    for _, r in m.iterrows():
        d = r["design"]
        row = dict(design=d)
        for c in ("ratio_0.4", "falls_0.4", "ratio_0.8", "falls_0.8", "ratio_1.1", "falls_1.1", "pushfall_0.5", "pushfall_1.0",
                  "pushfall_1.5", "step_fell", "step_final_ratio"):
            row[c + "_mj"], row[c + "_px"] = r[c + "_mj"], r[c + "_px"]
        # step response: RMS difference of speed magnitude (fraction of command) over the first 3 s
        f1, f2 = OUT / "traces" / f"mj{a.tag}_{d}.npz", OUT / "traces" / f"{a.px}{a.tag}_{d}.npz"
        if f1.exists() and f2.exists():
            z1, z2 = np.load(f1), np.load(f2)
            s1, s2 = z1["step"], z2["step"]
            n = min(len(s1), len(s2))
            if n > 50 and len(s1) and len(s2):
                row["step_rms_diff_m_s"] = float(np.sqrt(np.mean(np.sum((s1[:n] - s2[:n]) ** 2, axis=1))))
            p1, p2 = z1["pulse"], z2["pulse"]
            n = min(len(p1), len(p2))
            # open-loop pulse: tilt-rate proxy = RMS difference in tilt angle [rad] and ball angular rate [rad/s]
            row["pulse_tilt_rms_rad"] = float(np.sqrt(np.mean((p1[:n, 1:3] - p2[:n, 1:3]) ** 2)))
            row["pulse_ballw_rms_rad_s"] = float(np.sqrt(np.mean((p1[:n, 3:6] - p2[:n, 3:6]) ** 2)))
            row["pulse_final_tilt_mj_rad"] = float(np.hypot(*p1[n - 1, 1:3]))
            row["pulse_final_tilt_px_rad"] = float(np.hypot(*p2[n - 1, 1:3]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"comparison_{a.px}{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
