"""Nominal-design detail comparison from the recorded trajectories (rec/mj_* vs rec/px_*): settling, torque pulse, tracking, push, forces."""
import json
import numpy as np
import pandas as pd

from common import ROOT

OUT = ROOT / "outputs/study/isaac_port"
rows = []


def load(e, d, s):
    f = OUT / "rec" / f"{e}_{d}_{s}.npz"
    return {k: v for k, v in np.load(f).items()} if f.exists() else None


def fn_stats(r):
    c = r["contacts"]
    m = c[:, :, 8] > 0.5
    return (float(c[:, :, 6][m].mean()) if m.any() else np.nan, float(c[:, :, 7][m].mean()) if m.any() else np.nan,
            float(m.sum(1).mean()))


for d, scs in (("nom", ["settle", "pulse", "track04", "track08", "push"]), ("zeta66", ["push"]), ("zeta41", ["push", "track08"])):
    for s in scs:
        a, b = load("mj", d, s), load("px", d, s)
        if a is None or b is None:
            continue
        n = min(len(a["t"]), len(b["t"]))
        row = dict(design=d, scenario=s, frames_mj=len(a["t"]), frames_px=len(b["t"]), fell_at_mj=int(a["fell_at"]), fell_at_px=int(b["fell_at"]))
        row["tilt_rms_diff_deg"] = float(np.sqrt(np.mean((a["tilt"][:n] - b["tilt"][:n]) ** 2)))
        row["speed_rms_diff_m_s"] = float(np.sqrt(np.mean((a["speed"][:n] - b["speed"][:n]) ** 2)))
        row["ball_xy_end_diff_m"] = float(np.linalg.norm(a["ball_center"][n - 1, :2] - b["ball_center"][n - 1, :2]))
        row["ball_z_mean_mj"], row["ball_z_mean_px"] = float(a["ball_center"][:, 2].mean()), float(b["ball_center"][:, 2].mean())
        for tag, r in (("mj", a), ("px", b)):
            fnm, ftm, nc = fn_stats(r)
            row[f"mean_normal_N_{tag}"], row[f"mean_friction_N_{tag}"], row[f"mean_contacts_{tag}"] = fnm, ftm, nc
            row[f"final_speed_{tag}"] = float(r["speed"][-150:].mean())
            row[f"tilt_rms_{tag}"] = float(np.sqrt(np.mean(r["tilt"] ** 2)))
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "nominal_detail_recordings.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).T.to_string())
