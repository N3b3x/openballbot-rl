"""Collect clips.json + sweep CSVs + benchmarks into outputs/study/mjx_port/site_data.json and per-engine comparison CSVs."""
import json
import numpy as np
import pandas as pd
from common import OUT

res = pd.read_csv(OUT / "sweep_results_all.csv")
par = pd.read_csv(OUT / "sweep_parity_all.csv")
clips = json.load(open(OUT / "videos" / "clips.json"))
bench = {k: json.load(open(OUT / f"bench_{k}.json")) for k in ("warp", "cpu") if (OUT / f"bench_{k}.json").exists()}

# wide per-design table: one row per design, columns per engine
cpu = res[res.engine.str.startswith("cpu")].set_index("design")
wrp = res[res.engine.str.startswith("mujoco_warp")].set_index("design")
cols = ["ratio_0.4", "falls_0.4", "ratio_0.8", "falls_0.8", "ratio_1.1", "falls_1.1", "pushfall_0.5", "pushfall_1.0", "pushfall_1.5",
        "step_peak_ballspeed_mm_s"]
wide = pd.concat({"cpu": cpu[cols], "warp": wrp[cols]}, axis=1)
wide.columns = [f"{e}_{c}" for e, c in wide.columns]
wide = wide.join(par.set_index("design")[["step_forced_norm_err_pct", "speed_0.4_vel_rms_m_s", "speed_0.8_vel_rms_m_s", "speed_0.4_tilt_rms_deg",
                                          "speed_0.8_tilt_rms_deg", "push_1.0_fall_parity_pct", "settle_early_rms_tilt_deg"]])
wide.to_csv(OUT / "sweep_cpu_vs_warp_wide.csv")

outcome = {
    "settle": "Both engines topple (open-loop unstable plant); first 0.6 s agree (early tilt RMS error ~1.9 deg), later timing differs.",
    "torque_step": "Forced ball-speed response agrees to 5% of peak over the 0.3 s pulse; later tilt diverges as the plant is unstable.",
    "lqr_0.4": "Both track 0.4 m/s with no fall; speed ratio 0.99 (CPU) vs 0.98 (Warp).",
    "lqr_0.8": "Both track 0.8 m/s with no fall; speed ratio 1.03 (CPU) vs 1.04 (Warp).",
    "push_survive": "Both recover from a 1x body-weight, 50 ms push in this direction.",
    "push_fall": "Both lose the ball after a 1x body-weight, 50 ms push in this direction.",
    "push_disagree": "Borderline push direction: in the 16-direction sweep CPU falls and Warp recovers (14 vs 12-13 of 16 directions fall in total); in this render both fall. The GPU run is not bit-reproducible, so the outcome of edge cases varies between runs.",
}
items = []
for c in clips:
    items.append(dict(id=f"mjx_{c['name']}", title=c["title"], description=c["description"],
                      engines=["MuJoCo 3.17 CPU (float64, ContactFrameStepper)", "MuJoCo-Warp 3.17 GPU (float32, contact-frame kernel)"],
                      scenario=c["name"], parameters=dict(design="dz_nom", integrator="implicitfast", dt_s=0.002, action_repeat=5,
                                                          controller="LQR (dz_nom gains)" if c["name"] not in ("settle", "torque_step") else "open loop",
                                                          speed_factor=c["speed_factor"]),
                      outcome=outcome.get(c["name"], ""), fell=c["fell"], file=c["file"], poster=c["poster"], size_mb=c["mb"],
                      note="Simulation only. No hardware claims."))
site = dict(title="MuJoCo CPU vs MuJoCo-Warp GPU: physics fidelity check", generated_by="experiments/mjx_port",
            clips=items, tables=dict(per_design_cpu_vs_warp="outputs/study/mjx_port/sweep_cpu_vs_warp_wide.csv",
                                     per_design_results="outputs/study/mjx_port/sweep_results_all.csv",
                                     per_design_parity="outputs/study/mjx_port/sweep_parity_all.csv",
                                     nominal_parity="outputs/study/mjx_port/compare_nom.csv"),
            throughput=bench)
json.dump(site, open(OUT / "site_data.json", "w"), indent=1)
print(wide.round(3).to_string())
