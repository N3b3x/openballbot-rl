"""Compare CPU vs Warp trajectory files. Usage: python compare.py <cpu.npz> <warp.npz> [label]  -> dict of metrics (also importable)."""
import sys, json
import numpy as np
from common import *  # noqa
import runner


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def metrics(cpu, wrp, m_total=None):
    M = {}
    c, w = cpu["settle"], wrp["settle"]
    M["settle_rms_base_z_mm"] = 1e3 * rms(c[:, 0], w[:, 0])
    M["settle_rms_ball_z_mm"] = 1e3 * rms(c[:, 1], w[:, 1])
    M["settle_rms_tilt_deg"] = rms(c[:, 2], w[:, 2])
    M["settle_final_base_z_cpu_mm"] = 1e3 * c[-1, 0]; M["settle_final_base_z_warp_mm"] = 1e3 * w[-1, 0]
    # open loop is unstable (growth ~ e^(6/s) in tilt): compare the early window and the fitted growth rate separately
    e = slice(0, 60)
    M["settle_early_rms_tilt_deg"] = rms(c[e, 2], w[e, 2]); M["settle_early_rms_base_z_mm"] = 1e3 * rms(c[e, 0], w[e, 0])
    for tag, x in (("a", c), ("b", w)):
        t = np.arange(20, 80) * 0.01; y = np.log(np.maximum(x[20:80, 2], 1e-9))
        M[f"settle_tilt_growth_rate_{tag}_per_s"] = float(np.polyfit(t, y, 1)[0])
    c, w = cpu["step"], wrp["step"]
    fr = slice(50, 80)    # forced response window: 0.3 s torque pulse
    M["step_forced_rms_ballspeed_mm_s"] = 1e3 * rms(np.hypot(c[fr, 0], c[fr, 1]), np.hypot(w[fr, 0], w[fr, 1]))
    M["step_forced_rms_wheelw_rad_s"] = rms(c[fr, 3:6], w[fr, 3:6])
    M["step_forced_norm_err_pct"] = 100 * M["step_forced_rms_ballspeed_mm_s"] / (1e3 * np.hypot(c[fr, 0], c[fr, 1]).max())
    sel = slice(int(runner.STEP_T0 / 0.01), None)
    sp = lambda x: np.hypot(x[:, 0], x[:, 1])
    M["step_rms_ballspeed_mm_s"] = 1e3 * rms(sp(c)[sel], sp(w)[sel])
    M["step_peak_ballspeed_cpu_mm_s"] = 1e3 * sp(c).max(); M["step_peak_ballspeed_warp_mm_s"] = 1e3 * sp(w).max()
    M["step_rms_tilt_deg"] = rms(c[:, 6], w[:, 6])
    M["step_rms_wheelw_rad_s"] = rms(c[:, 3:6], w[:, 3:6])
    M["step_norm_err_pct"] = 100 * M["step_rms_ballspeed_mm_s"] / (1e3 * max(sp(c).max(), 1e-9))
    for name in [k.split("__")[0] for k in cpu if k.endswith("__fell")]:
        fc, fw = cpu[f"{name}__fell"], wrp[f"{name}__fell"]
        M[f"{name}_falls_cpu"] = int(fc.sum()); M[f"{name}_falls_warp"] = int(fw.sum())
        M[f"{name}_fall_parity_pct"] = 100 * float((fc == fw).mean())
        if name.startswith("speed"):
            cmd = test_specs(m_total or 4.021)[name]["cmd"]
            for tag, r in (("cpu", cpu), ("warp", wrp)):
                res = dict(vel=r[f"{name}__vel"], fell=r[f"{name}__fell"])
                M[f"{name}_ratio_{tag}"] = runner.summarize(res, cmd)[0]
            both = ~(fc | fw)
            if both.any():
                M[f"{name}_vel_rms_m_s"] = rms(cpu[f"{name}__vel"][:, both], wrp[f"{name}__vel"][:, both])
                M[f"{name}_tilt_rms_deg"] = rms(cpu[f"{name}__tilt"][:, both], wrp[f"{name}__tilt"][:, both])
        else:
            both = ~(fc | fw)
            if both.any():
                M[f"{name}_tilt_rms_deg_nofall"] = rms(cpu[f"{name}__tilt"][:, both], wrp[f"{name}__tilt"][:, both])
                M[f"{name}_peaktilt_cpu_deg"] = float(cpu[f"{name}__tilt"].max(0)[both].mean())
                M[f"{name}_peaktilt_warp_deg"] = float(wrp[f"{name}__tilt"].max(0)[both].mean())
            # fall-time agreement for runs that fell in both
            ff = fc & fw
            if ff.any():
                M[f"{name}_falltime_diff_s"] = float(np.nanmean(np.abs(cpu[f"{name}__fall_t"][ff] - wrp[f"{name}__fall_t"][ff])))
    return M


if __name__ == "__main__":
    c = dict(np.load(sys.argv[1])); w = dict(np.load(sys.argv[2]))
    M = metrics(c, w)
    for k, v in M.items(): print(f"{k:34s} {v:.4g}" if isinstance(v, float) else f"{k:34s} {v}")
    json.dump(M, open(sys.argv[2].replace(".npz", "_metrics.json"), "w"), indent=1)
