"""Design sweep: same protocol on CPU MuJoCo (ContactFrameStepper) and MuJoCo-Warp for each design variant.
python sweep.py nom zeta66 ... [--cpu-only] ; writes outputs/study/mjx_port/sweep_<design>_{cpu,warp}.npz and sweep_results.csv"""
import sys, csv, time
import numpy as np
from warp_sim import *    # noqa
import cpu_ref, runner, compare

KEYS = []

def row_from(res, d, specs):
    r = {}
    for sp in (0.4, 0.8, 1.1):
        n = f"speed_{sp}"
        ratio, falls = runner.summarize(dict(vel=res[n]["vel"], fell=res[n]["fell"]), specs[n]["cmd"])
        r[f"ratio_{sp}"], r[f"falls_{sp}"] = ratio, falls
    for al in (0.5, 1.0, 1.5):
        r[f"pushfall_{al}"] = float(res[f"push_{al}"]["fell"].mean())
    st = res["step"]
    r["step_peak_ballspeed_mm_s"] = 1e3 * float(np.hypot(st[:, 0], st[:, 1]).max())
    return r

if __name__ == "__main__":
    lowprio()
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    cpu_only = "--cpu-only" in sys.argv
    out_rows = []; cmp_rows = []
    for nm in names:
        d = Design(nm)
        m = load_cpu_model(design=d)
        specs = test_specs(d.m_total, d.f)
        print("==", nm, "tau", d.tau, "mass", d.m_total, "f", d.f, flush=True)
        t0 = time.time()
        rc = cpu_ref.run_all(lambda N: cpu_ref.CpuBackend(m, N), m, specs, log=lambda *a: None, design=d)
        tc = time.time() - t0
        np.savez(OUT / f"sweep_{nm}_cpu.npz", **cpu_ref.flatten(rc))
        out_rows.append(dict(design=nm, engine="cpu_mujoco_3.17", wall_s=round(tc, 1), **row_from(rc, d, specs)))
        print(out_rows[-1], flush=True)
        if cpu_only: continue
        t0 = time.time()
        try:
            rw = cpu_ref.run_all(lambda N: WarpBallbot(m, N), m, specs, log=lambda *a: None, design=d)
        except Exception as e:
            print("WARP FAILED", nm, repr(e)[:300]); out_rows.append(dict(design=nm, engine="mujoco_warp", error=repr(e)[:200])); continue
        tw = time.time() - t0
        np.savez(OUT / f"sweep_{nm}_warp.npz", **cpu_ref.flatten(rw))
        out_rows.append(dict(design=nm, engine="mujoco_warp_3.17_gpu", wall_s=round(tw, 1), **row_from(rw, d, specs)))
        print(out_rows[-1], flush=True)
        M = compare.metrics(cpu_ref.flatten(rc), cpu_ref.flatten(rw), d.m_total)
        cmp_rows.append(dict(design=nm, **M))
    def dump(rows, fn):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(OUT / fn, "w", newline="") as f:
            w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    tag = "_".join(names[:1]) if len(names) == 1 else "all"
    dump(out_rows, f"sweep_results_{tag}.csv"); 
    if cmp_rows: dump(cmp_rows, f"sweep_parity_{tag}.csv")
