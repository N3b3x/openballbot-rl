import sys, time
import numpy as np
from common import DesignCfg
from backend_px import PXBackend
import scenarios as S
cfg = DesignCfg("nom")
configs = [dict(rho=0.012, n_rollers=20), dict(rho=0.02, n_rollers=16), dict(rho=0.012, n_rollers=12), dict(rho=0.012, n_rollers=20, solver_pos=64, solver_vel=8)]
for kw in configs:
    t0 = time.time()
    try:
        be = PXBackend(cfg, **kw)
        fell, v = S.run_drive(be, cfg, np.array([0.4 * cfg.f, 0, 0]), 400)
        ratio = float(v[-150:, 0].mean() / (0.4 * cfg.f)) if not fell else float("nan")
        print("CFG", kw, "fell", fell, "steps", len(v), "ratio", ratio, "time", round(time.time() - t0, 1), flush=True)
    except Exception as e:
        print("CFG", kw, "ERR", repr(e)[:200], flush=True)
