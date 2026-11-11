"""Does MuJoCo CPU (float64) also go non-finite with the 0.001 free-rolling wheel-ball friction while standing? (S98 open question)
Standing with the planar LQR (common.LQRBatch) for 20 s, zero command, a few trials per design with a small random initial tilt.
  python cpu_stand_check.py [--designs nom,kb4,bal025] [--trials 8] [--seconds 20]     (PAIR_MU1=0.01 to test the raised value)
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mjx_port"))
from common import *  # noqa
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--designs", default="nom,kb4,bal025"); ap.add_argument("--trials", type=int, default=8); ap.add_argument("--seconds", type=float, default=20.0)
a = ap.parse_args()
rng = np.random.default_rng(0)
print("PAIR_MU1 =", os.environ.get("PAIR_MU1", "(xml value)"))
for dn in a.designs.split(","):
    des = Design(dn); m = load_cpu_model(design=des)
    print(dn, "pair_friction[0] =", m.pair_friction[:, :2].tolist())
    nonfin, falls, maxtilt = 0, 0, []
    for k in range(a.trials):
        d = mujoco.MjData(m)
        q0, v0 = reset_state(m, yaw=rng.uniform(0, 2 * np.pi))
        eps = rng.normal(0, 0.01, 2)      # about 0.6 deg initial tilt
        q0[BQ + 3:BQ + 7] = np.array([1.0, eps[0] / 2, eps[1] / 2, 0.0]); q0[BQ + 3:BQ + 7] /= np.linalg.norm(q0[BQ + 3:BQ + 7])
        d.qpos[:] = q0; d.qvel[:] = v0
        mujoco.mj_forward(m, d)
        lqr = LQRBatch(1, tau_max=des.tau)
        base = m.body("base").id
        bad = False; tmax = 0.0
        for t in range(int(a.seconds * 100)):
            cv = d.cvel[base][3:].reshape(1, 3)
            tau = lqr(d.qpos.reshape(1, -1), d.qvel.reshape(1, -1), cv, np.zeros((1, 3)))
            d.ctrl[:] = -tau[0]
            for _ in range(ACTION_REPEAT):
                mujoco.mj_step(m, d)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                bad = True; break
            tl = tilt_deg(d.qpos.reshape(1, -1))[0]; tmax = max(tmax, tl)
            if tl > 70:
                break
        nonfin += bad; falls += (not bad) and tmax > 70; maxtilt.append(tmax)
    print(f"  {dn}: non-finite {nonfin}/{a.trials}, tipped {falls}/{a.trials}, survivors max tilt median {np.median(maxtilt):.1f} deg")
