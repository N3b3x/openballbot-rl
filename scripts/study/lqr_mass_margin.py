"""How much mass error does the nominal discrete LQR gain tolerate? (planar model, design dz_nom, control period 10 ms)
For mass scale s the base mass m_a and the ball mass m_k are scaled by s with the inertias, the COM height l and the gains left at their nominal values
(this mimics the mass randomisation of the simulator). Reports the closed-loop spectral radius of the discrete plant with the nominal gain, and with
alternative gains: designed for the heaviest or lightest plant, the average gain, and a design with a larger control cost R.
  python scripts/study/lqr_mass_margin.py"""
import json, sys
from pathlib import Path
import numpy as np, scipy.linalg as la
from scipy.signal import cont2discrete
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
import ballbot_model as BM
M = json.load(open(ROOT / "ballbot_gym" / "models" / "variants" / "dz_nom.json")); P = dict(M["params"])
DT = 0.01
der = BM.derive_planar()


def plant(s):
    Q = dict(P); Q["m_a"] = P["m_a"] * s; Q["m_k"] = P["m_k"] * s
    A, B, *_ = BM.linearize(der, Q)
    Ad, Bd, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), DT)
    return Ad, Bd


def gain(Ad, Bd, r=1.0, qth=50.0, qphd=1.0, qthd=1.0):
    Qm = np.diag([1e-6, qth, qphd, qthd]) * DT
    X = la.solve_discrete_are(Ad, Bd, Qm, np.array([[r * DT]]))
    return np.linalg.solve(np.array([[r * DT]]) + Bd.T @ X @ Bd, Bd.T @ X @ Ad).ravel()


scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
pl = {s: plant(s) for s in scales}
gains = {"nominal": gain(*pl[1.0]), "designed at 1.4x": gain(*pl[1.4]), "designed at 0.6x": gain(*pl[0.6]),
         "R x3 (soft)": gain(*pl[1.0], r=3.0), "R x10": gain(*pl[1.0], r=10.0)}
gains["average of 0.6x, 1.0x, 1.4x"] = (gain(*pl[0.6]) + gain(*pl[1.0]) + gain(*pl[1.4])) / 3
print("largest closed-loop eigenvalue modulus excluding the marginal position mode (discrete, 10 ms; < 1 stable, > 1 unstable) of each gain on the plant with mass scale s; the open-loop plant is unstable")
print("%-30s" % "gain / s" + " ".join("%5.2f" % s for s in scales))
out = {}
for name, K in gains.items():
    def radius(s):      # the position mode (phi) is marginal for any gain: drop the eigenvalue at 1 and report the largest remaining modulus (> 1 means unstable)
        ev = np.linalg.eigvals(pl[s][0] - pl[s][1] @ K[None, :]); ev = ev[np.argsort(-np.abs(ev))]
        bad = [e for e in ev if abs(e) > 1 + 1e-6]
        if bad:
            return float(abs(bad[0]))
        return float(abs(ev[np.argmin(abs(ev - 1.0)) != np.arange(len(ev))][0]))
    rho = [radius(s) for s in scales]
    out[name] = rho
    print("%-30s" % name + " ".join("%5.3f" % r for r in rho))
print("\nK (nominal):", np.round(gains["nominal"], 3))
json.dump(dict(scales=scales, rho=out, K={k: v.tolist() for k, v in gains.items()}, dt=DT, design="nom"), open(ROOT / "docs" / "research" / "analysis" / "lqr_mass_margin.json", "w"), indent=1)
