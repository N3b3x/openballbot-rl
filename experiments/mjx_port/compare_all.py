import json, numpy as np, sys
import compare
S = compare.OUT if hasattr(compare, "OUT") else None
from common import OUT
L = lambda n: dict(np.load(OUT / f"traj_{n}.npz"))
ref = L("cpu_nom_317")
rows = {}
for lab, n in (("warp_vs_cpu317", "warp_nom"), ("cpu313_vs_cpu317", "cpu_nom_313"), ("cpu317_eps1e-6_vs_cpu317", "cpu_nom_317_eps")):
    rows[lab] = compare.metrics(ref, L(n))
json.dump(rows, open(OUT / "compare_nom.json", "w"), indent=1)
keys = list(rows["warp_vs_cpu317"].keys())
import csv
with open(OUT / "compare_nom.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric"] + list(rows))
    for k in keys: w.writerow([k] + [f"{rows[r].get(k, float('nan')):.4g}" for r in rows])
for k in keys: print(f"{k:34s}", *[f"{rows[r].get(k, float('nan')):10.4g}" for r in rows])
