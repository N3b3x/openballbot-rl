"""Run a list of ring.py configs, 2 worker processes at low priority, collect rows into outputs/study/ring_wheel/<out>.csv
  python sweep.py stage1
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
HERE = Path(__file__).parent
OUT = HERE.parents[1] / "outputs/study/ring_wheel"
stage = sys.argv[1]
cfgs = []
if stage == "stage1":
    cfgs.append(["--ideal"])
    for n in (12, 24):
        for soft in (0, 1):
            for damp in (0, 1e-4, 1e-3):
                for ph in (0, 0.5):
                    cfgs.append(["--n", n, "--rho", 0.012, "--damp", damp, "--soft", soft, "--phase", ph])
elif stage == "stage2":
    for n in (12, 24):
        for rho in (0.006, 0.009, 0.016):
            for soft in (0, 1):
                for ph in (0, 0.5):
                    cfgs.append(["--n", n, "--rho", rho, "--damp", 0, "--soft", soft, "--phase", ph])
elif stage == "stage3":      # larger rim radius (real omni wheels are bigger than the 25 mm idealised capsule)
    for n in (12, 24):
        for rim in (0.035, 0.05):
            for soft in (0, 1):
                for ph in (0, 0.5):
                    cfgs.append(["--n", n, "--rho", 0.012, "--rim", rim, "--damp", 0, "--soft", soft, "--phase", ph])

def run(c):
    args = [sys.executable, str(HERE / "ring.py")] + [str(x) for x in c]
    r = subprocess.run(args, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("ROW "):
            return json.loads(line[4:])
    return dict(name=" ".join(map(str, c)), err=(r.stderr or r.stdout)[-300:])

with ThreadPoolExecutor(2) as ex:
    rows = list(ex.map(run, cfgs))
df = pd.DataFrame(rows)
df.to_csv(OUT / f"{stage}.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(df.drop(columns=[c for c in df.columns if c.startswith("fellstep") or c in ("mu",)]).round(3).to_string())
