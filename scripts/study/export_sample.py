"""Copy the small artifacts the notebook needs into outputs/study/sample/ (tracked in git)."""
import shutil
import sys
from pathlib import Path

root = Path("outputs/study")
sample = root / "sample"
sample.mkdir(parents=True, exist_ok=True)
runs = sys.argv[1:] or ["proprio_s0", "depth_frozen_s0"]
for r in runs:
    dst = sample / r
    dst.mkdir(exist_ok=True)
    for f in ("final_model.zip", "vecnormalize.pkl", "run_config.json", "progress.csv"):
        src = root / "runs" / r / f
        if src.exists():
            shutil.copy(src, dst / f)
for name, f in (("h1_results.csv", "results/h1.csv"), ("h2_results.csv", "results/h2.csv")):
    if (root / f).exists():
        shutil.copy(root / f, sample / name)
print("sample dir:", sorted(p.name for p in sample.iterdir()))
