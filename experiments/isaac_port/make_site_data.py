"""Collect clips.json, comparison CSVs and benchmarks into outputs/study/isaac_port/site_data.json (same shape as mjx_port)."""
import json

import pandas as pd

from common import ROOT

OUT = ROOT / "outputs/study/isaac_port"
clips = json.load(open(OUT / "videos" / "clips.json"))
bench = {}
for k, f in (("mujoco_cpu_1proc", "bench_mj.json"), ("physx_gpu_small", "bench_px_gpu_small.json"), ("physx_gpu_mid", "bench_px_gpu_mid.json"),
             ("physx_gpu_4096", "bench_px_gpu_4096.json")):
    if (OUT / f).exists():
        bench[k] = json.load(open(OUT / f))
tables = {}
for k, f in (("per_design_mujoco_vs_physx", "comparison_px_r.csv"), ("per_design_mujoco_reduced_protocol", "sweep_mj_r.csv"),
             ("per_design_physx_reduced_protocol", "sweep_px_r.csv"), ("per_design_mujoco_full_protocol", "sweep_mj.csv"),
             ("nominal_recordings_detail", "nominal_detail_recordings.csv")):
    if (OUT / f).exists():
        tables[k] = f"outputs/study/isaac_port/{f}"
site = dict(title="MuJoCo CPU vs NVIDIA PhysX (Isaac Sim 6.2): can the ballbot be ported?", generated_by="experiments/isaac_port",
            note="Simulation only. No hardware claims. PhysX has no anisotropic friction; the PhysX model uses a ring of passive rollers per wheel as an emulation (see REPORT.md).",
            clips=clips, tables=tables, throughput=bench, report="experiments/isaac_port/REPORT.md")
json.dump(site, open(OUT / "site_data.json", "w"), indent=1)
print("clips", len(clips), "tables", list(tables))
