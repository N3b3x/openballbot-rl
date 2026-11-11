"""Terrain evaluation of a Warp-trained policy on the stepped ridge (steps of 1, 2, 3 cm at |y| = 0.5, 1.1, 1.7 m, mirrored).
Worlds start at y=0 with random x and yaw and drive outward (+y or -y) at the commanded speed for 12 s. Reports, per start
direction, the fraction of worlds that stay up and the fraction that got past each edge (ball centre beyond the edge + 0.1 m).
  ..\\mjx_port\\.venv\\Scripts\\python.exe eval_terrain.py --run wpterr_nom_s0 [--speed 0.3] [--flat]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as T
from eval_policy import load, RUN

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--speed", type=float, default=0.3)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--flat", action="store_true", help="no terrain (sanity: same policy on flat floor)")
ap.add_argument("--dr", action="store_true"); ap.add_argument("--surfaces", action="store_true")
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda")
sdim = ck.get("sdim", 0)
sensor, mount = {0: ("none", "head"), 35: ("scan", "head"), 252: ("depth", "head"), 504: ("depth", "cones")}[sdim]
net = T.Net(sdim, 3, hdim=ck.get('hdim', 0), pdim=ck.get('pdim', 0), use_priv=False).to("cuda"); net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var, steps = ck["mean"].to("cuda"), ck["var"].to("cuda"), ck["steps"]
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
T.START_Y = (0.0,)
n = a.n
env = T.GpuBallbot(n, seed=5, dr=a.dr, design="nom", terrain=not a.flat, sensor=sensor, mount=mount, hist_k=ck.get('hist', 0), asym=ck.get('asym', False), surfaces=a.surfaces)
env.resample, env.max_steps = 10 ** 9, 10 ** 9
env.reset(torch.ones(n, dtype=torch.bool, device="cuda")); env.steps[:] = 0
sgn = torch.where(torch.arange(n, device="cuda") % 2 == 0, 1.0, -1.0)
env.cmd = torch.stack([torch.zeros(n, device="cuda"), sgn * a.speed, torch.zeros(n, device="cuda")], 1)
o = env.noisy(env.reset_obs())
fell_any = torch.zeros(n, dtype=torch.bool, device="cuda")
ymax = torch.zeros(n, device="cuda")
for t in range(1200):
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act)
    o = env.noisy(o)
    fell_any |= fell
    ymax = torch.maximum(ymax, env.gxpos[:, env.ball_g, 1] * sgn)
row = dict(run=a.run, sensor=sensor, flat=a.flat, speed=a.speed, train_steps=steps)
for name, mask in (("plus_y", sgn > 0), ("minus_y", sgn < 0)):
    row[name] = dict(stay_up=float((~fell_any[mask]).float().mean()),
                     **{f"past_{e}m": float((ymax[mask] > e + 0.1).float().mean()) for e in T.STEP_Y})
print(json.dumps(row, indent=1))
json.dump(row, open(RUN / a.run / f"eval_terrain{'_flat' if a.flat else ''}.json", "w"), indent=1)
