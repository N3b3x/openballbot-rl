"""Per-terrain-type evaluation on the tiled map. Each world is assigned one tile of the requested kind and driven in a random direction
at a fixed speed for 8 s. success = still upright when it first leaves the 2.9 m tile square (or at the end), fall = tilt/ball-offset failure.
  ..\mjx_port\.venv\Scripts\python.exe eval_tiles.py --run NAME [--speed 0.6] [--dr]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T, tiles as TL
from eval_policy import RUN

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--speed", type=float, default=0.6)
ap.add_argument("--n", type=int, default=64, help="worlds per tile kind"); ap.add_argument("--dr", action="store_true"); ap.add_argument("--surfaces", action="store_true")
ap.add_argument("--tile-seed", type=int, default=0); ap.add_argument("--steps", type=int, default=800)
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda"); sdim = ck.get("sdim", 0)
sensor, mount = {0: ("none", "head"), 35: ("scan", "head"), 252: ("depth", "head"), 504: ("depth", "cones")}[sdim]
net = T.Net(sdim, 3, hdim=ck.get('hdim', 0), pdim=ck.get('pdim', 0), use_priv=False).to("cuda"); net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
kinds = TL.TYPES; K = len(kinds); n = a.n; N = K * n
env = T.GpuBallbot(N, seed=7, dr=a.dr, design="nom", tiles=True, tile_seed=a.tile_seed, sensor=sensor, mount=mount, hist_k=ck.get('hist', 0), asym=ck.get('asym', False), surfaces=a.surfaces)
env.resample, env.max_steps = 10 ** 9, 10 ** 9
env.reset(torch.ones(N, dtype=torch.bool, device="cuda")); env.steps[:] = 0
kid = torch.arange(N, device="cuda") // n
# assign one tile of the right kind to each world (cycle through the tiles of that kind), then place the robot there
tpos = torch.zeros(N, dtype=torch.long, device="cuda")
for k in range(K):
    ids = (env.tile_kind == k).nonzero()[:, 0]
    tpos[kid == k] = ids[torch.arange(n, device="cuda") % len(ids)]
env.cur_tile[:] = tpos
c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device="cuda") * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device="cuda") * 0.4 - 0.2
for j in (0, 10):
    env.qpos[:, j] = xs; env.qpos[:, j + 1] = ys; env.qpos[:, j + 2] = env.hgt(xs, ys) + env.q0[j + 2]
env.qvel[:] = 0; env.warm[:] = 0
ang = torch.rand(N, device="cuda") * 2 * np.pi
env.cmd = torch.stack([a.speed * torch.cos(ang), a.speed * torch.sin(ang), torch.zeros(N, device="cuda")], 1)
o = env.noisy(env.reset_obs())
fell_any = torch.zeros(N, dtype=torch.bool, device="cuda"); done_ok = torch.zeros(N, dtype=torch.bool, device="cuda"); frozen = torch.zeros(N, dtype=torch.bool, device="cuda")
for t in range(a.steps):
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
    new_fall = fell & ~frozen
    fell_any |= new_fall
    ex = trunc & ~frozen & ~fell_any
    done_ok |= ex
    frozen |= fell_any | done_ok
alive_end = ~frozen
res = {}
for k in range(K):
    mk = kid == k
    sev = env.tile_sev[tpos][mk]
    res[kinds[k]] = dict(fall=float(fell_any[mk].float().mean()), crossed=float(done_ok[mk].float().mean()), stayed=float(alive_end[mk].float().mean()))
print(json.dumps(res))
out = RUN / a.run / f"eval_tiles_v{a.speed}.json"
json.dump(dict(run=a.run, speed=a.speed, dr=a.dr, kinds=res), open(out, "w"), indent=1)
print("fall rate by kind:", {k: round(v["fall"], 2) for k, v in res.items()})
