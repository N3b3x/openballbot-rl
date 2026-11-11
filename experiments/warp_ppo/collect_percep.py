"""Collect labelled depth data for pretraining the perception encoder (simulation only).

A competent blind policy drives 2048 worlds over the tiled terrain (all 11 designs, surfaces, pushes and pulls on). Every few control steps we store
  depth    noisy VL53L9-like 18x14 zone ranges (the exact deployable sensor, with range noise and 2 % dropout),
  proprio  the 25-D base observation (tilt, rates, velocity, command, goal features),
  scan     the 7x5 privileged height grid ahead (relative to the height under the ball, divided by 6 cm) = the label,
  haz      hazard labels from the scan: max rise and max drop within 0.1 to 0.5 m ahead in the centre column, in cm.
  python collect_percep.py --policy g2_goal_s0 --out percep_v1 --steps 1800 --every 6
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RUNS = Path(__file__).resolve().parents[2] / "outputs" / "study" / "warp_ppo"
_pa = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--policy"), "g2_goal_s0")
_a0 = json.load(open(_RUNS / _pa / "args.json")) if (_RUNS / _pa / "args.json").exists() else {}
if _a0.get("retainer") and not os.environ.get("RETAINER"):      # robot and contact model of the driving policy (S98, S102)
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
import train_dev as T

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="g2_goal_s0"); ap.add_argument("--out", required=True)
ap.add_argument("--nworld", type=int, default=2048); ap.add_argument("--steps", type=int, default=1800); ap.add_argument("--every", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
DEV = "cuda"
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"
ck = torch.load(RUN / a.policy / "model.pt", map_location=DEV)
designs = ck["designs"]
env = T.GpuBallbot(a.nworld, a.seed, dr=True, design=designs[0], designs=designs if len(designs) > 1 else None, tiles=True, surfaces=True, pull=True,
                   goal_p=ck.get("goal_p", 0.4), sensor="depth", mount="head", hist_k=ck.get("hist", 0), asym=ck.get("asym", False),
                   design_cond=ck.get("ddim", 0) > 0)
net = T.Net(0, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to(DEV)
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")}); net.eval()
mean, var = ck["mean"].to(DEV), ck["var"].to(DEV)
SD = env.sdim                                    # depth zones (252)
if _a0.get("lqr_base", 0) > 0:      # residual policy on the LQR base: rebuild it exactly as in training
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, designs, a_max=_a0.get("lqr_amax", 0.5), r=_a0.get("lqr_r", 1.0)), _a0["lqr_base"]
env.cross_p = _a0.get("cross_focus", 0.0)


def pol(o):
    """The policy is blind: drop the depth columns, keep [obs25 | hist | design | (priv ignored)]."""
    x = torch.cat([o[:, :T.OBS_DIM], o[:, T.OBS_DIM + SD:]], 1)
    with torch.no_grad():
        return net(((x - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)


o = env.noisy(env.reset_obs())
D, P, S = [], [], []
t0 = time.time()
for t in range(a.steps):
    act = pol(o)
    o_next, r, fell, trunc, info = env.step(act)
    done = fell | trunc
    env.reset(done)
    o_next = torch.where(done[:, None], env.reset_obs(), o_next)
    o = env.noisy(o_next)
    if t % a.every == 0 and t > 20:
        D.append(o[:, T.OBS_DIM:T.OBS_DIM + SD].half().cpu()); P.append(o[:, :T.OBS_DIM].half().cpu()); S.append(env.scan_feat().half().cpu())
    if t % 300 == 0:
        print(t, f"{time.time()-t0:.0f}s", flush=True)
D, P, S = torch.cat(D), torch.cat(P), torch.cat(S)
sc = S.float().reshape(-1, len(T.SCAN_FWD), len(T.SCAN_LAT)) * 6.0          # cm
mid = len(T.SCAN_LAT) // 2
near = sc[:, :3, mid]                                                       # forward 0.1, 0.3, 0.5 m, centre column
H = torch.stack([near.max(1).values.clamp_min(0), (-near.min(1).values).clamp_min(0)], 1).half()   # rise, drop in cm
out = RUN / "percep"; out.mkdir(exist_ok=True)
torch.save(dict(depth=D, proprio=P, scan=S, haz=H, designs=designs, policy=a.policy, fwd=T.SCAN_FWD, lat=T.SCAN_LAT), out / f"{a.out}.pt")
print("saved", D.shape, "samples; hazard (rise>=2cm):", float((H[:, 0].float() >= 2).float().mean()), "drop>=2cm:", float((H[:, 1].float() >= 2).float().mean()))
