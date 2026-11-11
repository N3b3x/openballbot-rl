"""Where does the simulator state go non-finite while the robot stands still? (simulation only, MuJoCo-Warp GPU)

Standing 20 s tests (S94 to S97) count a non-finite state as a fall, and the traces show upright robots (tilt 1 to 2 deg) whose
state becomes NaN in one control step. This script runs the standing test with the LQR alone (or a policy) and, at the first
non-finite step of each world, records the design, the time, the previous step's contact count and the previous |qvel|, |qacc|,
plus which geom pairs were in contact. Environment knobs: RETAINER=off, FIX_FRAMES=0, NCONMAX=32 (read by train_dev).
  TRAIN_MOD=train_dev PUSH_SCALE=0 TILES_FLAT=1 EVAL_NOEXIT=1 python nan_hunt.py --run g18_ret_rs025_s2 --steps 2000 --n 16 [--lqr-only]
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=16); ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--lqr-only", action="store_true"); ap.add_argument("--designs", default=None, help="comma list; default: the run's designs")
ap.add_argument("--out", default="nanhunt")
ap.add_argument("--no-dr", action="store_true", help="no domain randomisation and no floor material classes")
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda")
_a0 = json.load(open(RUN / a.run / "args.json"))
if os.environ.get("RETAINER") == "off":
    os.environ.pop("RETAINER")
elif not os.environ.get("RETAINER") and _a0.get("retainer"):
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):      # contact model of the run (S101 fix)
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
sdim = ck.get("sdim", 0)
designs = a.designs.split(",") if a.designs else (ck.get("designs") or ["nom"])
net = T.Net(sdim, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to("cuda")
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
kinds = TL.TYPES; K = len(kinds); n = a.n; D = len(designs); N = D * K * n
env = T.GpuBallbot(N, seed=7, dr=not a.no_dr, design=designs[0], designs=designs if D > 1 or ck.get("ddim", 0) else None, tiles=True, tile_seed=0,
                   sensor="none", mount="head", hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=not a.no_dr, pull=False,
                   goal_p=1.0, design_cond=ck.get("ddim", 0) > 0)
dev = "cuda"
env.push_scale = float(os.environ.get("PUSH_SCALE", "0"))
if _a0.get("lqr_base", 0) > 0:
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, designs, a_max=_a0.get("lqr_amax", 0.5), r=_a0.get("lqr_r", 1.0)), _a0["lqr_base"]
env.resample, env.max_steps = 10 ** 9, 10 ** 9
env.reset(torch.ones(N, dtype=torch.bool, device=dev)); env.steps[:] = 0
w = torch.arange(N, device=dev); did, kid = w // (K * n), (w // n) % K
if env.K > 1:
    env.set_design(w, did)
tpos = torch.zeros(N, dtype=torch.long, device=dev)
for k in range(K):
    ids = (env.tile_kind == k).nonzero()[:, 0]
    tpos[kid == k] = ids[torch.arange(N, device=dev)[kid == k] % len(ids)]
env.cur_tile[:] = tpos
c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device=dev) * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device=dev) * 0.4 - 0.2
q = env.q0_tab[env.didx].clone(); q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
env.goal_mode[:] = False; env.cmd = torch.zeros(N, 3, device=dev)
o = env.noisy(env.reset_obs())

d = env.be.d; m = env.be.m
mjm = env.be.mjm
import mujoco
gname = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}" for g in range(mjm.ngeom)]

def contacts():
    nac = int(d.nacon.numpy()[0])
    geom = d.contact.geom.numpy()[:nac]; wid = d.contact.worldid.numpy()[:nac]
    dist = d.contact.dist.numpy()[:nac]
    return geom, wid, dist

def wpt(x):
    return torch.utils.dlpack.from_dlpack(wp_to_torch(x)) if False else torch.as_tensor(x.numpy(), device="cpu")

alive = torch.ones(N, dtype=torch.bool, device=dev)
prev = dict(ncon=np.zeros(N, dtype=int), qvel=np.zeros(N), qacc=np.zeros(N), pairs=[[] for _ in range(N)], mindist=np.full(N, 9.0))
events = []
first = True
for t in range(a.steps):
    # snapshot of the state before this step
    geom, wid, dist = contacts()
    ncon = np.bincount(wid, minlength=N)
    qv = env.qvel.abs().max(1).values.cpu().numpy()
    qacc = np.abs(d.qacc.numpy()).max(1)
    if first:
        counts = {}
        for g, wi in zip(geom, wid):
            counts[(gname[g[0]], gname[g[1]])] = counts.get((gname[g[0]], gname[g[1]]), 0) + 1
        print("contact pairs at t=0 (total over worlds):", {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:12]})
        act_pairs = {}
        for g, wi, dd in zip(geom, wid, dist):
            if dd < 0.001:
                act_pairs[(gname[g[0]], gname[g[1]])] = act_pairs.get((gname[g[0]], gname[g[1]]), 0) + 1
        print("active (dist < 1 mm) pairs at t=0 per world:", {f"{k[0]}|{k[1]}": round(v / N, 2) for k, v in sorted(act_pairs.items(), key=lambda kv: -kv[1])[:12]})
        print("ncon per world at t=0: min %d median %d max %d (nconmax %d)" % (ncon.min(), np.median(ncon), ncon.max(), d.naconmax // N))
        first = False
    with torch.no_grad():
        act = torch.zeros(N, 3, device=dev) if a.lqr_only else pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
    bad = ~(torch.isfinite(env.qpos).all(1) & torch.isfinite(env.qvel).all(1))
    new = alive & bad
    if new.any():
        geom2, wid2, dist2 = contacts()
        for wi in new.nonzero()[:, 0].tolist():
            pairs = [f"{gname[g[0]]}|{gname[g[1]]}" for g, ww in zip(geom, wid) if ww == wi]
            mind = float(min([dd for dd, ww in zip(dist, wid) if ww == wi], default=9.0))
            events.append(dict(world=wi, design=designs[int(did[wi])], kind=kinds[int(kid[wi])], t_s=t / 100, ncon_before=int(ncon[wi]), ncon_at_nan=int((wid2 == wi).sum()),
                               max_abs_qvel_before=float(qv[wi]), max_abs_qacc_before=float(qacc[wi]), min_dist_before=mind, pairs_before=pairs,
                               tilt_before=float(info["tilt"][wi]) if torch.isfinite(info["tilt"][wi]) else None))
    alive = alive & ~fell
print(json.dumps(dict(run=a.run, lqr_only=a.lqr_only, retainer=os.environ.get("RETAINER"), fix_frames=os.environ.get("FIX_FRAMES", "1"), nconmax=int(d.naconmax // N),
                      worlds=N, nonfinite=len(events), nonfinite_rate=len(events) / N, falls_total=int((~alive).sum()),
                      by_design={dn: sum(e["design"] == dn for e in events) / (K * n) for dn in designs},
                      ncon_before_hist=dict(zip(*[list(map(int, x)) for x in np.unique([e["ncon_before"] for e in events], return_counts=True)])) if events else {},
                      ncon_at_nan_hist=dict(zip(*[list(map(int, x)) for x in np.unique([e["ncon_at_nan"] for e in events], return_counts=True)])) if events else {},
                      qvel_before_median=float(np.median([e["max_abs_qvel_before"] for e in events])) if events else None,
                      qacc_before_median=float(np.median([e["max_abs_qacc_before"] for e in events])) if events else None,
                      qacc_before_p90=float(np.percentile([e["max_abs_qacc_before"] for e in events], 90)) if events else None,
                      t_median_s=float(np.median([e["t_s"] for e in events])) if events else None,
                      example=events[:3])))
(RUN / a.run / f"{a.out}.json").write_text(json.dumps(events, indent=1))
