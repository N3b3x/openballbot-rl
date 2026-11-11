"""Ground-truth driving logs for the state-estimation study (simulation only, MuJoCo-Warp GPU).

The recommended controller drives with the training command distribution (random velocity commands resampled every 3 s, half the worlds
chasing chained goals), DR on, noise on, arrestors and contact model from the run, either on flat ground or across the terrain tiles.
Per world and control step (100 Hz) we store what a real robot's sensors would see, without noise (noise is added later by eval_odom.py,
so one log serves many sensor models), and the ground truth:
  wheel_q, wheel_w   wheel angles (rad) and rates (rad/s): the motor encoders
  quat, gyro         base orientation (w, x, y, z) and base angular velocity in the body frame: what an IMU integrates and measures
  base_v             base linear velocity (world); the IMU accelerometer is derived from it by differencing (eval_odom.py)
  ball_p, ball_v     ball centre position and velocity (world): the truth the estimator must recover
  ball_w             ball angular velocity (world)
  alive              False from the first fall on (the rest of that world is ignored)
  python collect_odom.py --run g26_ret_rs025_mu01_s1 --where flat --n 64 --seconds 60 --out odom_flat
"""
import argparse, importlib, json, math, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
RUNS = Path(__file__).resolve().parents[2] / "outputs" / "study" / "warp_ppo"
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="g26_ret_rs025_mu01_s1"); ap.add_argument("--where", choices=["flat", "tiles"], default="flat")
ap.add_argument("--design", default="nom"); ap.add_argument("--n", type=int, default=64); ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--no-dr", action="store_true"); ap.add_argument("--vmax", type=float, default=0.4, help="velocity-command cap (m/s) for realistic driving; training uses up to 1.2"); ap.add_argument("--no-surfaces", action="store_true")
a = ap.parse_args()
_a0 = json.load(open(RUNS / a.run / "args.json"))
if os.environ.get("RETAINER") == "off":
    os.environ.pop("RETAINER")
elif _a0.get("retainer") and not os.environ.get("RETAINER"):
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
T = importlib.import_module("train_dev"); import tiles as TL
ck = torch.load(RUNS / a.run / "model.pt", map_location="cuda")
designs = ck.get("designs") or ["nom"]
net = T.Net(0, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to("cuda")
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
N = a.n; dev = "cuda"
env = T.GpuBallbot(N, seed=a.seed, dr=not a.no_dr, design=designs[0], designs=designs if len(designs) > 1 or ck.get("ddim", 0) else None, tiles=True, tile_kinds=(["flat"] if a.where == "flat" else None),
                   hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=not a.no_surfaces, pull=False, goal_p=_a0.get("goal_p", 0.5), design_cond=ck.get("ddim", 0) > 0)
env.push_scale = 0.0
from lqr_gpu import GpuLQR
env.lqr, env.rs = GpuLQR(env, designs, a_max=_a0.get("lqr_amax", 0.5), r=_a0.get("lqr_r", 1.0)), _a0["lqr_base"]
TL.EXIT_R = 99.0
env.max_steps = 10 ** 9; env.resample = 300; env.cmd_vmax = a.vmax
env.reset(torch.ones(N, dtype=torch.bool, device=dev)); env.steps[:] = 0
if env.K > 1:
    env.set_design(torch.arange(N, device=dev), torch.full((N,), designs.index(a.design), device=dev))
if a.where == "flat":
    flat = TL.TYPES.index("flat")
    pool = (env.tile_xy.abs().amax(1) <= 3.0).nonzero()[:, 0]      # all-flat map: start near the centre, 9 m or more from the edge
else:
    pool = (env.tile_xy.abs().amax(1) <= 6.0).nonzero()[:, 0]
g = torch.Generator(device=dev); g.manual_seed(a.seed)
tpos = pool[torch.randint(0, len(pool), (N,), device=dev, generator=g)]
env.cur_tile[:] = tpos
c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device=dev) * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device=dev) * 0.4 - 0.2
q = env.q0_tab[env.didx].clone(); q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
env.sample_cmd(torch.ones(N, dtype=torch.bool, device=dev)); env.update_goal_cmd()
o = env.noisy(env.reset_obs())
steps = int(a.seconds * 100)
keys = dict(wheel_q=(7, 10, "qpos"), wheel_w=(6, 9, "qvel"), quat=(3, 7, "qpos"), gyro=(3, 6, "qvel"), base_v=(0, 3, "qvel"),
            ball_p=(10, 13, "qpos"), ball_quat=(13, 17, "qpos"), ball_v=(9, 12, "qvel"), ball_w=(12, 15, "qvel"), base_p=(0, 3, "qpos"))
L = {k: np.zeros((steps, N, b - a_), np.float32) for k, (a_, b, _) in keys.items()}
L["cmd"] = np.zeros((steps, N, 2), np.float32); L["alive"] = np.zeros((steps, N), bool); L["kind"] = np.zeros((steps, N), np.int8)
alive = torch.ones(N, dtype=torch.bool, device=dev)
for t in range(steps):
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
    alive = alive & ~fell & torch.isfinite(env.qpos).all(1)
    for k, (a_, b, src) in keys.items():
        L[k][t] = (env.qpos if src == "qpos" else env.qvel)[:, a_:b].float().cpu().numpy()
    L["cmd"][t] = env.cmd[:, :2].cpu().numpy(); L["alive"][t] = alive.cpu().numpy()
    if t % 1000 == 0:
        print(t, "alive", int(alive.sum()), flush=True)
meta = dict(run=a.run, where=a.where, vmax=a.vmax, design=a.design, n=N, seconds=a.seconds, dt=0.01, pair_mu1=os.environ.get("PAIR_MU1"),
            ball_r=float(T.Design(a.design).f ** 2 * 0.09), ball_geom_offset=env.mjm.geom("the_ball").pos.tolist(), note="qvel angular of free joints is body-local; wheel_q is the hinge angle; ball_p/ball_v are the ball BODY origin, the ball centre is ball_p + R_ball @ ball_geom_offset")
out = RUNS / "odom"; out.mkdir(exist_ok=True)
np.savez_compressed(out / f"{a.out}.npz", meta=json.dumps(meta), **L)
print("saved", out / f"{a.out}.npz", "alive at end", int(alive.sum()), "of", N)
