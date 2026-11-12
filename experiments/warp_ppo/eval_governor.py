"""Evaluate a checkpoint with the terrain-aware speed layer (governor.py) on the tiled terrain (simulation only, MuJoCo-Warp GPU).
Same world layout as eval_tiles2.py (designs x terrain kinds x n worlds, DR, surfaces, optional pushes and pulls) but longer episodes
(default 15 s) because a governed robot crosses obstacles slowly. Hazard source:
  none        no governor (baseline, same episode length)
  oracle      privileged height map ahead (upper bound)
  enc:NAME    learned estimator percep/NAME_gov.pt applied to a simulated sensor (its config is stored in the file)
Metrics per terrain kind and overall: fall, success (goal: reach within 0.2 m; vel: leave the tile without falling), median time,
and for vel the mean progress along the command (m).
  TRAIN_MOD=train_dev PUSH_SCALE=0 python eval_governor.py --run g18_ret_rs025_s1 --task vel --speed 0.3 --source oracle --out gov_oracle
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
from governor import Governor
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--task", choices=["vel", "goal"], default="vel"); ap.add_argument("--speed", type=float, default=0.3)
ap.add_argument("--n", type=int, default=16); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--pull", action="store_true")
ap.add_argument("--source", default="none"); ap.add_argument("--mode", choices=["slow", "refuse"], default="slow"); ap.add_argument("--rise-max", type=float, default=0.024); ap.add_argument("--drop-max", type=float, default=0.08); ap.add_argument("--out", required=True); ap.add_argument("--tile-seed", type=int, default=0); ap.add_argument("--eval-designs", default=None)
ap.add_argument("--v-slow", type=float, default=0.1); ap.add_argument("--gain", type=float, default=0.6); ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--hold", type=float, default=1.0); ap.add_argument("--d-near", type=float, default=0.25)
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda")
_a0 = json.load(open(RUN / a.run / "args.json")) if (RUN / a.run / "args.json").exists() else {}
if _a0.get("retainer") and not os.environ.get("RETAINER"):
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):      # S101 fix: evaluate on the contact model the run was trained on (was missing; runs trained at 0.01 were evaluated at 0.001)
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
assert ck.get("sdim", 0) == 0, "governor evaluation expects a blind balance policy"
designs = a.eval_designs.split(",") if a.eval_designs else (ck.get("designs") or ["nom"])
net = T.Net(0, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to("cuda")
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")

enc = None; sensor, mount, res = "none", "head", (18, 14)
if a.source.startswith("enc:"):
    enc = torch.load(RUN / "percep" / f"{a.source[4:]}_gov.pt", map_location="cuda")
    sensor, mount, res = "depth", enc["mount"], tuple(enc["res"])
    T.SENSOR_FOV = tuple(enc.get("fov", T.SENSOR_FOV)); T.SENSOR_PITCH = enc.get("pitch", T.SENSOR_PITCH); T.SENSOR_UP = enc.get("up", T.SENSOR_UP)      # estimators trained on collect_percep.py data use the default head geometry
kinds = TL.TYPES; K = len(kinds); n = a.n; D = len(designs); N = D * K * n
env = T.GpuBallbot(N, seed=7, dr=True, design=designs[0], designs=designs if D > 1 or ck.get("ddim", 0) else None, tiles=True, tile_seed=a.tile_seed,
                   sensor=sensor, mount=mount, depth_res=res, hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=True, pull=a.pull,
                   goal_p=1.0 if a.task == "goal" else 0.0, design_cond=ck.get("ddim", 0) > 0)
SD = env.sdim
pol = lambda o: net(((torch.cat([o[:, :T.OBS_DIM], o[:, T.OBS_DIM + SD:]], 1) - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
dev = "cuda"
env.push_scale = float(os.environ.get("PUSH_SCALE", "1"))
if _a0.get("lqr_base", 0) > 0:
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, designs, a_max=_a0.get("lqr_amax", 0.5), r=_a0.get("lqr_r", 1.0)), _a0["lqr_base"]
    if _a0.get("rs_adapt"): env.rs_adapt = [float(x) for x in _a0["rs_adapt"].split(",")]      # S111 adaptive authority
env.goal_centre = bool(_a0.get("goal_centre")); env.db_centre = bool(_a0.get("db_centre"))      # S118 / S122, as the other evaluation scripts
if _a0.get("motion_ref") and not os.environ.get("NO_MREF"):      # S115: the motion layer the run was trained with (before this, S95 to S104 evaluations of motion-layer runs had no layer)
    env.enable_motion_ref(curve_p=0.0)
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
ang = torch.rand(N, device=dev) * 2 * np.pi
start = torch.stack([xs, ys], 1); udir = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
if a.task == "vel":
    env.goal_mode[:] = False
    env.cmd = torch.stack([a.speed * env.f_w * torch.cos(ang), a.speed * env.f_w * torch.sin(ang), torch.zeros(N, device=dev)], 1)
    user_cmd = env.cmd.clone()
    if getattr(env, "mref", None) is not None:      # desired command = the scenario's, shaped from rest with the robot facing it (as eval_tiles2.py)
        env.cmd_des = env.cmd[:, :2].clone(); env.cmd_w[:] = 0
        env.mref.reset(torch.ones(N, dtype=torch.bool, device=dev), heading=ang)
else:
    env.goal_mode[:] = True; env.path_left[:] = 0
    env.goal[:, 0] = xs + 1.1 * torch.cos(ang); env.goal[:, 1] = ys + 1.1 * torch.sin(ang)
    env.update_goal_cmd()

gov = Governor(env, v_slow=a.v_slow, gain=a.gain, d_near=a.d_near, thr=a.thr, hold=a.hold) if a.source != "none" else None
if enc is not None:
    import torch.nn as nn
    est = nn.Sequential(nn.Linear(enc["nin"], enc["width"]), nn.GELU(), nn.Linear(enc["width"], enc["width"]), nn.GELU(), nn.Linear(enc["width"], 35)).to(dev)
    est.load_state_dict(enc["net"]); est.eval()
    dm, ds, pm, ps = enc["dmean"].to(dev), enc["dstd"].to(dev), enc["pmean"].to(dev), enc["pstd"].to(dev)
_orig_update = env.update_goal_cmd
last_obs = {"o": None}


def governed_update():
    """Called inside env.step after the command update and before the observation: the vel task restores the user command first, then the governor rescales it."""
    _orig_update()
    if a.task == "vel":
        env.cmd = user_cmd.clone()
    if gov is None:
        return
    if a.source == "oracle":
        scan = env.scan_feat()
    else:
        o = last_obs["o"]
        if o is None:
            return
        dpt = o[:, T.OBS_DIM:T.OBS_DIM + SD]; pr = o[:, :T.OBS_DIM]
        x = torch.cat([((dpt - dm) / ds).clamp(-10, 10), ((pr - pm) / ps).clamp(-6, 6)], 1)
        with torch.no_grad():
            scan = est(x)
    if a.mode == "refuse":
        gov.refuse(scan, rise_max=a.rise_max, drop_max=a.drop_max)
    else:
        gov.apply(scan)


env.update_goal_cmd = governed_update
o = env.noisy(env.reset_obs()); last_obs["o"] = o
fell_any = torch.zeros(N, dtype=torch.bool, device=dev); done_ok = torch.zeros(N, dtype=torch.bool, device=dev); frozen = torch.zeros(N, dtype=torch.bool, device=dev)
t_reach = torch.full((N,), float(a.steps), device=dev); prog = torch.zeros(N, device=dev)
# S101 behaviour comparison: per-world motion statistics while alive (simulation only)
B = {k: torch.zeros(N, device=dev) for k in ("n", "v_along", "v_cmd", "slow_steps", "tilt_max", "tilt_sum", "act_sq", "tilt_at_fall", "v_at_fall", "t_fall")}
for t in range(a.steps):
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o); last_obs["o"] = o
    alive = ~frozen
    _live = (alive & ~fell).float()
    _c = env.cmd[:, :2]; _cn = _c.norm(dim=1)
    _va = torch.where(_cn > 1e-6, (env._vel[:, :2] * _c).sum(1) / _cn.clamp_min(1e-6), torch.zeros_like(_cn))
    _tl = torch.nan_to_num(info["tilt"], nan=90.0)
    B["n"] += _live; B["v_along"] += torch.nan_to_num(_va) * _live; B["v_cmd"] += _cn * _live
    _ref = user_cmd[:, :2].norm(dim=1) if a.task == "vel" else _cn      # goal task: the goal command itself is the reference
    B["slow_steps"] += ((torch.nan_to_num(_va) < 0.2 * _ref).float()) * _live
    B["tilt_max"] = torch.where(_live > 0, torch.maximum(B["tilt_max"], _tl), B["tilt_max"]); B["tilt_sum"] += _tl * _live
    B["act_sq"] += (act ** 2).sum(1) * _live
    _nf = fell & alive
    B["tilt_at_fall"] = torch.where(_nf, _tl, B["tilt_at_fall"]); B["v_at_fall"] = torch.where(_nf, torch.nan_to_num(_va), B["v_at_fall"]); B["t_fall"] = torch.where(_nf, torch.full_like(B["t_fall"], t * 0.01), B["t_fall"])
    fell_any |= fell & alive
    if a.task == "vel":
        p = ((env.qpos[:, 10:12] - start) * udir).sum(1)
        prog = torch.where(alive & ~fell, torch.maximum(prog, p), prog)
        hit = trunc & alive & ~fell_any
    else:
        hit = alive & ~fell_any & ((env.goal - env.qpos[:, 10:12]).norm(dim=1) < 0.2)
    t_reach = torch.where(hit & ~done_ok, torch.full_like(t_reach, t), t_reach); done_ok |= hit
    frozen |= fell_any | done_ok


REF = gov.stopped if (gov is not None and hasattr(gov, "stopped")) else torch.zeros(N, dtype=torch.bool, device=dev)


def cell(m):
    r = dict(fall=float(fell_any[m].float().mean()), success=float(done_ok[m].float().mean()), refused=float(REF[m].float().mean()),
             median_time_s=float(t_reach[m & done_ok].median() * 0.01) if (m & done_ok).any() else None)
    if a.task == "vel":
        r["progress_m"] = float(prog[m].mean())
    n = B["n"][m].clamp_min(1)
    r["behav"] = dict(mean_v_along=float((B["v_along"][m] / n).mean()), mean_v_cmd=float((B["v_cmd"][m] / n).mean()),
                      slow_frac=float((B["slow_steps"][m] / n).mean()), tilt_max_median=float(B["tilt_max"][m].median()),
                      tilt_mean=float((B["tilt_sum"][m] / n).mean()), residual_rms=float(((B["act_sq"][m] / n) / 3).sqrt().mean()),
                      tilt_at_fall_median=float(B["tilt_at_fall"][m & fell_any].median()) if (m & fell_any).any() else None,
                      v_at_fall_median=float(B["v_at_fall"][m & fell_any].median()) if (m & fell_any).any() else None,
                      t_fall_median_s=float(B["t_fall"][m & fell_any].median()) if (m & fell_any).any() else None)
    return r


res = dict(run=a.run, motion_ref=getattr(env, "mref", None) is not None, task=a.task, speed=a.speed, source=a.source, steps=a.steps, push_scale=env.push_scale, pull=a.pull, tile_seed=a.tile_seed,
           mode=a.mode, governor=dict(rise_max=a.rise_max, drop_max=a.drop_max, v_slow=a.v_slow, gain=a.gain, thr=a.thr, hold=a.hold, d_near=a.d_near) if gov else None,
           mean_speed_scale=float(np.mean(gov.scale_log)) if gov else 1.0, retainer=os.environ.get("RETAINER"), pair_mu1=(float(os.environ["PAIR_MU1"]) if os.environ.get("PAIR_MU1") else None),
           overall=cell(torch.ones(N, dtype=torch.bool, device=dev)), by_kind={kinds[k]: cell(kid == k) for k in range(K)})
sev = env.tile_sev[tpos]
res["by_severity"] = {}
for k in range(K):      # falls by feature size: lower / upper half of the severity range drawn for this kind
    m = kid == k
    if m.sum() < 4 or kinds[k] in ("flat",):
        continue
    med = float(sev[m].median())
    res["by_severity"][kinds[k]] = dict(split=med, small=cell(m & (sev <= med)), large=cell(m & (sev > med)),
                                        sev_min=float(sev[m].min()), sev_max=float(sev[m].max()))
res["per_world"] = dict(kind=[kinds[int(k)] for k in kid.tolist()], severity=[round(float(x), 4) for x in sev.tolist()], fell=fell_any.int().tolist(), success=done_ok.int().tolist())      # for finer binning offline (crossing against feature height)
(RUN / a.run / f"{a.out}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["overall"]), "| scale", round(res["mean_speed_scale"], 2), "|", {k: (round(v["fall"], 2), round(v["success"], 2)) for k, v in res["by_kind"].items()})
