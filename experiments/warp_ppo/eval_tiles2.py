"""Per-terrain, per-design evaluation for train_dev checkpoints (and old ones). Tasks: vel (fixed commanded velocity, random heading) or goal (reach a goal 1.1 m away).
  TRAIN_MOD=train_dev python eval_tiles2.py --run NAME --task vel --speed 0.6 [--dr --surfaces] [--eval-designs nom,s0.5]
Layout: worlds = designs x terrain kinds x n. Per cell: fall, crossed (vel) or reached (goal), tracking error of the velocity command (vel).
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--task", choices=["vel", "goal"], default="vel")
ap.add_argument("--speed", type=float, default=0.6); ap.add_argument("--n", type=int, default=32)
ap.add_argument("--dr", action="store_true"); ap.add_argument("--surfaces", action="store_true"); ap.add_argument("--pull", action="store_true")
ap.add_argument("--eval-designs", default=None); ap.add_argument("--steps", type=int, default=800); ap.add_argument("--tile-seed", type=int, default=0)
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda")
_a0 = json.load(open(RUN / a.run / "args.json")) if (RUN / a.run / "args.json").exists() else {}
_ret_override = os.environ.get("RETAINER")      # set explicitly = evaluate on a robot with (or, "off", without) ball arrestors regardless of training
if _ret_override == "off":
    os.environ.pop("RETAINER")
elif not _ret_override and _a0.get("retainer"):
    os.environ["RETAINER"] = _a0["retainer"]
_mu_override = os.environ.get("PAIR_MU1")      # set explicitly = evaluate with a different wheel-ball free-rolling friction than the run was trained with (S98)
if not _mu_override and _a0.get("pair_mu1") is not None:
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
sdim = ck.get("sdim", 0); sensor, mount = {0: ("none", "head"), 35: ("scan", "head"), 252: ("depth", "head"), 504: ("depth", "cones")}[sdim]
tr_designs = ck.get("designs") or ["nom"]
ev_designs = a.eval_designs.split(",") if a.eval_designs else tr_designs
net = T.Net(sdim, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to("cuda")
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
if os.environ.get("LQR_ONLY"):      # diagnostic: zero the learned residual, drive with the LQR base alone
    pol = lambda o: torch.zeros(o.shape[0], 3, device="cuda")
kinds = TL.TYPES; K = len(kinds); n = a.n; D = len(ev_designs); N = D * K * n
env = T.GpuBallbot(N, seed=7, dr=a.dr, design=ev_designs[0], designs=ev_designs if D > 1 or ck.get("ddim", 0) or ck.get("pdim", 0) > 46 else None, tiles=True, tile_seed=a.tile_seed,
                   sensor=sensor, mount=mount, hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=a.surfaces, pull=a.pull,
                   goal_p=1.0 if a.task == "goal" else ck.get("goal_p", 0.0), design_cond=ck.get("ddim", 0) > 0)
dev = "cuda"
import os
env.push_scale = float(os.environ.get("PUSH_SCALE", "1"))
env.goal_centre = bool(_a0.get("goal_centre")); env.db_centre = bool(_a0.get("db_centre"))      # S118: goal logic on the true ball centre if the run was trained that way
_args = json.load(open(RUN / a.run / "args.json")) if (RUN / a.run / "args.json").exists() else {}
if _args.get("lqr_base", 0) > 0:      # the policy is a residual on the LQR base controller: rebuild it exactly as in training
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, ev_designs, a_max=_args.get("lqr_amax", 0.5), r=_args.get("lqr_r", 1.0)), _args["lqr_base"]
    if _args.get("rs_adapt"): env.rs_adapt = [float(x) for x in _args["rs_adapt"].split(",")]      # S111 adaptive authority
if _args.get("motion_ref") and not os.environ.get("NO_MREF"):      # S110: a run trained with the motion-reference layer is evaluated with it (its commands pass through motion_ref.py, no extra curving)
    env.enable_motion_ref(curve_p=0.0)
if os.environ.get("EVAL_NOEXIT"):       # measure steady-state tracking at speed: do not end the episode when the ball leaves its tile
    TL.EXIT_R = 99.0
env.dr_m, env.dr_f, env.dr_t = (float(os.environ.get(k, "1")) for k in ("DR_M", "DR_F", "DR_T"))
if os.environ.get("DR_SCALE"):      # master randomisation scale for robustness evaluation
    env.set_dr_scale(float(os.environ["DR_SCALE"]))
for _k, _a in (("DR_M", "dr_m"), ("DR_F", "dr_f"), ("DR_T", "dr_t"), ("DR_D", "dr_d"), ("DR_L", "dr_l"), ("DR_N", "dr_n")):      # single-factor overrides apply after the master scale
    if os.environ.get(_k):
        setattr(env, _a, float(os.environ[_k]))
for _k in [x for x in os.environ.get("DR_OFF", "").split(",") if x]:   # switch single randomisation terms off: mass,friction,damping,torque,latency_p,obs_noise
    env.drc[_k] = (0.1, 0.1) if _k == "damping" else 0.0
env.resample, env.max_steps = 10 ** 9, 10 ** 9
env.reset(torch.ones(N, dtype=torch.bool, device=dev)); env.steps[:] = 0
w = torch.arange(N, device=dev)
did, kid = w // (K * n), (w // n) % K
if env.K > 1:
    env.set_design(w, did)
tpos = torch.zeros(N, dtype=torch.long, device=dev)
for k in range(K):
    ids = (env.tile_kind == k).nonzero()[:, 0]
    tpos[kid == k] = ids[torch.arange(N, device=dev)[kid == k] % len(ids)]
if os.environ.get("TILES_FLAT"):      # flat benchmark: interior tiles only (at least 7.5 m to the height-field border), so a robot driving 0.9 m/s for 8 s or 0.3 m/s for 20 s never reaches the edge of the map (edge drop-offs were counted as falls before this fix, see log S92)
    inner = ((env.tile_xy.abs().amax(1)) <= 4.6).nonzero()[:, 0]
    tpos = inner[torch.arange(N, device=dev) % len(inner)]
env.cur_tile[:] = tpos
c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device=dev) * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device=dev) * 0.4 - 0.2
q = env.q0_tab[env.didx].clone()
q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
ang = torch.rand(N, device=dev) * 2 * np.pi
if a.task == "vel":
    env.goal_mode[:] = False
    env.cmd = torch.stack([a.speed * env.f_w * torch.cos(ang), a.speed * env.f_w * torch.sin(ang), torch.zeros(N, device=dev)], 1)
else:
    env.goal_mode[:] = True; env.path_left[:] = 0
    env.goal[:, 0] = xs + 1.1 * torch.cos(ang); env.goal[:, 1] = ys + 1.1 * torch.sin(ang)
    env.update_goal_cmd()
if getattr(env, "mref", None) is not None:      # motion layer: desired command = the scenario's, shaped from rest with the robot facing it
    env.cmd_des = env.cmd[:, :2].clone(); env.cmd_w[:] = 0
    env.mref.reset(torch.ones(N, dtype=torch.bool, device=dev), heading=ang)
o = env.noisy(env.reset_obs())
fall_xy = torch.zeros(N, 2, device=dev)
fell_any = torch.zeros(N, dtype=torch.bool, device=dev); done_ok = torch.zeros(N, dtype=torch.bool, device=dev); frozen = torch.zeros(N, dtype=torch.bool, device=dev)
nonfin_any = torch.zeros(N, dtype=torch.bool, device=dev)      # S98: a non-finite simulator state counts as a fall in the termination; report it separately
err_sum = torch.zeros(N, device=dev); err_n = torch.zeros(N, device=dev); lerr_sum = torch.zeros(N, device=dev); lrat_sum = torch.zeros(N, device=dev); ln = torch.zeros(N, device=dev); t_reach = torch.full((N,), float(a.steps), device=dev)
for t in range(a.steps):
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
    alive = ~frozen
    fall_xy = torch.where((fell & alive)[:, None], env.qpos[:, 10:12], fall_xy)
    fell_any |= fell & alive
    nonfin_any |= fell & alive & ~(torch.isfinite(env.qpos).all(1) & torch.isfinite(env.qvel).all(1))
    if a.task == "vel":
        done_ok |= trunc & alive & ~fell_any
        if t >= 300:       # steady state: after the first 3 s, error and achieved speed along the command
            _c = env.cmd[:, :2]; _cn = _c.norm(dim=1).clamp_min(1e-6); _ok = (alive & ~fell).float()
            _e = torch.where(_ok > 0, (env._vel[:, :2] - _c).norm(dim=1), torch.zeros_like(_cn)); _r = torch.where(_ok > 0, (env._vel[:, :2] * _c).sum(1) / (_cn * _cn), torch.zeros_like(_cn))      # where, not multiply: a fallen world can carry non-finite velocities
            lerr_sum += torch.nan_to_num(_e, nan=0.0, posinf=0.0); lrat_sum += torch.nan_to_num(_r, nan=0.0, posinf=0.0); ln += _ok
        err_sum += torch.where(alive & ~fell, (env._vel[:, :2] - env.cmd[:, :2]).norm(dim=1), torch.zeros(N, device=dev)); err_n += (alive & ~fell).float()
    else:
        d = (env.goal - env.qpos[:, 10:12]).norm(dim=1)
        hit = alive & ~fell_any & (d < 0.2)
        t_reach = torch.where(hit & ~done_ok, torch.full_like(t_reach, t), t_reach); done_ok |= hit
    frozen |= fell_any | done_ok
res = {}
def cell(mask):
    r = dict(fall=float(fell_any[mask].float().mean()), success=float(done_ok[mask].float().mean()), nonfinite=float(nonfin_any[mask].float().mean()))
    fe = fell_any & mask
    r["fall_at_map_edge"] = float((fall_xy[fe].abs().amax(1) > TL.HALF - 0.6).float().mean()) if fe.any() else 0.0      # fell within 0.6 m of the height-field border: an artefact of the finite map, not a controller fall
    if a.task == "vel":
        r["track_err"] = float(err_sum[mask].sum() / err_n[mask].sum().clamp_min(1)); r["late_err"] = float(lerr_sum[mask].sum() / ln[mask].sum().clamp_min(1)); r["late_speed_ratio"] = float(lrat_sum[mask].sum() / ln[mask].sum().clamp_min(1))
    else:
        r["median_time_s"] = float((t_reach[mask & done_ok].median() * 0.01)) if (mask & done_ok).any() else None
    return r
res["by_kind"] = {kinds[k]: cell(kid == k) for k in range(K)}
res["by_design"] = {ev_designs[i]: cell(did == i) for i in range(D)}
res["overall"] = cell(torch.ones(N, dtype=torch.bool, device=dev))
_ovr = {k: os.environ[k] for k in ("DR_M", "DR_F", "DR_T", "DR_D", "DR_L", "DR_N") if os.environ.get(k)}
_env = dict(overrides=_ovr, push_scale=float(os.environ.get("PUSH_SCALE", "1")), dr_scale=(float(os.environ["DR_SCALE"]) if os.environ.get("DR_SCALE") else None),
            tiles_flat=bool(os.environ.get("TILES_FLAT")), no_exit=bool(os.environ.get("EVAL_NOEXIT")), dr_off=os.environ.get("DR_OFF", ""),
            dr_mft=[env.dr_m, env.dr_f, env.dr_t], lqr_base=_args.get("lqr_base", 0.0), eval_designs=ev_designs)
# the file name carries every setting that changes the result, so a later evaluation never overwrites an earlier one under a different setting
tag = f"eval2_{a.task}_{a.speed if a.task == 'vel' else 'goal'}{'_dr' if a.dr else ''}{'_surf' if a.surfaces else ''}{'_pull' if a.pull else ''}"
if _env["push_scale"] != 1.0: tag += f"_ps{_env['push_scale']:g}"
if _env["dr_scale"] is not None: tag += f"_drs{_env['dr_scale']:g}"
if a.tile_seed != 0: tag += f"_ts{a.tile_seed}"
if a.eval_designs and not set(ev_designs) <= set(tr_designs): tag += "_des-" + "-".join(ev_designs)      # held-out designs: never overwrite the standard files
if _ret_override: tag += f"_ret{_ret_override.replace(',', '-')}"
_env["retainer"] = os.environ.get("RETAINER")
if _env["tiles_flat"]: tag += "_flat"
if _env["no_exit"]: tag += "_noexit"
if os.environ.get("LQR_ONLY"): tag += "_lqronly"
if os.environ.get("PAIR_MU1"): _env["pair_mu1"] = float(os.environ["PAIR_MU1"])      # wheel-ball free-rolling friction in effect (S98)
if _mu_override: tag += f"_mu{_mu_override}"      # only an explicit override gets its own files; the run's own value is the standard file
if _ovr: tag += "_" + "".join(f"{k[3:].lower()}{v}" for k, v in _ovr.items())
if _env["dr_off"]: tag += "_off-" + _env["dr_off"].replace(",", "-")
tag += ".json"
json.dump(dict(run=a.run, args=vars(a), env_settings=_env, **res), open(RUN / a.run / tag, "w"), indent=1)
print(json.dumps(res["overall"]), "|", {k: (round(v["fall"], 2), round(v["success"], 2)) for k, v in res["by_kind"].items()})
if D > 1: print({k: (round(v["fall"], 2), round(v["success"], 2)) for k, v in res["by_design"].items()})
