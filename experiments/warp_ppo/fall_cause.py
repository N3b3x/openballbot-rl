"""Why does the robot fall? Classify every fall of a checkpoint on the goal task (simulation only, MuJoCo-Warp GPU).

The model has two free bodies: the base (wheels attached) and the ball. They are held together only by gravity and the three
wheel-ball contacts. A fall in training is tilt > 70 deg, or base offset from the ball centre > 30 deg, or a non-finite state.
For each world we track, every control step:
  tilt      base z axis against world vertical (deg)
  off       direction ball centre -> base origin against vertical (deg), the termination quantity
  mis       angle between the base z axis and that direction (deg): ~0 while the wheels sit on the ball, grows when the base slides off
  dgap      distance ball centre -> base origin minus its seated value at reset (m): > 0 when the base lifts off the ball
  sat       fraction of wheels at the torque limit
  push/pull whether a push (50 ms) or a pull is acting, and the time since the last push started
"Unseated" = mis > 5 deg (the seat distance is not used: it changes by ~1 cm between rest and steady motion). "Tipping" = tilt > 25 deg. For each fall we record which came first and the state at the fall.
  TRAIN_MOD=train_dev PUSH_SCALE=1 python fall_cause.py --run g12_dr175_s0 --pull --out fallcause_ps1
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
from lqr_gpu import rotmat
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=16); ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--pull", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--lqr-only", action="store_true", help="drive with the LQR base alone (residual zeroed)")
ap.add_argument("--stand", action="store_true", help="standing test instead of the goal task: velocity task with a zero command, no goal (use --steps 2000 for 20 s)")
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda")
_a0 = json.load(open(RUN / a.run / "args.json")) if (RUN / a.run / "args.json").exists() else {}
if os.environ.get("RETAINER") == "off":
    os.environ.pop("RETAINER")
elif not os.environ.get("RETAINER") and _a0.get("retainer"):
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):      # contact model of the run (S101 fix)
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
sdim = ck.get("sdim", 0); sensor, mount = {0: ("none", "head"), 35: ("scan", "head"), 252: ("depth", "head"), 504: ("depth", "cones")}[sdim]
designs = ck.get("designs") or ["nom"]
net = T.Net(sdim, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to("cuda")
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
kinds = TL.TYPES; K = len(kinds); n = a.n; D = len(designs); N = D * K * n
env = T.GpuBallbot(N, seed=7, dr=True, design=designs[0], designs=designs if D > 1 or ck.get("ddim", 0) else None, tiles=True, tile_seed=0,
                   sensor=sensor, mount=mount, hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=True, pull=a.pull,
                   goal_p=1.0, design_cond=ck.get("ddim", 0) > 0)
dev = "cuda"
env.push_scale = float(os.environ.get("PUSH_SCALE", "1"))
_args = json.load(open(RUN / a.run / "args.json")) if (RUN / a.run / "args.json").exists() else {}
if _args.get("lqr_base", 0) > 0:
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, designs, a_max=_args.get("lqr_amax", 0.5), r=_args.get("lqr_r", 1.0)), _args["lqr_base"]
if os.environ.get("DR_SCALE"):
    env.set_dr_scale(float(os.environ["DR_SCALE"]))
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
env.goal_mode[:] = True; env.path_left[:] = 0
env.goal[:, 0] = xs + 1.1 * torch.cos(ang); env.goal[:, 1] = ys + 1.1 * torch.sin(ang)
env.update_goal_cmd()
if a.stand:
    env.goal_mode[:] = False
    env.cmd = torch.zeros(N, 3, device=dev)
o = env.noisy(env.reset_obs())


def geo():
    R = rotmat(env.qpos[:, 3:7]); up = R[:, :, 2]
    rel = env.xpos[:, env.base] - env.gxpos[:, env.ball_g]
    dist = rel.norm(dim=1).clamp_min(1e-9); u = rel / dist[:, None]
    tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1)))
    off = torch.rad2deg(torch.atan2(rel[:, :2].norm(dim=1), rel[:, 2].clamp_min(1e-6)))
    mis = torch.rad2deg(torch.acos((up * u).sum(1).clamp(-1, 1)))
    return tilt, off, mis, dist, u


env.step(torch.zeros(N, 3, device=dev))       # one settling step so the seated distance is measured with the wheels in contact
_, _, _, d0, _ = geo()
f32 = lambda: torch.zeros(N, device=dev)
alive = torch.ones(N, dtype=torch.bool, device=dev); reached = torch.zeros(N, dtype=torch.bool, device=dev)
t_unseat = torch.full((N,), -1, device=dev); t_tip = torch.full((N,), -1, device=dev); t_fall = torch.full((N,), -1, device=dev)
since_push = torch.full((N,), 1e4, device=dev)
tilt_unseat = torch.full((N,), -1.0, device=dev); vtilt_unseat = torch.zeros(N, device=dev)
rec = {k: f32() for k in ("tilt", "off", "mis", "dgap", "sat", "push_on", "pull_on", "since_push", "max_mis", "max_dgap", "sat_frac_1s", "pull_frac_1s")}
max_mis, max_dgap, max_tilt = f32(), f32(), f32()
sat_hist = torch.zeros(N, 100, device=dev); pull_hist = torch.zeros(N, 100, device=dev)
mis_hist = torch.zeros(N, 100, device=dev)
for t in range(a.steps):
    with torch.no_grad():
        act = torch.zeros(N, 3, device=dev) if a.lqr_only else pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
    tilt, off, mis, dist, _ = geo()
    dgap = dist - d0
    sat = ((env.ctrl.abs() / env.tau_w[:, None]) > 0.98).float().mean(1)
    push_on = (env.push_left > 0) | (env.push_left == 0) & (since_push == 0)
    since_push = torch.where(env.push_left == 4, torch.zeros_like(since_push), since_push + 1)
    pull_on = env.pull_left > 0
    sat_hist = torch.roll(sat_hist, 1, 1); sat_hist[:, 0] = sat
    pull_hist = torch.roll(pull_hist, 1, 1); pull_hist[:, 0] = pull_on.float()
    mis_hist = torch.roll(mis_hist, 1, 1); mis_hist[:, 0] = mis
    um = alive & (mis > 5) & (t_unseat < 0)
    t_unseat = torch.where(um, torch.full_like(t_unseat, t), t_unseat)
    tilt_unseat = torch.where(um, tilt, tilt_unseat)
    vtilt = torch.where(um, torch.rad2deg(env.qvel[:, 3:5].norm(dim=1)), vtilt_unseat) if False else vtilt_unseat
    tm = alive & (tilt > 25) & (t_tip < 0)
    t_tip = torch.where(tm, torch.full_like(t_tip, t), t_tip)
    max_tilt = torch.where(alive, torch.maximum(max_tilt, torch.nan_to_num(tilt, nan=180.0)), max_tilt)
    max_mis = torch.where(alive, torch.maximum(max_mis, torch.nan_to_num(mis, nan=180.0)), max_mis)
    max_dgap = torch.where(alive, torch.maximum(max_dgap, torch.nan_to_num(dgap, nan=1.0)), max_dgap)
    newfall = alive & fell
    if newfall.any():
        for k, v in (("tilt", tilt), ("off", off), ("mis", mis), ("dgap", dgap), ("sat", sat), ("push_on", push_on.float()), ("pull_on", pull_on.float()),
                     ("since_push", since_push), ("max_mis", max_mis), ("max_dgap", max_dgap), ("sat_frac_1s", sat_hist.mean(1)), ("pull_frac_1s", pull_hist.mean(1))):
            rec[k] = torch.where(newfall, torch.nan_to_num(v, nan=999.0), rec[k])
        t_fall = torch.where(newfall, torch.full_like(t_fall, t), t_fall)
    if t == 0:
        TR = {k: [] for k in ("tilt", "off", "mis", "dist", "bz", "kz", "pull", "push", "alive")}
    for k, v in (("tilt", tilt), ("off", off), ("mis", mis), ("dist", dist), ("bz", env.qpos[:, 2]), ("kz", env.qpos[:, 12]), ("pull", pull_on.float()), ("push", push_on.float()), ("alive", alive.float())):
        TR[k].append(v[:256].detach().cpu())
    d = (env.goal - env.qpos[:, 10:12]).norm(dim=1)
    if not a.stand:
        reached |= alive & ~fell & (d < 0.2)
    alive = alive & ~fell & ~reached
F = t_fall >= 0
nf = int(F.sum())
def frac(m):
    return float((m & F).sum()) / max(nf, 1)
unseat_first = F & (t_unseat >= 0) & ((t_tip < 0) | (t_unseat <= t_tip))
tip_first = F & (t_tip >= 0) & ((t_unseat < 0) | (t_tip < t_unseat))
neither = F & (t_unseat < 0) & (t_tip < 0)
term_off = F & (rec["off"] > 30) & (rec["tilt"] <= 70)
term_tilt = F & (rec["tilt"] > 70)
lead = (t_tip - t_unseat)[unseat_first & (t_tip >= 0)]
res = dict(run=a.run, worlds=N, falls=nf, fall_rate=nf / N, success=float(reached.float().mean()), push_scale=env.push_scale, pull=a.pull, lqr_only=a.lqr_only,
           tilt_at_unseat_deg=dict(p10=float(tilt_unseat[F & (tilt_unseat >= 0)].quantile(0.1)) if (F & (tilt_unseat >= 0)).any() else None, median=float(tilt_unseat[F & (tilt_unseat >= 0)].median()) if (F & (tilt_unseat >= 0)).any() else None, p90=float(tilt_unseat[F & (tilt_unseat >= 0)].quantile(0.9)) if (F & (tilt_unseat >= 0)).any() else None),
           survivors_max_tilt_median=float(max_tilt[~F].median()), survivors_max_tilt_p90=float(max_tilt[~F].quantile(0.9)),
           first_event=dict(unseated_first=frac(unseat_first), tipped_first=frac(tip_first), neither=frac(neither)),
           terminated_by=dict(offset_only=frac(term_off), tilt=frac(term_tilt)),
           unseat_lead_steps_median=float(lead.float().median()) if lead.numel() else None,
           at_fall=dict(mis_median=float(rec["mis"][F].median()) if nf else None, max_mis_median=float(rec["max_mis"][F].median()) if nf else None,
                        max_dgap_median_cm=float(rec["max_dgap"][F].median() * 100) if nf else None,
                        sat_frac_last_1s_mean=float(rec["sat_frac_1s"][F].mean()) if nf else None,
                        pull_active_frac=frac(rec["pull_on"] > 0), pull_frac_last_1s_mean=float(rec["pull_frac_1s"][F].mean()) if nf else None,
                        push_within_0p5s_frac=frac(rec["since_push"] <= 50), push_within_1s_frac=frac(rec["since_push"] <= 100)),
           survivors_max_mis_median=float(max_mis[~F].median()), survivors_max_dgap_median_cm=float(max_dgap[~F].median() * 100),
           by_kind={kinds[k]: dict(fall=float(F[kid == k].float().mean()), unseated_first=float((unseat_first & (kid == k)).sum()) / max(int((F & (kid == k)).sum()), 1)) for k in range(K)},
           by_design={designs[i]: dict(fall=float(F[did == i].float().mean()), unseated_first=float((unseat_first & (did == i)).sum()) / max(int((F & (did == i)).sum()), 1),
                                       tipped_first=float((tip_first & (did == i)).sum()) / max(int((F & (did == i)).sum()), 1),
                                       t_fall_median_s=float(t_fall[F & (did == i)].float().median() / 100) if (F & (did == i)).any() else None,
                                       sat_frac_last_1s_mean=float(rec["sat_frac_1s"][F & (did == i)].mean()) if (F & (did == i)).any() else None,
                                       survivors_max_tilt_median=float(max_tilt[~F & (did == i)].median()) if (~F & (did == i)).any() else None) for i in range(D)},
           stand=a.stand)
(RUN / a.run / f"{a.out}.json").write_text(json.dumps(res, indent=1))
torch.save(dict({k: torch.stack(v, 1) for k, v in TR.items()}, d0=d0[:256].cpu(), t_fall=t_fall[:256].cpu(), kid=kid[:256].cpu(), did=did[:256].cpu()), RUN / a.run / f"{a.out}_trace.pt")
print(json.dumps({k: v for k, v in res.items() if k not in ("by_kind", "by_design")}, indent=1))
print({k: (round(v["fall"], 2), round(v["unseated_first"], 2)) for k, v in res["by_kind"].items()})
print({k: (round(v["fall"], 2), round(v["unseated_first"], 2)) for k, v in res["by_design"].items()})
