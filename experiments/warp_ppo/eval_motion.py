"""Tracking quality and goal behaviour of a balance checkpoint, with raw commands or through the motion-reference layer (motion_ref.py).
Simulation only, MuJoCo-Warp GPU. Flat interior tiles (no terrain features, no map edge), every design of the checkpoint, DR on, noise on,
arrestors and contact model from the run's args.json, no pushes unless PUSH_SCALE is set.
Scenarios:
  goal    a goal 1.2 m away in a random direction; 15 s, so the robot arrives and then must hold. Raw = the training goal rule
          (v = 1.5 d toward the goal, zero inside 5 cm); ref = the same desired velocity through MotionRef with arrival and hold.
  curve   desired velocity of 0.3 m/s whose heading turns steadily at 0.6 rad/s (a 0.5 m radius circle), 15 s.
  zigzag  desired heading switches by 60 degrees left and right every 2 s at 0.3 m/s (planner-like corrections), 15 s.
Metrics: falls; actual heading-rate RMS of the ball velocity minus the commanded one; speed along the desired direction / desired;
sideways speed; for goal also time to arrive (first within 0.1 m), overshoot (largest distance after first arrival), hold error RMS and
hold speed RMS over the last 8 s, and command reversals (command direction flipping by more than 90 degrees within 0.5 s).
  TRAIN_MOD=train_dev python eval_motion.py --run g26_ret_rs025_mu01_s1 --scenario goal --mode ref --out motion_goal_ref
"""
import argparse, importlib, json, math, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
from motion_ref import MotionRef
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--scenario", choices=["goal", "curve", "zigzag"], default="goal")
ap.add_argument("--mode", choices=["raw", "ref"], default="raw"); ap.add_argument("--n", type=int, default=48); ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--speed", type=float, default=0.3); ap.add_argument("--out", required=True)
ap.add_argument("--a-lat", type=float, default=0.35); ap.add_argument("--a-brk", type=float, default=0.5); ap.add_argument("--kp", type=float, default=0.8)
a = ap.parse_args()
ck = torch.load(RUN / a.run / "model.pt", map_location="cuda")
_a0 = json.load(open(RUN / a.run / "args.json")) if (RUN / a.run / "args.json").exists() else {}
if _a0.get("retainer") and not os.environ.get("RETAINER"):
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
designs = ck.get("designs") or ["nom"]
net = T.Net(0, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to("cuda")
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to("cuda"), ck["var"].to("cuda")
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
if os.environ.get("LQR_ONLY"):      # diagnostic: residual zeroed, the LQR alone tracks the command
    pol = lambda o: torch.zeros(o.shape[0], 3, device="cuda")
D = len(designs); n = a.n; N = D * n; dev = "cuda"
env = T.GpuBallbot(N, seed=11, dr=True, design=designs[0], designs=designs if D > 1 or ck.get("ddim", 0) else None, tiles=True,
                   hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=True, pull=False, goal_p=1.0, design_cond=ck.get("ddim", 0) > 0)
env.push_scale = float(os.environ.get("PUSH_SCALE", "0"))
env.goal_centre = bool(_a0.get("goal_centre")); env.db_centre = bool(_a0.get("db_centre"))      # S118: goal logic on the true ball centre if the run was trained that way
if _a0.get("lqr_base", 0) > 0:
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, designs, a_max=_a0.get("lqr_amax", 0.5), r=_a0.get("lqr_r", 1.0)), _a0["lqr_base"]
    _rsa = os.environ.get("RS_ADAPT") or _a0.get("rs_adapt")      # RS_ADAPT: deployment-time override of the trained setting (S115)
    if _rsa: env.rs_adapt = [float(x) for x in _rsa.split(",")]      # S111 adaptive authority
TL.EXIT_R = 99.0
env.resample, env.max_steps = 10 ** 9, 10 ** 9
env.reset(torch.ones(N, dtype=torch.bool, device=dev)); env.steps[:] = 0
w = torch.arange(N, device=dev); did = w // n
if env.K > 1:
    env.set_design(w, did)
flat = TL.TYPES.index("flat")
inner = ((env.tile_kind == flat) & (env.tile_xy.abs().amax(1) <= 4.6)).nonzero()[:, 0]
tpos = inner[torch.arange(N, device=dev) % len(inner)]
env.cur_tile[:] = tpos
c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device=dev) * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device=dev) * 0.4 - 0.2
q = env.q0_tab[env.didx].clone(); q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
ang = torch.rand(N, device=dev) * 2 * math.pi
fw = env.f_w if torch.is_tensor(env.f_w) else torch.full((N,), float(env.f_w), device=dev)
mref = MotionRef(N, dev, a_lat=a.a_lat, a_brk=a.a_brk, kp=a.kp); mref.reset(torch.ones(N, dtype=torch.bool, device=dev), heading=ang)
if a.scenario == "goal":
    env.goal_mode[:] = True; env.path_left[:] = 0
    env.goal[:, 0] = xs + 1.2 * torch.cos(ang); env.goal[:, 1] = ys + 1.2 * torch.sin(ang)
else:
    env.goal_mode[:] = False
tstep = {"t": 0}
_orig = env.update_goal_cmd


def desired():
    t = tstep["t"] * 0.01
    if a.scenario == "curve":
        h = ang + 0.6 * t
    else:
        h = ang + math.radians(60) * (1 if int(t // 2) % 2 == 0 else -1) * (t >= 2)
    return torch.stack([a.speed * fw * torch.cos(h), a.speed * fw * torch.sin(h)], 1)


def update():
    """Called inside env.step: the training goal rule first (goal scenario), then the chosen command shaping."""
    if a.scenario == "goal":
        _orig()
        vd = env.cmd[:, :2].clone()
        if a.mode == "ref":
            env.cmd = torch.cat([mref.step(vd, fw, env.ball_xy(), env.goal), torch.zeros(N, 1, device=dev)], 1)
    else:
        vd = desired()
        cmd = mref.step(vd, fw) if a.mode == "ref" else vd
        env.cmd = torch.cat([cmd, torch.zeros(N, 1, device=dev)], 1)
    DES["v"] = vd


DES = {"v": torch.zeros(N, 2, device=dev)}


def centre_xy():
    """True ball centre for the metrics, independent of the run's goal_centre setting."""
    _f = env.goal_centre; env.goal_centre = True
    c = env.ball_xy(); env.goal_centre = _f
    return c

env.update_goal_cmd = update
update()
o = env.noisy(env.reset_obs())
alive = torch.ones(N, dtype=torch.bool, device=dev)
S = {k: torch.zeros(N, device=dev) for k in ("n", "hr_act", "hr_cmd", "along", "side", "rev")}
first = torch.full((N,), -1.0, device=dev); over = torch.zeros(N, device=dev)
hold_e = torch.zeros(N, device=dev); hold_v = torch.zeros(N, device=dev); hold_n = torch.zeros(N, device=dev)
prev_vh = None; prev_ch = None; cmd_hist = []
for t in range(a.steps):
    tstep["t"] = t
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o)
    v = torch.nan_to_num(env._vel[:, :2]); cm = env.cmd[:, :2]
    live = (alive & ~fell).float()
    vh = torch.atan2(v[:, 1], v[:, 0]); ch = torch.atan2(cm[:, 1], cm[:, 0])
    moving = (v.norm(dim=1) > 0.05) & (cm.norm(dim=1) > 0.05)
    if prev_vh is not None:
        dv = torch.remainder(vh - prev_vh + math.pi, 2 * math.pi) - math.pi; dc = torch.remainder(ch - prev_ch + math.pi, 2 * math.pi) - math.pi
        m = live * moving.float()
        S["hr_act"] += (dv / 0.01) ** 2 * m; S["hr_cmd"] += (dc / 0.01) ** 2 * m; S["n"] += m
    prev_vh, prev_ch = vh, ch
    vd = DES["v"]; vdn = vd.norm(dim=1).clamp_min(1e-6); u = vd / vdn[:, None]
    S["along"] += ((v * u).sum(1) / vdn) * live * (vdn > 0.05).float(); S["side"] += (v[:, 0] * -u[:, 1] + v[:, 1] * u[:, 0]).abs() * live * (vdn > 0.05).float()
    cmd_hist.append(cm.clone())
    if t >= 50:
        old = cmd_hist[-51]
        flip = ((old * cm).sum(1) < 0) & (old.norm(dim=1) > 0.03) & (cm.norm(dim=1) > 0.03)
        S["rev"] += flip.float() * live
        cmd_hist[-51] = None
    if a.scenario == "goal":
        d = (env.goal - centre_xy()).norm(dim=1)      # S118: measured at the true ball centre for every run (was the ball body origin)
        newly = alive & (first < 0) & (d < 0.1)
        first = torch.where(newly, torch.full_like(first, t * 0.01), first)
        over = torch.where(alive & (first >= 0), torch.maximum(over, d), over)
        if t >= a.steps - 800:
            mh = (alive & ~fell & (first >= 0)).float()
            hold_e += d ** 2 * mh; hold_v += v.norm(dim=1) ** 2 * mh; hold_n += mh
    alive = alive & ~fell
nm = S["n"].clamp_min(1); nt = (a.steps * torch.ones(N, device=dev))
res = dict(run=a.run, scenario=a.scenario, goal_metric_point="ball_centre", goal_centre_logic=bool(env.goal_centre), rs_adapt=getattr(env, "rs_adapt", None), mode=a.mode, lqr_only=bool(os.environ.get("LQR_ONLY")), steps=a.steps, speed=a.speed, pair_mu1=os.environ.get("PAIR_MU1"), push_scale=env.push_scale,
           worlds=N, fall=float((~alive).float().mean()),
           heading_rate_rms_actual=float((S["hr_act"].sum() / S["n"].sum().clamp_min(1)).sqrt()),
           heading_rate_rms_command=float((S["hr_cmd"].sum() / S["n"].sum().clamp_min(1)).sqrt()),
           along_ratio=float((S["along"] / nt).mean() * a.steps / max(1, a.steps)), side_speed_mean=float((S["side"] / nt).mean()),
           cmd_reversals_per_world=float(S["rev"].mean() / 50))
if a.scenario == "goal":
    ok = alive & (first >= 0)
    res.update(arrived=float((first >= 0).float().mean()), time_to_arrive_median_s=float(first[first >= 0].median()) if (first >= 0).any() else None,
               overshoot_median_m=float(over[ok].median()) if ok.any() else None, overshoot_p90_m=float(over[ok].quantile(0.9)) if ok.any() else None,
               hold_err_rms_m=float((hold_e[ok].sum() / hold_n[ok].sum().clamp_min(1)).sqrt()) if ok.any() else None,
               hold_speed_rms=float((hold_v[ok].sum() / hold_n[ok].sum().clamp_min(1)).sqrt()) if ok.any() else None)
res["by_design"] = {designs[i]: float((~alive[did == i]).float().mean()) for i in range(D)}
(RUN / a.run / f"{a.out}.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "by_design"}))
