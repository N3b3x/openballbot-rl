"""Terrain-aware navigation around unclimbable obstacles (simulation only, MuJoCo-Warp GPU).

Map: every tile is a 'blocks' tile (tiles.py): flat ground, four 8 cm blocks across the tile axes 0.75 m from the centre (well above the
~2.6 cm traction limit of the 9 cm ball, log S88), two smaller random blocks. Each world starts at its tile centre and must reach a goal
1.2 m away along a random axis, i.e. behind a block. The balance controller is the frozen blind checkpoint (LQR + residual, arrestors).
A navigation layer above it sets the planar velocity command every control step:
  candidate headings: 24 directions; a heading is blocked if any known obstacle point lies in a corridor along it
  (0.05 to `look` m ahead, half width `half_w`); the chosen heading minimises |heading - goal bearing| + 0.3 |heading - previous|;
  speed min(v_max, 1.5 * goal distance), Froude-scaled; no free heading -> stop.
Obstacle points (an obstacle = ground more than `thr` above the ground under the ball):
  blind      no perception, straight to the goal (baseline)
  oracle     privileged height map sampled on the candidate corridors (upper bound)
  sensor     a simulated depth sensor: every noisy range reading (range-proportional noise, 2 % dropout, as in training) is turned into a
             3-D hit point with the known sensor pose; hits above `thr` are kept in a per-world memory of the last `mem` points
             (world frame), which the corridor test uses. Configs: --res WxH --fov H,V (deg) --pitch (deg) --up (m) --mount head|cones.
  TRAIN_MOD=train_dev PUSH_SCALE=0 python eval_nav.py --run g18_ret_rs025_s1 --source sensor --res 18x14 --fov 55,42 --out nav_tof
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
T = importlib.import_module(os.environ.get("TRAIN_MOD", "train_dev")); import tiles as TL
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--source", choices=["blind", "oracle", "sensor", "bump"], default="blind", help="bump = blind with contact-based escape (escape.py)"); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=48, help="worlds per design"); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--pull", action="store_true")
ap.add_argument("--res", default="18x14"); ap.add_argument("--fov", default="55,42"); ap.add_argument("--pitch", type=float, default=-25.0); ap.add_argument("--up", type=float, default=0.15)
ap.add_argument("--mount", choices=["head", "cones"], default="head"); ap.add_argument("--noise", type=float, default=0.005)
ap.add_argument("--look", type=float, default=0.6); ap.add_argument("--half-w", type=float, default=0.22); ap.add_argument("--thr", type=float, default=0.03)
ap.add_argument("--v-max", type=float, default=0.3); ap.add_argument("--hyst", type=float, default=1.0); ap.add_argument("--carrot", type=float, default=0.2); ap.add_argument("--replan", type=int, default=20); ap.add_argument("--aim", choices=["cmd", "target"], default="cmd", help="sensor gimbal: along the command (training assumption) or at the planner target heading"); ap.add_argument("--prox", type=float, default=0.0, help="sensor maps: proximity slow-down range (m), 0 = off"); ap.add_argument("--w-max", type=float, default=3.0); ap.add_argument("--a-max", type=float, default=0.6); ap.add_argument("--record", type=int, default=-1, help="save the trajectory of this world index for rendering"); ap.add_argument("--planner", choices=["smooth", "wave", "corridor"], default="smooth"); ap.add_argument("--mem", type=int, default=600); ap.add_argument("--goal-r", type=float, default=1.2)
a = ap.parse_args()
dev = "cuda"
ck = torch.load(RUN / a.run / "model.pt", map_location=dev)
_a0 = json.load(open(RUN / a.run / "args.json"))
if _a0.get("retainer") and not os.environ.get("RETAINER"):
    os.environ["RETAINER"] = _a0["retainer"]
if _a0.get("pair_mu1") is not None and not os.environ.get("PAIR_MU1"):      # contact model of the run (S101 fix)
    os.environ["PAIR_MU1"] = f"{_a0['pair_mu1']:g}"
designs = ck.get("designs") or ["nom"]
net = T.Net(0, 3, hdim=ck.get("hdim", 0), pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to(dev)
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to(dev), ck["var"].to(dev)
res = tuple(int(x) for x in a.res.split("x"))
if a.source == "sensor":
    fh, fv = (float(x) for x in a.fov.split(","))
    T.SENSOR_FOV = (np.radians(fh), np.radians(fv)); T.SENSOR_PITCH = np.radians(a.pitch); T.SENSOR_UP = a.up
D = len(designs); N = D * a.n
env = T.GpuBallbot(N, seed=11, dr=True, design=designs[0], designs=designs if D > 1 or ck.get("ddim", 0) else None, tiles=True, tile_kinds=["blocks"],
                   sensor="depth" if a.source == "sensor" else "none", mount=a.mount, depth_res=res, depth_noise=a.noise,
                   hist_k=ck.get("hist", 0), asym=ck.get("asym", False), surfaces=True, pull=a.pull, goal_p=0.0, design_cond=ck.get("ddim", 0) > 0)
SD = env.sdim
pol = lambda o: net(((torch.cat([o[:, :T.OBS_DIM], o[:, T.OBS_DIM + SD:]], 1) - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)
env.push_scale = float(os.environ.get("PUSH_SCALE", "1"))
env.goal_centre = bool(_a0.get("goal_centre")); env.db_centre = bool(_a0.get("db_centre"))
if os.environ.get("NAV_ORIGIN"): env.goal_centre = False      # diagnostic: planner, motion-layer hold and success on the ball body origin even for goal-centre runs      # S118: goal logic on the true ball centre if the run was trained that way
if _a0.get("lqr_base", 0) > 0:
    from lqr_gpu import GpuLQR
    env.lqr, env.rs = GpuLQR(env, designs, a_max=_a0.get("lqr_amax", 0.5), r=_a0.get("lqr_r", 1.0)), _a0["lqr_base"]
    if _a0.get("rs_adapt"): env.rs_adapt = [float(x) for x in _a0["rs_adapt"].split(",")]      # S111 adaptive authority
TL.EXIT_R = 99.0      # no tile-exit truncation: the episode ends at the goal, a fall or the time limit
env.resample, env.max_steps = 10 ** 9, 10 ** 9
env.reset(torch.ones(N, dtype=torch.bool, device=dev)); env.steps[:] = 0
w = torch.arange(N, device=dev); did = w // a.n
if env.K > 1:
    env.set_design(w, did)
tpos = w % len(env.tile_info); env.cur_tile[:] = tpos
c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device=dev) * 0.2 - 0.1; ys = c[:, 1] + torch.rand(N, device=dev) * 0.2 - 0.1
q = env.q0_tab[env.didx].clone(); q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
axis = torch.randint(0, 4, (N,), device=dev)
dirs4 = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=dev)
goal = c + dirs4[axis] * a.goal_r
env.goal_mode[:] = False
env.cmd = torch.zeros(N, 3, device=dev)

NH = 24
cand = torch.arange(NH, device=dev) * (2 * np.pi / NH)
cu = torch.stack([torch.cos(cand), torch.sin(cand)], 1)                      # (NH, 2)
along = torch.linspace(0.05, a.look, 12, device=dev); across = torch.linspace(-a.half_w, a.half_w, 5, device=dev)
prev_h = torch.atan2(goal[:, 1] - ys, goal[:, 0] - xs)
mem = torch.full((N, a.mem, 2), 1e6, device=dev); mem_i = 0
R_BALL = 0.09


def ground_z():
    return env.gxpos[:, env.ball_g, 2] - R_BALL * env.f_w ** 2      # ball radius scales with f^2 (f = sqrt(r / 0.09))


def blocked_oracle(px, py, lim):
    """Corridor test on the privileged map: (N, NH) bool."""
    bx, by = env.ball_xy()[:, 0], env.ball_xy()[:, 1]
    g0 = ground_z()
    # sample points: (N, NH, len(along), len(across))
    ox = along[None, None, :, None] * cu[None, :, 0, None, None] - across[None, None, None, :] * cu[None, :, 1, None, None]
    oy = along[None, None, :, None] * cu[None, :, 1, None, None] + across[None, None, None, :] * cu[None, :, 0, None, None]
    X = bx[:, None, None, None] + ox; Y = by[:, None, None, None] + oy
    h = env.hgt(X.reshape(N, -1), Y.reshape(N, -1)).reshape(X.shape)
    inside = (along[None, None, :, None] <= lim[:, None, None, None]).expand_as(h)
    return (((h - g0[:, None, None, None]) > a.thr) & inside).flatten(2).any(2)


def sensor_hits(o):
    """Noisy ranges from the observation -> world hit points (N, R, 3)."""
    d = o[:, T.OBS_DIM:T.OBS_DIM + SD]
    Rd = env.ray_dirs
    if a.mount == "cones":
        qw, qx, qy, qz = env.qpos[:, 3], env.qpos[:, 4], env.qpos[:, 5], env.qpos[:, 6]
        Rm = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                          2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                          2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1).reshape(-1, 3, 3)
        v = torch.einsum("nij,rj->nri", Rm, Rd); p0 = env.qpos[:, None, 0:3] + torch.einsum("nij,rj->nri", Rm, env.ray_off)
    else:
        yaw = env.sensor_yaw(); cy, sy = torch.cos(yaw), torch.sin(yaw)
        v = torch.stack([Rd[:, 0][None] * cy[:, None] - Rd[:, 1][None] * sy[:, None], Rd[:, 0][None] * sy[:, None] + Rd[:, 1][None] * cy[:, None],
                         Rd[:, 2][None].expand(N, -1)], -1)
        p0 = env.qpos[:, None, 0:3].clone(); p0[..., 2] += T.SENSOR_UP
    valid = d < T.SENSOR_RANGE[1] - 1e-3
    return p0 + v * d[..., None], valid


def blocked_memory(lim):
    bx, by = env.ball_xy()[:, 0], env.ball_xy()[:, 1]
    rel = mem - torch.stack([bx, by], 1)[:, None]                                   # (N, M, 2)
    al = torch.einsum("nmk,hk->nhm", rel, cu); ac = torch.einsum("nmk,hk->nhm", rel, torch.stack([-cu[:, 1], cu[:, 0]], 1))
    return ((al > 0.05) & (al < lim[:, None, None]) & (ac.abs() < a.half_w)).any(2)


def plan(o):
    global prev_h, mem, mem_i
    bx, by = env.ball_xy()[:, 0], env.ball_xy()[:, 1]
    eg = goal - torch.stack([bx, by], 1); dg = eg.norm(dim=1); hg = torch.atan2(eg[:, 1], eg[:, 0])
    if a.source in ("blind", "bump"):
        blk = torch.zeros(N, NH, dtype=torch.bool, device=dev)
    elif a.source == "oracle":
        blk = blocked_oracle(bx, by, torch.minimum(dg, torch.full_like(dg, a.look)))
    else:
        P, valid = sensor_hits(o)
        obst = valid & ((P[..., 2] - ground_z()[:, None]) > a.thr) & ((P[..., :2] - torch.stack([bx, by], 1)[:, None]).norm(dim=2) < 1.5)
        k = P.shape[1]
        idx = (mem_i + torch.arange(k, device=dev)) % a.mem
        newp = torch.where(obst[..., None], P[..., :2], mem[:, idx])
        mem[:, idx] = newp; mem_i = (mem_i + k) % a.mem
        blk = blocked_memory(torch.minimum(dg, torch.full_like(dg, a.look)))
    wrap = lambda x: torch.atan2(torch.sin(x), torch.cos(x))
    if a.source == "bump":      # blind + contact memory + back-up recovery (escape.py) on the smooth planner
        pos = torch.stack([bx, by], 1)
        v = SP.command(pos, env.f_w, lambda: SP.occupancy_points(mem))
        q_ = env.qpos[:, 3:7]; tilt = torch.rad2deg(torch.acos((1 - 2 * (q_[:, 1] ** 2 + q_[:, 2] ** 2)).clamp(-1, 1)))
        vel_ = env._vel[:, :2] if hasattr(env, "_vel") else torch.zeros_like(pos)
        out, newp, stall = ESC.update(pos, vel_, v, tilt, goal, env.f_w, ACTIVE["m"])
        if newp is not None:
            k = newp.shape[1]
            idx = (mem_i + torch.arange(k, device=dev)) % a.mem
            mem[:, idx] = torch.where(stall[:, None, None], newp, mem[:, idx]); mem_i = (mem_i + k) % a.mem
            SP.d = None      # re-plan now with the new obstacle
        env.cmd = torch.cat([out, torch.zeros_like(out[:, :1])], 1)
        return None
    if a.planner == "smooth" and a.source != "blind":
        occ_fn = SP.occupancy_oracle if a.source == "oracle" else (lambda: SP.occupancy_points(mem))
        v = SP.command(torch.stack([bx, by], 1), env.f_w, occ_fn, obs_pts=(mem if a.source == 'sensor' else None), prox=a.prox)
        if a.aim == 'target':
            env.yaw_override = SP.h_target.clone()
        env.cmd = torch.cat([v, torch.zeros_like(v[:, :1])], 1)
        return None
    if a.planner == "wave" and a.source != "blind":
        return wave_plan(bx, by, dg)
    cost = wrap(cand[None] - hg[:, None]).abs() + a.hyst * wrap(cand[None] - prev_h[:, None]).abs()
    cost = torch.where(blk, torch.full_like(cost, 1e9), cost)
    best = cost.argmin(1); h = cand[best]; free = ~blk.gather(1, best[:, None])[:, 0]
    prev_h = torch.where(free, h, prev_h)
    sp = torch.minimum(1.5 * dg, torch.full_like(dg, a.v_max)) * env.f_w
    sp = torch.where(free & (dg > 0.05), sp, torch.zeros_like(sp))
    env.cmd = torch.stack([sp * torch.cos(h), sp * torch.sin(h), torch.zeros_like(sp)], 1)
    return blk


G, CS = 64, 0.05                       # wavefront grid: 64 x 64 cells of 5 cm around the tile centre (3.2 m window)
gi = (torch.arange(G, device=dev) - G / 2 + 0.5) * CS
GX = c[:, 0, None, None] + gi[None, None, :]; GY = c[:, 1, None, None] + gi[None, :, None]     # (N, G, G): rows y, cols x
DF = {"d": None, "t": 0}
import torch.nn.functional as Fn


def occupancy(bx, by):
    if a.source == "oracle":
        h = env.hgt(GX.expand(N, G, G).reshape(N, -1), GY.expand(N, G, G).reshape(N, -1)).reshape(N, G, G)
        return (h - env.hgt(c[:, 0], c[:, 1])[:, None, None]) > a.thr
    ix = ((mem[..., 0] - c[:, 0, None]) / CS + G / 2).floor().long(); iy = ((mem[..., 1] - c[:, 1, None]) / CS + G / 2).floor().long()
    ok_ = (ix >= 0) & (ix < G) & (iy >= 0) & (iy < G)
    flat = torch.where(ok_, iy * G + ix, torch.zeros_like(ix))
    occ = torch.zeros(N, G * G, device=dev)
    occ.scatter_add_(1, flat, ok_.float())
    return (occ > 0).reshape(N, G, G)


def wave_plan(bx, by, dg):
    """Distance-to-goal field by wavefront propagation on the inflated occupancy grid (recomputed every 20 steps); the robot
    steers to the lowest-cost of 24 points 0.15 m around it. No local minima on a static map, unlike the corridor rule."""
    if DF["d"] is None or DF["t"] % 20 == 0:
        occ = occupancy(bx, by).float()[:, None]
        k = int(round(a.half_w / CS)) * 2 + 1
        infl = Fn.max_pool2d(occ, k, stride=1, padding=k // 2)[:, 0] > 0      # obstacles grown by the robot half width
        gx = ((goal[:, 0] - c[:, 0]) / CS + G / 2).floor().long().clamp(0, G - 1); gy = ((goal[:, 1] - c[:, 1]) / CS + G / 2).floor().long().clamp(0, G - 1)
        d = torch.full((N, G, G), 1e4, device=dev)
        d[torch.arange(N, device=dev), gy, gx] = 0
        infl[torch.arange(N, device=dev), gy, gx] = False
        for _ in range(110):
            nb = -Fn.max_pool2d(-d[:, None], 3, stride=1, padding=1)[:, 0] + 1.0
            d = torch.where(infl, torch.full_like(d, 1e4), torch.minimum(d, nb))
        DF["d"] = d
    DF["t"] += 1
    d = DF["d"]
    px = bx[:, None] + 0.15 * cu[None, :, 0]; py = by[:, None] + 0.15 * cu[None, :, 1]
    ix = ((px - c[:, 0, None]) / CS + G / 2).floor().long().clamp(0, G - 1); iy = ((py - c[:, 1, None]) / CS + G / 2).floor().long().clamp(0, G - 1)
    vals = d[torch.arange(N, device=dev)[:, None], iy, ix]                   # (N, 24)
    best = vals.argmin(1); h = cand[best]
    reachable = vals.gather(1, best[:, None])[:, 0] < 1e4
    sp = torch.minimum(1.5 * dg, torch.full_like(dg, a.v_max)) * env.f_w
    sp = torch.where(reachable & (dg > 0.05), sp, torch.zeros_like(sp))
    env.cmd = torch.stack([sp * torch.cos(h), sp * torch.sin(h), torch.zeros_like(sp)], 1)


from nav import WavePlanner
from escape import ContactEscape
ESC = ContactEscape(N, dev)
ACTIVE = {"m": torch.ones(N, dtype=torch.bool, device=dev)}
SP = WavePlanner(env, torch.arange(N, device=dev), c, goal, half_w=a.half_w, thr=a.thr, v_max=a.v_max, carrot=a.carrot, w_max=a.w_max, a_max=a.a_max, replan=a.replan)
_orig = env.update_goal_cmd
cur_obs = {"o": None}


MREF = None
if _a0.get("motion_ref") and not os.environ.get("NO_MREF"):      # S114: the planner's command through the motion layer the run was trained with (arrival braking and hold at the goal)
    from motion_ref import MotionRef
    _mkw = {k: float(v) for k, v in (p.split("=") for p in os.environ["MREF_KW"].split(","))} if os.environ.get("MREF_KW") else {}      # e.g. MREF_KW=a_acc=0.6,a_brk=0.8,a_lat=0.5
    MREF = MotionRef(N, dev, **_mkw)
    _g0 = goal - env.ball_xy()
    MREF.reset(torch.ones(N, dtype=torch.bool, device=dev), heading=torch.atan2(_g0[:, 1], _g0[:, 0]))


def nav_update():
    _orig()
    if cur_obs["o"] is not None:
        plan(cur_obs["o"])
        if MREF is not None:
            env.cmd = torch.cat([MREF.step(env.cmd[:, :2], env.f_w, env.ball_xy(), goal), env.cmd[:, 2:]], 1)


env.update_goal_cmd = nav_update
o = env.noisy(env.reset_obs()); cur_obs["o"] = o; plan(o)
fell_any = torch.zeros(N, dtype=torch.bool, device=dev); done_ok = torch.zeros(N, dtype=torch.bool, device=dev); frozen = torch.zeros(N, dtype=torch.bool, device=dev)
t_reach = torch.full((N,), float(a.steps), device=dev); hit_block = torch.zeros(N, dtype=torch.bool, device=dev)
REC = []
path = torch.zeros(N, device=dev); prev_xy = env.ball_xy().clone(); prev_cmd = env.cmd[:, :2].clone(); hrate2 = torch.zeros(N, device=dev); jerk = torch.zeros(N, device=dev); nmove = torch.zeros(N, device=dev)
for t in range(a.steps):
    with torch.no_grad():
        act = pol(o)
    o, r, fell, trunc, info = env.step(act); o = env.noisy(o); cur_obs["o"] = o
    if a.record >= 0 and not bool(frozen[a.record]):
        k_ = a.record
        REC.append(dict(qpos=env.qpos[k_].cpu().numpy().copy(), cmd=env.cmd[k_].cpu().numpy().copy(), goal=goal[k_].cpu().numpy().copy(), tilt=float(info['tilt'][k_]), speed=float(env._vel[k_, :2].norm()), fell=bool(fell[k_]), syaw=float(env.sensor_yaw()[k_])))
    alive = ~frozen
    xy = env.ball_xy()
    path += torch.where(alive & ~fell, (xy - prev_xy).norm(dim=1), torch.zeros_like(path)); prev_xy = xy.clone()
    cmv = env.cmd[:, :2]; mov = (cmv.norm(dim=1) > 0.05) & (prev_cmd.norm(dim=1) > 0.05) & alive & ~fell
    dh = torch.atan2(prev_cmd[:, 0] * cmv[:, 1] - prev_cmd[:, 1] * cmv[:, 0], (prev_cmd * cmv).sum(1)) / 0.01
    hrate2 += torch.where(mov, dh ** 2, torch.zeros_like(dh)); nmove += mov.float(); jerk += torch.where(alive & ~fell, (cmv - prev_cmd).norm(dim=1) / 0.01, torch.zeros_like(path)); prev_cmd = cmv.clone()
    fell_any |= fell & alive
    on_block = (env.hgt(env.ball_xy()[:, 0], env.ball_xy()[:, 1]) - env.hgt(xs, ys)) > 0.04
    hit_block |= on_block & alive
    reach = alive & ~fell_any & ((goal - env.ball_xy()).norm(dim=1) < 0.2)
    t_reach = torch.where(reach & ~done_ok, torch.full_like(t_reach, t), t_reach); done_ok |= reach
    frozen |= fell_any | done_ok
    ACTIVE["m"] = ~frozen
ok = torch.ones(N, dtype=torch.bool, device=dev)
st = ~fell_any & ~done_ok
fd = (goal - env.ball_xy()).norm(dim=1); spd = env.cmd[:, :2].norm(dim=1)
from_c = (env.ball_xy() - c).norm(dim=1)
diag = dict(escape_events=ESC.events if a.source == "bump" else None, escape_gave_up=float(ESC.gave_up().float().mean()) if a.source == "bump" else None,
            escape_fails_median=float(ESC.fails.float().median()) if a.source == "bump" else None, stuck_final_goal_dist_median=float(fd[st].median()) if st.any() else None, stuck_cmd_zero_frac=float((spd[st] < 1e-3).float().mean()) if st.any() else None, stuck_dist_from_start_median=float(from_c[st].median()) if st.any() else None)


straight = (goal - torch.stack([xs, ys], 1)).norm(dim=1)


def cell(m):
    mm = m & done_ok
    return dict(path_ratio=float((path[mm] / straight[mm]).median()) if mm.any() else None, heading_rate_rms=float(torch.sqrt(hrate2[mm].sum() / nmove[mm].sum().clamp_min(1))) if mm.any() else None,
                cmd_change_mean=float((jerk[mm] / (t_reach[mm] + 1)).mean()) if mm.any() else None,
                fall=float(fell_any[m].float().mean()), success=float(done_ok[m].float().mean()), stuck=float((~fell_any & ~done_ok)[m].float().mean()),
                over_block=float(hit_block[m].float().mean()), median_time_s=float(t_reach[m & done_ok].median() * 0.01) if (m & done_ok).any() else None)


res = dict(run=a.run, source=a.source, position_point=('ball_centre' if env.goal_centre else 'ball_body_origin'), motion_ref=MREF is not None, mref_kw=os.environ.get("MREF_KW"), sensor=dict(res=a.res, fov=a.fov, pitch=a.pitch, up=a.up, mount=a.mount, noise=a.noise) if a.source == "sensor" else None,
           planner=dict(kind=a.planner, aim=a.aim, replan=a.replan, prox=a.prox, carrot=a.carrot, w_max=a.w_max, a_max=a.a_max, hyst=a.hyst, look=a.look, half_w=a.half_w, thr=a.thr, v_max=a.v_max, mem=a.mem, goal_r=a.goal_r), steps=a.steps, worlds=N,
           push_scale=env.push_scale, pull=a.pull, retainer=os.environ.get("RETAINER"), overall=cell(ok), diag=diag,
           by_design={designs[i]: cell(did == i) for i in range(D)})
(RUN / a.run / f"{a.out}.json").write_text(json.dumps(res, indent=1))
if REC:
    import pickle; pickle.dump(dict(rec=REC, design=designs[int(did[a.record])], tile=int(tpos[a.record]), source=a.source, sensor=res["sensor"]), open(RUN / a.run / f"{a.out}_traj.pkl", "wb"))
print(a.out, json.dumps(res["overall"]), json.dumps(diag))
