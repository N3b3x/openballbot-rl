"""Record a short closed-loop rollout of a balance checkpoint for the ROS 2 core parity test (simulation only, MuJoCo-Warp GPU).

The rollout runs the checkpoint exactly as the evaluation scripts do (args.json re-applied: arrestors, wheel-ball friction, LQR base,
adaptive residual authority, motion-reference layer) on a small batch of worlds, with training-like command resampling (velocity and
goal episodes, curving commands), domain randomisation and pushes on. Two phases:

  parity  observation noise and actuation latency OFF, so every policy input is a deterministic function of the logged simulator state.
          Logged per control tick: the simulator state the controller sees (qpos, qvel, base cvel), the motion-layer inputs and output,
          the clean 25-D observation, the full policy input vector, the policy action, the LQR action and its internal state, the residual
          authority, the action sent to the actuators and the actuator command.
  noisy   observation noise and latency ON (as in training): only the policy inputs and actions, used to verify the exported actor on
          realistic observations.

  experiments/mjx_port/.venv/Scripts/python.exe record_parity.py --run g34_mref_rsad_db_s0 --n 3 --steps 600
Output: outputs/study/warp_ppo/<run>/parity_rollout.npz
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUN = HERE.parents[1] / "outputs" / "study" / "warp_ppo"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="g34_mref_rsad_db_s0")
ap.add_argument("--n", type=int, default=3, help="worlds per design (all designs of the checkpoint)")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--noisy-steps", type=int, default=300)
ap.add_argument("--resample", type=int, default=200, help="command resampling period in control steps")
ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()

args = json.load(open(RUN / a.run / "args.json"))
if args.get("retainer"):
    os.environ["RETAINER"] = args["retainer"]
if args.get("pair_mu1") is not None:
    os.environ["PAIR_MU1"] = f"{args['pair_mu1']:g}"
import train_dev as T  # noqa: E402  (env vars above must be set before the models are loaded)
import tiles as TL  # noqa: E402
from lqr_gpu import GpuLQR  # noqa: E402

dev = "cuda"
ck = torch.load(RUN / a.run / "model.pt", map_location=dev)
designs = ck.get("designs") or ["nom"]
net = T.Net(0, 3, hdim=ck["hdim"], pdim=ck["pdim"], use_priv=False, ddim=ck["ddim"], width=ck["width"], nlayers=ck["nlayers"]).to(dev)
net.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
mean, var = ck["mean"].to(dev), ck["var"].to(dev)
pol = lambda o: net(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1)

D = len(designs); N = D * a.n
env = T.GpuBallbot(N, seed=a.seed, dr=True, design=designs[0], designs=designs if D > 1 else None, tiles=True, hist_k=ck["hist"],
                   asym=ck["asym"], surfaces=True, pull=bool(args.get("pull")), goal_p=float(args.get("goal_p", 0.0)), design_cond=ck["ddim"] > 0)
env.goal_centre = bool(args.get("goal_centre"))      # S118: goal logic on the true ball centre for runs trained that way
env.lqr, env.rs = GpuLQR(env, designs, a_max=args.get("lqr_amax", 0.5), r=args.get("lqr_r", 1.0)), args["lqr_base"]
if args.get("rs_adapt"):
    env.rs_adapt = [float(x) for x in args["rs_adapt"].split(",")]
if args.get("motion_ref"):
    env.enable_motion_ref(curve_p=args.get("curve_p", 0.5))
TL.EXIT_R = 99.0
env.push_scale = 1.0
env.resample, env.max_steps = a.resample, 10 ** 9

# ---- phase 1 settings: deterministic policy inputs (no observation noise, no actuation latency)
env.drc["obs_noise"] = 0.0
env.dr_l = 0.0

all_ = torch.ones(N, dtype=torch.bool, device=dev)
env.reset(all_)
w = torch.arange(N, device=dev)
if env.K > 1:
    env.set_design(w, w % D)      # every design represented
# start each world near the centre of its tile (the tile chosen by reset), standing still, at its design's rest pose
c = env.tile_xy[env.cur_tile]
xs = c[:, 0] + torch.rand(N, device=dev) * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device=dev) * 0.4 - 0.2
yaw = torch.rand(N, device=dev) * 2 * math.pi - math.pi
q = env.q0_tab[env.didx].clone()
q[:, 3] = torch.cos(yaw / 2); q[:, 4] = 0; q[:, 5] = 0; q[:, 6] = torch.sin(yaw / 2)
q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0; env.ctrl[:] = 0
env.steps[:] = 0
env.vint[:] = 0; env.last_a[:] = 0; env.hist[:] = 0
env.lqr.reset(all_)
if hasattr(env, "_vel"):
    del env._vel                  # the first LQR call must see a zero measured velocity, as after a reset

# ---- hooks: log the motion-layer call (inputs and output) and the LQR call
MREF = []
_mstep = env.mref.step


def mref_hook(v_des, f_w, pos=None, goal=None, goal_on=None):
    st0 = (env.mref.s.clone(), env.mref.th.clone(), env.mref.hold.clone())
    out = _mstep(v_des, f_w, pos, goal, goal_on=goal_on)
    MREF.append(dict(v_des=v_des.clone(), pos=pos.clone(), goal=goal.clone(), goal_on=goal_on.clone(), out=out.clone(),
                     s0=st0[0], th0=st0[1], hold0=st0[2]))
    return out


env.mref.step = mref_hook
LQR = []
_lact = env.lqr.act


def lqr_hook():
    u = _lact()
    LQR.append(dict(u=u.clone(), vref=env.lqr.vref.clone(), int_v=env.lqr.int_v.clone()))
    return u


env.lqr.act = lqr_hook

# initial command for tick 0: fresh commands and goals, motion layer started from rest facing the command
env.sample_cmd(all_)
env.mref.reset(all_, heading=torch.atan2(env.cmd[:, 1], env.cmd[:, 0]))
env.update_goal_cmd()
env._shape_cmd(all_)
o_clean = env.reset_obs()
o = env.noisy(o_clean)

L = {k: [] for k in ("qpos", "qvel", "cvel", "cmd", "obs_clean", "x", "act", "lqr", "lqr_vref", "lqr_int", "rs", "sent", "ctrl", "tq",
                     "v_des", "mpos", "goal", "goal_on", "m_s0", "m_th0", "m_hold0", "fell")}
z6 = torch.zeros(N, 6, device=dev)
cvel_t = z6      # base cvel seen by the tick-0 observation: zero (reset_obs), the robot is at rest
for t in range(a.steps):
    m = MREF[t]
    L["qpos"].append(env.qpos.clone()); L["qvel"].append(env.qvel.clone()); L["cvel"].append(cvel_t.clone()); L["cmd"].append(env.cmd.clone())
    L["v_des"].append(m["v_des"]); L["mpos"].append(m["pos"]); L["goal"].append(m["goal"]); L["goal_on"].append(m["goal_on"])
    L["m_s0"].append(m["s0"]); L["m_th0"].append(m["th0"]); L["m_hold0"].append(m["hold0"])
    L["obs_clean"].append(o_clean.clone()); L["x"].append(o.clone())
    with torch.no_grad():
        act = pol(o)
    L["act"].append(act.clone())
    o_clean, r, fell, trunc, info = env.step(act)
    o_clean = o_clean.clone()
    o = env.noisy(o_clean)
    cvel_t = env.cvel[:, env.base].clone()
    L["lqr"].append(LQR[t]["u"]); L["lqr_vref"].append(LQR[t]["vref"]); L["lqr_int"].append(LQR[t]["int_v"])
    lo, hi = env.rs_adapt[0], env.rs_adapt[1]
    L["rs"].append(lo + (hi - lo) * env._rs_w)
    L["sent"].append(env.prev_a.clone()); L["ctrl"].append(env.ctrl.clone()); L["tq"].append(env.tq.clone()); L["fell"].append(fell.clone())

npy = lambda v: torch.stack(v).cpu().numpy()
out = {k: npy(v) for k, v in L.items()}
# per-world design constants of the parity phase (phase 2 resets fallen worlds, which draws new designs)
out.update(didx=env.didx.cpu().numpy(), f_w=env.f_w.cpu().numpy(), tau_w=env.tau_w.cpu().numpy())

# ---- phase 2: noisy observations and latency as in training, policy inputs and actions only
env.drc["obs_noise"] = 0.01
env.dr_l = 1.0
env.set_dr_scale(float(args.get("dr_scale", 1.0)))
env.drc["obs_noise"] = 0.01
X2, A2 = [], []
for t in range(a.noisy_steps):
    with torch.no_grad():
        act = pol(o)
    X2.append(o.clone()); A2.append(act.clone())
    o_clean, r, fell, trunc, info = env.step(act)
    o = env.noisy(o_clean)
    env.reset(fell)
    o = torch.where(fell[:, None], env.noisy(env.reset_obs()), o)
out["x_noisy"], out["act_noisy"] = npy(X2), npy(A2)

out.update(dvec_tab=env.dvec_tab.cpu().numpy(),
           q0_tab=env.q0_tab.cpu().numpy(), designs=np.array(designs), rs_adapt=np.array(env.rs_adapt), run=np.array(a.run),
           cmd_vmax=np.array(env.cmd_vmax), resample=np.array(a.resample),
           goal_centre=np.array(bool(getattr(env, "goal_centre", False))))
dst = RUN / a.run / "parity_rollout.npz"
np.savez_compressed(dst, **out)
alive = ~np.logical_or.accumulate(out["fell"], 0)
print(f"saved {dst}: {a.steps} ticks x {N} worlds ({D} designs), alive at end {alive[-1].mean():.2f}, "
      f"goal-mode share {out['goal_on'].mean():.2f}, noisy phase {a.noisy_steps} ticks")
