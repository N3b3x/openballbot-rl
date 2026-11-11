"""DAgger distillation: privileged teacher (height scan + design vector) -> deployable student (proprioception + history + noisy depth sensor).

The student acts in the environment; the teacher labels the student's own states with its action; the student regresses to those labels on a replay of recent rollouts.
Teacher and student see the same world state, only their inputs differ:
  teacher input = [obs25 (noisy), privileged height scan 35-D, design vector 6-D (if the teacher was trained with it)]    (teacher trained with --sensor scan --design-cond, hist 0)
  student input = [obs25 (noisy), depth sensor (noisy rays), history of the last K (obs, action) pairs, optional design vector]
  python distill.py --teacher NAME --name NAME_student --tiles --surfaces --pull --goal-p 0.4 --designs nom,... --width 512 --layers 3 --hist 16 --steps 150e6
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_dev as T

DEV = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--nworld", type=int, default=4096); ap.add_argument("--steps", type=float, default=100e6)
ap.add_argument("--n-steps", type=int, default=32); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--minibatches", type=int, default=8)
ap.add_argument("--replay", type=int, default=4, help="number of recent rollouts kept in the replay buffer")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lr-end", type=float, default=1e-4)
ap.add_argument("--beta0", type=float, default=0.5, help="initial probability of executing the teacher action; linear decay to 0 over the first 20 percent")
ap.add_argument("--width", type=int, default=512); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--hist", type=int, default=16)
ap.add_argument("--sensor", default="depth"); ap.add_argument("--mount", default="head"); ap.add_argument("--design-cond", action="store_true")
ap.add_argument("--tiles", action="store_true"); ap.add_argument("--surfaces", action="store_true"); ap.add_argument("--pull", action="store_true")
ap.add_argument("--goal-p", type=float, default=0.4); ap.add_argument("--designs", default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-dr", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed)
RUN = T.ROOT / "outputs" / "study" / "warp_ppo"
out = RUN / a.name; out.mkdir(parents=True, exist_ok=True)
json.dump(vars(a), open(out / "args.json", "w"))

ck = torch.load(RUN / a.teacher / "model.pt", map_location=DEV)
assert ck.get("sdim") == 35 and ck.get("hist", 0) == 0, "teacher must be trained with --sensor scan and no history"
tnet = T.Net(35, 3, hdim=0, pdim=ck.get("pdim", 0), use_priv=False, ddim=ck.get("ddim", 0), width=ck.get("width", 256), nlayers=ck.get("nlayers", 2)).to(DEV)
tnet.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")}); tnet.eval()
t_mean, t_var = ck["mean"].to(DEV), ck["var"].to(DEV)
t_dd, t_pd = ck.get("ddim", 0), ck.get("pdim", 0)
designs = a.designs.split(",") if a.designs else (ck.get("designs") or None)
env = T.GpuBallbot(a.nworld, a.seed, dr=not a.no_dr, design=(designs or ["nom"])[0], designs=designs if designs and len(designs) > 1 else None, tiles=a.tiles,
                   surfaces=a.surfaces, pull=a.pull, goal_p=a.goal_p, sensor=a.sensor, mount=a.mount, hist_k=a.hist, design_cond=a.design_cond)
N, TT = a.nworld, a.n_steps


def teacher_action(o_student):
    base = o_student[:, :T.OBS_DIM]
    parts = [base, env.scan_feat()]
    if t_dd:
        parts.append(env.dvec_tab[env.didx])
    if t_pd:
        parts.append(torch.zeros(N, t_pd, device=DEV))
    x = ((torch.cat(parts, 1) - t_mean) / torch.sqrt(t_var + 1e-8)).clamp(-10, 10)
    with torch.no_grad():
        return tnet(x).clamp(-1, 1)


student = T.Net(env.sdim, 3, hdim=env.hdim, pdim=0, use_priv=False, ddim=env.ddim, width=a.width, nlayers=a.layers).to(DEV)
opt = torch.optim.Adam(student.parameters(), lr=a.lr, fused=True)
on = T.RunNorm(env.obs_dim)
o_raw = env.noisy(env.obs()); on.update(o_raw)
n_iter = int(a.steps // (N * TT)); total, t0 = 0, time.time()
buf_o, buf_a = [], []
log = open(out / "log.csv", "w"); log.write("iter,steps,time_s,loss,fall_rate,beta,student_teacher_mse\n")
for it in range(n_iter):
    lr = a.lr + (a.lr_end - a.lr) * it / max(1, n_iter - 1)
    for g in opt.param_groups: g["lr"] = lr
    beta = a.beta0 * max(0.0, 1 - it / (0.2 * n_iter))
    O, A = [], []
    z0 = torch.zeros((), device=DEV); falls, gap = z0.clone(), z0.clone()
    for t in range(TT):
        x = on(o_raw)
        with torch.no_grad():
            sa = student(x).clamp(-1, 1)
        ta = teacher_action(o_raw)
        use_t = (torch.rand(N, device=DEV) < beta)[:, None]
        act = torch.where(use_t, ta, sa)
        O.append(x); A.append(ta)
        gap += ((sa - ta) ** 2).mean()
        o_next, r, fell, trunc, info = env.step(act)
        falls += fell.sum()
        done = fell | trunc
        env.reset(done)
        o_next = torch.where(done[:, None], env.reset_obs(), o_next)
        o_raw = env.noisy(o_next); on.update(o_raw)
    buf_o.append(torch.cat(O)); buf_a.append(torch.cat(A))
    buf_o, buf_a = buf_o[-a.replay:], buf_a[-a.replay:]
    fo, fa = torch.cat(buf_o), torch.cat(buf_a)
    M = fo.shape[0]; mb = M // (a.minibatches * len(buf_o))
    for ep in range(a.epochs):
        perm = torch.randperm(M, device=DEV)
        for k in range(a.minibatches * len(buf_o)):
            ix = perm[k * mb:(k + 1) * mb]
            loss = ((student(fo[ix]) - fa[ix]) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
    total += N * TT
    if it % 5 == 0 or it == n_iter - 1:
        row = f"{it},{total},{time.time()-t0:.0f},{float(loss):.5f},{float(falls)/(N*TT):.5f},{beta:.3f},{float(gap)/TT:.5f}"
        log.write(row + "\n"); log.flush(); print(row, f"| {total/(time.time()-t0)/1e3:.0f}k env-steps/s", flush=True)
    if it % 50 == 0 or it == n_iter - 1:
        torch.save(dict(ac={"pi." + k: v for k, v in student.state_dict().items()}, mean=on.mean.cpu(), var=on.var.cpu(), steps=total, sdim=env.sdim, hdim=env.hdim,
                        pdim=0, ddim=env.ddim, width=a.width, nlayers=a.layers, hist=a.hist, asym=False, designs=env.design_names, goal_p=a.goal_p), out / "model.pt")
print("done", total, "steps in", time.time() - t0, "s")
