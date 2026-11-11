"""Batched GPU LQR for the Warp ballbot (same law as ballbot_gym/controllers/lqr.py: per-plane LQR on the derived planar model, velocity reference governor,
slow integral on the measured velocity, yaw-rate loop). Used as (1) a calibration baseline on the same worlds and benchmark as the learned policies and
(2) a possible imitation prior at low speed. Simulation only.

  TILES_FLAT=1 EVAL_NOEXIT=1 python lqr_gpu.py --speeds 0.3 0.6 0.9 --designs nom
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, scipy.linalg as la, torch
from scipy.signal import cont2discrete

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]


def design_tables(names, q_theta=50.0, q_phidot=1.0, q_thetadot=1.0, r=1.0, dt=0.01, dev="cuda"):
    """Per-design LQR gain K (discrete at the control period), wheel-to-ball map W, ball radius r_ball."""
    Ks, Ws, Rb = [], [], []
    for n in names:
        M = json.load(open(ROOT / "ballbot_gym" / "models" / "variants" / f"dz_{n}.json"))
        A, B = np.array(M["A"]), np.array(M["B"])
        Q = np.diag([1e-6, q_theta, q_phidot, q_thetadot])
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), dt)
        X = la.solve_discrete_are(Ad, Bd, Q * dt, np.array([[r * dt]]))
        K = np.linalg.solve(np.array([[r * dt]]) + Bd.T @ X @ Bd, Bd.T @ X @ Ad).ravel()
        Ks.append(K); Ws.append(np.array(M["params"]["W"])); Rb.append(M["params"]["r"])
    f = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=dev)
    return f(Ks), f(Ws), f(Rb)


def rotmat(q):
    """(N,4) quaternions (w,x,y,z) -> (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


class GpuLQR:
    def __init__(self, env, names, ki=1.0, k_yaw=5.0, a_max=0.5, dt=0.01, r=1.0):
        self.env, self.ki, self.k_yaw, self.a_max, self.dt = env, ki, k_yaw, a_max, dt
        self.K, self.W, self.rb = design_tables(names, r=r, dt=dt)
        self.vref = torch.zeros(env.N, 2, device="cuda"); self.int_v = torch.zeros(env.N, 2, device="cuda")

    def reset(self, mask):
        self.vref[mask] = 0; self.int_v[mask] = 0

    def act(self):
        e = self.env; di = e.didx
        R = rotmat(e.qpos[:, 3:7]); Rb = rotmat(e.qpos[:, 13:17])
        up = R[:, :, 2]
        tilt_y, tilt_x = torch.atan2(up[:, 0], up[:, 2]), torch.atan2(up[:, 1], up[:, 2])
        wb = torch.einsum("nij,nj->ni", R, e.qvel[:, 3:6])             # base angular velocity, world
        wk = torch.einsum("nij,nj->ni", Rb, e.qvel[:, 12:15])          # ball angular velocity, world
        cmd = e.cmd
        dv = cmd[:, :2] - self.vref
        nv = dv.norm(dim=1, keepdim=True).clamp_min(1e-9)
        lim = self.a_max * self.dt
        self.vref = self.vref + dv * torch.clamp(lim / nv, max=1.0)
        v_meas = e._vel[:, :2] if hasattr(e, "_vel") else torch.zeros_like(self.vref)
        self.int_v = (self.int_v + (self.vref - v_meas) * self.dt).clamp(-2, 2)
        rb = self.rb[di]
        K = self.K[di]
        taus = []
        for i, (th, thd, phd) in enumerate([(tilt_y, wb[:, 1], wk[:, 1]), (tilt_x, -wb[:, 0], -wk[:, 0])]):
            phd_ref = (self.vref[:, i] + self.ki * self.int_v[:, i]) / rb
            s = torch.stack([torch.zeros_like(th), th, phd - phd_ref, thd], 1)
            taus.append(-(K * s).sum(1))
        tau_z = -self.k_yaw * (cmd[:, 2] - wb[:, 2])
        T = torch.stack([-taus[1], taus[0], tau_z], 1)                  # torque on the ball, world axes
        Tb = torch.einsum("nji,nj->ni", R, T)                           # R^T T
        tau_w = torch.linalg.solve(self.W[di].transpose(1, 2), Tb[:, :, None])[:, :, 0]
        return (-tau_w / e.tau_w[:, None]).clamp(-1, 1)


def main():
    import tiles as TL
    import train_dev as T
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.3, 0.6, 0.9]); ap.add_argument("--designs", default="nom")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--steps", type=int, default=800); ap.add_argument("--dr", action="store_true"); ap.add_argument("--dr-mass", type=float, default=None, help="mass randomisation only, at this scale (all other terms and pushes off)"); ap.add_argument("--ki", type=float, default=1.0); ap.add_argument("--amax", type=float, default=0.5); ap.add_argument("--ball-fixed", type=float, default=None, help="fix the ball mass to this factor in every world (analysis)"); ap.add_argument("--refresh-const", action="store_true", help="with --ball-fixed: recompute invweight0 / subtreemass on the CPU model and write them into the GPU model"); ap.add_argument("--mass-body", default="both", choices=["both", "base", "ball"]); ap.add_argument("--r", type=float, default=1.0)
    a = ap.parse_args()
    names = a.designs.split(",")
    if os.environ.get("EVAL_NOEXIT"):
        TL.EXIT_R = 99.0
    D, K_, n = len(names), len(TL.TYPES), a.n
    N = D * n
    env = T.GpuBallbot(N, seed=7, dr=(a.dr or a.dr_mass is not None or a.ball_fixed is not None), design=names[0], designs=names if D > 1 else None, tiles=True, tile_seed=0)
    dev = "cuda"
    env.resample, env.max_steps = 10 ** 9, 10 ** 9
    if a.ball_fixed is not None:
        env.set_dr_scale(0.0); env.push_scale = 0.0
    if a.dr_mass is not None:
        env.set_dr_scale(0.0); env.dr_m = a.dr_mass; env.push_scale = 0.0
        env.dr_mbody = {"both": [1.0, 1.0], "base": [1.0, 0.0], "ball": [0.0, 1.0]}[a.mass_body]
    for sp in a.speeds:
        env.reset(torch.ones(N, dtype=torch.bool, device=dev)); env.steps[:] = 0
        w = torch.arange(N, device=dev); did = w // n
        if env.K > 1:
            env.set_design(w, did)
        flat_ids = (env.tile_kind == TL.TYPES.index("flat")).nonzero()[:, 0]
        if os.environ.get("EVAL_NOEXIT"):      # no tile-exit truncation: interior tiles only, so nobody drives off the map edge (log S92)
            flat_ids = flat_ids[env.tile_xy[flat_ids].abs().amax(1) <= 4.6]
        tpos = flat_ids[w % len(flat_ids)]
        env.cur_tile[:] = tpos
        c = env.tile_xy[tpos]; xs = c[:, 0] + torch.rand(N, device=dev) * 0.4 - 0.2; ys = c[:, 1] + torch.rand(N, device=dev) * 0.4 - 0.2
        q = env.q0_tab[env.didx].clone(); q[:, 0] += xs; q[:, 1] += ys; q[:, 10] += xs; q[:, 11] += ys
        hz = env.hgt(xs, ys); q[:, 2] += hz; q[:, 12] += hz
        env.qpos[:] = q; env.qvel[:] = 0; env.warm[:] = 0
        if a.ball_fixed is not None:
            env.t_mass[:, env.body_ids[1]] = env.mass_tab[env.didx, env.body_ids[1]] * a.ball_fixed
            _sub = env._subtree(env.t_mass)
            (env.t_sub if env.t_sub is not None else env.dw['body_subtreemass']).copy_(_sub)
            if a.refresh_const:
                import copy, mujoco, warp as wp
                m2 = copy.deepcopy(env.mjm); kb = env.body_ids[1]; m2.body_mass[kb] *= a.ball_fixed
                d2 = mujoco.MjData(m2); mujoco.mj_setConst(m2, d2)
                for nm in ("body_invweight0", "dof_invweight0", "body_subtreemass"):
                    arr = getattr(env.be.m, nm); new_np = np.asarray(getattr(m2, nm), dtype=np.float32)
                    cur = arr.numpy(); print("  refresh", nm, "old", np.round(cur.reshape(-1)[:6], 4), "new", np.round(new_np.reshape(-1)[:6], 4), flush=True)
                    arr.assign(new_np.reshape(cur.shape))
        ang = torch.rand(N, device=dev) * 2 * np.pi
        env.goal_mode[:] = False
        env.cmd = torch.stack([sp * env.f_w * torch.cos(ang), sp * env.f_w * torch.sin(ang), torch.zeros(N, device=dev)], 1)
        env._vel = torch.zeros(N, 3, device=dev)
        lqr = GpuLQR(env, names, ki=a.ki, a_max=a.amax, r=a.r)
        lqr.vref = torch.zeros(N, 2, device=dev); lqr.int_v = torch.zeros(N, 2, device=dev)
        fell_any = torch.zeros(N, dtype=torch.bool, device=dev)
        lerr = torch.zeros(N, device=dev); lrat = torch.zeros(N, device=dev); ln = torch.zeros(N, device=dev)
        for t in range(a.steps):
            act = lqr.act()
            o, r, fell, trunc, info = env.step(act)
            alive = ~fell_any
            fell_any |= fell & alive
            if t >= 300:
                cm = env.cmd[:, :2]; cn = cm.norm(dim=1).clamp_min(1e-6); ok = (alive & ~fell).float()
                lerr += (env._vel[:, :2] - cm).norm(dim=1) * ok; lrat += ((env._vel[:, :2] * cm).sum(1) / (cn * cn)) * ok; ln += ok
        for k in range(D):
            m = (w // n) == k
            print(f"LQR {names[k]:7s} cmd {sp}: steady-state err {float(lerr[m].sum() / ln[m].sum().clamp_min(1)):.3f}, achieved/command {float(lrat[m].sum() / ln[m].sum().clamp_min(1)):.2f}, falls {float(fell_any[m].float().mean()):.2f}", flush=True)


if __name__ == "__main__":
    main()
