"""Test the derived-model predictions on the behaviour of real controllers (LQR and learned policies).

  (1) Non-minimum phase: after a forward step command the ball must first move backward (the robot pushes the
      ball away to create the lean). We measure the initial reverse displacement of the ball.
  (2) Lean-acceleration law: theta_ss = kappa * a with kappa from the model. We regress the body lean (toward
      the motion) on the ball acceleration over step / reversal manoeuvres and compare the slope with kappa.

  python scripts/study/analyze_theory.py --runs lqr trk_g_s0 trk_h_s0 --config trk_d_if
Writes outputs/study/results/theory_checks.json and paper_v2/figures/fig_theory.pdf.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))
from policy_io import load_controller  # noqa: E402
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import ballbot_gym  # noqa: F401,E402


def rollout(env, ctrl, norm, cmd_fn, steps, seed=0):
    b = env.unwrapped
    m, d = b.model, b.data
    b.cmd_fixed = np.asarray(cmd_fn(0), float)
    obs, _ = env.reset(seed=seed)
    kd = m.joint("ball_free_joint").dofadr[0]
    g = m.geom("the_ball").id
    base = m.body("base").id
    rec = []
    for t in range(steps):
        c = np.asarray(cmd_fn(t), float)
        b.cmd = c
        obs["cmd"] = c.astype(np.float32)
        with torch.no_grad():
            a, _ = ctrl.predict(norm(obs), deterministic=True)
        obs, r, te, tr, info = env.step(a)
        rel = d.xpos[base] - d.geom_xpos[g]
        lean_y = float(np.arctan2(rel[1], rel[2]))                # body-over-ball lean toward +y
        up = d.xmat[base].reshape(3, 3)[:, 2]
        tilt = float(np.arctan2(up[1], up[2]))                     # body tilt toward +y (model theta, plane y)
        bd = m.joint("base_free_joint").dofadr[0]
        Rb = d.xmat[base].reshape(3, 3)
        thd = float(-(Rb @ d.qvel[bd + 3:bd + 6])[0])              # tilt rate toward +y = -omega_x (world)
        kq = m.joint("ball_free_joint").qposadr[0]
        Rk = d.xmat[m.body("ball").id].reshape(3, 3)
        phd = float(-(Rk @ d.qvel[kd + 3:kd + 6])[0])              # ball rolling rate toward +y = -omega_x
        rec.append((t * 0.01, float(d.geom_xpos[g][1]), float(d.qvel[kd + 1]), lean_y, tilt, thd, phd))
        if te:
            break
    return np.array(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["lqr", "trk_g_s0", "trk_h_s0"])
    ap.add_argument("--config", default="trk_d_if")
    a = ap.parse_args()
    M = json.load(open(ROOT / "outputs/study/results/model.json"))
    kappa = M["lean_per_accel"]
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{a.config}.yaml"))
    cfg["camera"]["enabled"] = False
    cfg["env"]["cmd_resample_steps"] = 10 ** 9
    cfg["env"]["max_ep_steps"] = 10 ** 6
    cfg["env"].pop("dr", None)
    cfg["env"]["init_yaw_random"] = False
    out, traces = {}, {}
    for run in a.runs:
        env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                               env_config={k: cfg[k] for k in ("camera", "env", "logging")}, seed=0, eval_env=True)()
        ctrl, norm = load_controller(run, env)
        step = rollout(env, ctrl, norm, lambda t: (0, 0.5, 0) if t >= 50 else (0, 0, 0), 350)
        y0 = step[49, 1] if len(step) > 50 else step[0, 1]
        early = step[50:110]
        nmp = float(early[:, 1].min() - y0) if len(early) else np.nan          # < 0: ball moved backward first
        # lean-acceleration regression over step + reversal + stop manoeuvres
        segs = [step, rollout(env, ctrl, norm, lambda t: (0, 0.5, 0) if t < 200 else (0, -0.5, 0), 450),
                rollout(env, ctrl, norm, lambda t: (0, 0.5, 0) if t < 250 else (0, 0, 0), 450)]
        # exact identity of the planar model (motor torque and wheel damping cancel when the two Lagrange
        # equations are added):  (M11 + M12) phidd + (M12 + M22) thetadd = m_a g l sin(theta)
        P = M["params"]
        M11 = P["I_k"] + P["J_eff"] + (P["m_a"] + P["m_k"]) * P["r"] ** 2
        M12 = -P["J_eff"] + P["l"] * P["m_a"] * P["r"]
        M22 = P["I_a"] + P["J_eff"] + P["l"] ** 2 * P["m_a"]
        lhs, rhs = [], []
        for sgm in segs:
            if len(sgm) < 30:
                continue
            k = np.ones(7) / 7
            phdd = np.gradient(np.convolve(sgm[:, 6], k, mode="same"), 0.01)
            thdd = np.gradient(np.convolve(sgm[:, 5], k, mode="same"), 0.01)
            sel = slice(10, len(sgm) - 10)
            lhs.append(((M11 + M12) * phdd + (M12 + M22) * thdd)[sel])
            rhs.append((P["m_a"] * 9.81 * P["l"] * np.sin(sgm[:, 4]))[sel])
        lhs, rhs = np.concatenate(lhs), np.concatenate(rhs)
        id_slope = float(np.polyfit(rhs, lhs, 1)[0])
        id_r2 = float(np.corrcoef(rhs, lhs)[0, 1] ** 2)
        L, A = [], []
        for sgm in segs:
            if len(sgm) < 30:
                continue
            v = np.convolve(sgm[:, 6] * P["r"], np.ones(9) / 9, mode="same")    # rolling speed r * phidot
            acc = np.gradient(v, 0.01)
            lean = np.convolve(sgm[:, 4], np.ones(9) / 9, mode="same")
            thdd = np.gradient(np.convolve(sgm[:, 5], np.ones(9) / 9, mode="same"), 0.01)
            # the law theta = kappa a holds when the tilt is not accelerating: keep quasi-static samples only
            qs = np.abs(thdd) < 0.5
            qs[:10] = qs[-10:] = False
            L.append(lean[qs]); A.append(acc[qs])
        L, A = np.concatenate(L), np.concatenate(A)
        slope = float(np.polyfit(A, L, 1)[0])
        r2 = float(np.corrcoef(A, L)[0, 1] ** 2)
        out[run] = dict(nmp_reverse_m=nmp, lean_slope=slope, kappa=kappa, r2=r2, n=int(len(L)),
                        identity_slope=id_slope, identity_r2=id_r2,
                        max_lean_deg=float(np.degrees(np.abs(L).max())), max_acc=float(np.abs(A).max()))
        traces[run] = dict(step=step.tolist(), lean=L[::5].tolist(), acc=A[::5].tolist(),
                           id_lhs=lhs[::5].tolist(), id_rhs=rhs[::5].tolist())
        print(f"{run:12s} initial reverse ball motion {nmp * 100:+.1f} cm | lean slope {slope:.3f} rad/(m/s^2) "
              f"vs kappa {kappa:.3f} (R^2 {r2:.2f}) | max lean {out[run]['max_lean_deg']:.1f} deg | "
              f"momentum identity slope {id_slope:.2f} (ideal 1) R^2 {id_r2:.2f}")
    json.dump(dict(results=out, traces=traces), open(ROOT / "outputs/study/results/theory_checks.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    C = {"lqr": "#2a78d6", "trk_g_s0": "#eb6834", "trk_h_s0": "#1baf7a"}
    lab = {"lqr": "LQR (derived model)", "trk_g_s0": "PPO trk_g", "trk_h_s0": "PPO trk_h"}
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3))
    for run, tr in traces.items():
        s = np.array(tr["step"])
        if len(s) < 60:
            continue
        t = s[:, 0] - 0.5
        ax[0].plot(t, (s[:, 1] - s[49, 1]) * 100, color=C.get(run, "#6250d6"), lw=1.5, label=lab.get(run, run))
        ax[1].plot(t, np.degrees(s[:, 4]), color=C.get(run, "#6250d6"), lw=1.5)
        ax[2].scatter(tr["acc"], np.degrees(tr["lean"]), s=3, color=C.get(run, "#6250d6"), alpha=0.35, linewidths=0)
    xs = np.linspace(-1.5, 1.5, 10)
    ax[2].plot(xs, np.degrees(kappa * xs), color="#333333", lw=1.2, ls="--", label=r"theory $\theta=\kappa a$")
    ax[0].set_xlim(-0.2, 1.2); ax[0].set_ylim(-2, 25)
    ax[0].set_xlabel("time after step command [s]"); ax[0].set_ylabel("ball displacement [cm]")
    ax[0].axhline(0, color="#9a9a9a", lw=0.8)
    ax[1].set_xlim(-0.2, 2.5); ax[1].set_xlabel("time after step command [s]"); ax[1].set_ylabel("lean toward motion [deg]")
    ax[2].set_xlim(-1.5, 1.5); ax[2].set_ylim(-15, 15)
    ax[2].set_xlabel(r"ball acceleration [m/s$^2$]"); ax[2].set_ylabel("lean [deg]")
    ax[0].legend(frameon=False, fontsize=6); ax[2].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    (ROOT / "paper_v2/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "paper_v2/figures/fig_theory.pdf")
    fig.savefig(ROOT / "outputs/study/results/fig_theory.png", dpi=200)


if __name__ == "__main__":
    main()
