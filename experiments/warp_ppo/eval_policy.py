"""Evaluate a Warp-trained policy with the study protocol (eval_design.py) on the CPU engine or on Warp.
  python eval_policy.py --engine cpu  --run wp_nom_s0     (main python: mujoco 3.13 + ContactFrameStepper env)
  ..\mjx_port\.venv\Scripts\python.exe eval_policy.py --engine warp --run wp_nom_s0
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs" / "study" / "warp_ppo"
SPEEDS, ALPHAS = (0.4, 0.8, 1.1), (0.5, 1.0, 1.5)
KEYS = ["orientation", "angular_vel", "vel", "motor_state", "actions", "cmd", "heading", "vint"]


def load(run, dev):
    ck = torch.load(RUN / run / "model.pt", map_location=dev)
    mlp = lambda o: nn.Sequential(nn.Linear(22, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, o))
    pi = mlp(3).to(dev); pi.load_state_dict({k[3:]: v for k, v in ck["ac"].items() if k.startswith("pi.")})
    mean, var = ck["mean"].to(dev), ck["var"].to(dev)
    return lambda o: pi(((o - mean) / torch.sqrt(var + 1e-8)).clamp(-10, 10)).clamp(-1, 1), ck["steps"]


def eval_cpu(run, design="nom"):
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
    import eval_design as ed
    pol, steps = load(run, "cpu")
    env, m_tot, f = ed.make_env(design, "ppo")

    class C:
        def predict(self, o, deterministic=True):
            return o, None
    norm = lambda obs: torch.tensor(np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in KEYS]))[None]
    ctrl = type("P", (), {"predict": lambda self, x, deterministic=True: (pol(x)[0].numpy(), None)})()
    rng = np.random.default_rng(0)
    row = dict(engine="cpu")
    for sp in SPEEDS:
        ratios, falls = [], 0
        for k in range(4):
            ang = k * np.pi / 2
            cmd = sp * f * np.array([np.cos(ang), np.sin(ang), 0.0])
            fell, v = ed.run_cmd(env, ctrl, norm, cmd, 400)
            falls += fell
            if not fell:
                ratios.append(float(np.dot(v[-150:].mean(0), cmd[:2]) / np.dot(cmd[:2], cmd[:2])))
        row[f"ratio_{sp}"] = float(np.mean(ratios)) if ratios else np.nan; row[f"falls_{sp}"] = falls
    for al in ALPHAS:
        falls = 0
        for k in range(16):
            phi = rng.uniform(0, 2 * np.pi)
            F = al * m_tot * 9.81 * np.array([np.cos(phi), np.sin(phi), 0.0])
            fell, _ = ed.run_cmd(env, ctrl, norm, np.array([0.0, 0.4 * f, 0.0]), 500, push=F)
            falls += fell
        row[f"pushfall_{al}"] = falls / 16
    return row, steps


def eval_warp(run, design="nom", cfg=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train as T
    pol, steps = load(run, "cuda")
    D = T.Design(design); m_tot, f = D.m_total, D.f
    row = dict(engine="warp")
    for sp in SPEEDS:
        n = 4
        env = T.GpuBallbot(n, seed=1, dr=False, design=design, cfg_name=cfg)
        env.resample, env.max_steps = 10 ** 9, 10 ** 9
        env.reset(torch.ones(n, dtype=torch.bool, device="cuda")); env.steps[:] = 0
        ang = torch.arange(4, device="cuda") * np.pi / 2
        env.cmd = torch.stack([sp * f * torch.cos(ang), sp * f * torch.sin(ang), torch.zeros(4, device="cuda")], 1)
        env.reset_obs_cmd = True
        o = env.reset_obs(); fell_any = torch.zeros(n, dtype=torch.bool, device="cuda"); vs = []
        for t in range(400):
            with torch.no_grad():
                a = pol(o)
            o, r, fell, trunc, info = env.step(a)
            fell_any |= fell
            vs.append(env._vel[:, :2].clone())
        v = torch.stack(vs)[-150:].mean(0)
        ratio = ((v * env.cmd[:, :2]).sum(1) / (env.cmd[:, :2] ** 2).sum(1))
        ok = ~fell_any
        row[f"ratio_{sp}"] = float(ratio[ok].mean()) if ok.any() else np.nan; row[f"falls_{sp}"] = int(fell_any.sum())
        del env
    for al in ALPHAS:
        n = 16
        env = T.GpuBallbot(n, seed=2, dr=False, design=design, cfg_name=cfg)
        env.resample, env.max_steps = 10 ** 9, 10 ** 9
        env.reset(torch.ones(n, dtype=torch.bool, device="cuda")); env.steps[:] = 0
        env.cmd = torch.tensor([[0.0, 0.4 * f, 0.0]], device="cuda").repeat(n, 1)
        phi = torch.arange(n, device="cuda") * 2 * np.pi / n
        env.ext_push = al * m_tot * 9.81 * torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], 1)
        env.ext_on = False
        o = env.reset_obs(); fell_any = torch.zeros(n, dtype=torch.bool, device="cuda")
        for t in range(500):
            env.ext_on = 200 <= t < 205
            with torch.no_grad():
                a = pol(o)
            o, r, fell, trunc, info = env.step(a)
            fell_any |= fell
        row[f"pushfall_{al}"] = float(fell_any.float().mean())
        del env
    return row, steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["cpu", "warp"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--design", default="nom")
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    row, steps = eval_cpu(a.run, a.cfg or a.design) if a.engine == "cpu" else eval_warp(a.run, a.design, a.cfg)
    row.update(run=a.run, design=a.design, train_steps=steps)
    print(json.dumps(row))
    with open(RUN / a.run / f"eval_{a.engine}.json", "w") as f:
        json.dump(row, f, indent=1)
