"""Evaluate one controller on a list of ballbot designs (Froude-scaled protocol, nondimensional where possible).

Per design d with scale s:  speed grid v = {0.4, 0.8, 1.1} * sqrt(s) in 4 headings (random start yaw), 4 s each ->
ratio achieved/commanded speed + falls;  push test: one 50 ms horizontal push of alpha * m_total * g, alpha in {0.5, 1, 1.5},
random direction, while driving at 0.4 sqrt(s) -> falls / trials.

  python scripts/study/eval_design.py --run lqr --designs nom s0.5 --mode lqr --out outputs/study/results/dz_lqr.csv
  python scripts/study/eval_design.py --run dz_nom_s0 --designs nom s0.5 --mode ppo      # single-design PPO policy
  python scripts/study/eval_design.py --run hyb_pool_s0 --designs nom s0.5 --mode hybrid # morphology-conditioned hybrid
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "study"))
from policy_io import load_controller  # noqa: E402
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import ballbot_gym  # noqa: F401,E402

SPEEDS = (0.4, 0.8, 1.1)
ALPHAS = (0.5, 1.0, 1.5)


def make_env(design, mode):
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"dz_{design}.yaml"))
    cfg["camera"]["enabled"] = False
    e = cfg["env"]
    e.update(cmd_resample_steps=10 ** 9, max_ep_steps=10 ** 6, init_yaw_random=True)
    e.pop("dr", None)
    mj = json.load(open(e["lqr_model_json"]))
    if mode in ("hybrid", "residual_ctx", "ppo_ctx"):
        e["morph_ctx"] = mj["ctx"]
    if mode == "hybrid":
        e["hybrid_gate"] = True
    if mode == "residual_ctx":
        e["residual_lqr"], e["residual_scale"] = True, 0.1
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                           env_config={k: cfg[k] for k in ("camera", "env", "logging")}, seed=0, eval_env=True)()
    m = mj["params"]
    return env, float(m["m_a"] + m["m_k"]), float(np.sqrt(m["r"] / 0.09))


def run_cmd(env, ctrl, norm, cmd, steps, push=None):
    b = env.unwrapped
    base = b.model.body("base").id
    b.cmd_fixed = cmd
    obs, _ = env.reset(seed=int(np.random.randint(1 << 30)))
    pos = []
    for t in range(steps):
        b.cmd = cmd
        obs["cmd"] = cmd.astype(np.float32)
        with torch.no_grad():
            act, _ = ctrl.predict(norm(obs), deterministic=True)
        if push is not None and 200 <= t < 205:
            b.data.xfrc_applied[base, :3] = push
        obs, r, te, tr, info = env.step(act)
        if te:
            return True, None
        pos.append(np.array(obs["vel"][:2], dtype=float))
    return False, np.array(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--mode", choices=["lqr", "ppo", "ppo_ctx", "hybrid", "residual_ctx"], default="ppo")
    ap.add_argument("--tag", default="")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "outputs/study/results/design_eval.csv"))
    a = ap.parse_args()
    rows = []
    for d in a.designs:
        env, m_tot, f = make_env(d, a.mode)
        ctrl, norm = load_controller(a.run, env)
        rng = np.random.default_rng(0)
        row = dict(run=a.tag or a.run, design=d, mode=a.mode)
        for sp in SPEEDS:
            ratios, falls = [], 0
            for k in range(4):
                ang = k * np.pi / 2
                cmd = sp * f * np.array([np.cos(ang), np.sin(ang), 0.0])
                fell, v = run_cmd(env, ctrl, norm, cmd, 400)
                falls += fell
                if not fell:
                    vm = v[-150:].mean(0)
                    ratios.append(float(np.dot(vm, cmd[:2]) / np.dot(cmd[:2], cmd[:2])))
            row[f"ratio_{sp}"] = float(np.mean(ratios)) if ratios else np.nan
            row[f"falls_{sp}"] = falls
        for al in ALPHAS:
            falls = 0
            for k in range(a.trials):
                phi = rng.uniform(0, 2 * np.pi)
                F = al * m_tot * 9.81 * np.array([np.cos(phi), np.sin(phi), 0.0])
                fell, _ = run_cmd(env, ctrl, norm, np.array([0.0, 0.4 * f, 0.0]), 500, push=F)
                falls += fell
            row[f"pushfall_{al}"] = falls / a.trials
        rows.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    if out.exists():
        old = pd.read_csv(out)
        key = set(zip(df.run, df.design))
        old = old[[(r, dd) not in key for r, dd in zip(old.run, old.design)]]
        df = pd.concat([old, df])
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
