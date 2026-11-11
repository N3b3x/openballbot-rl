"""Push-recovery benchmark: drive at 0.5 m/s (+y), apply one 50 ms horizontal push of fixed magnitude from a
random direction at t = 2 s, count falls in the following 3 s. Optional parameter randomisation (mass, friction,
damping, torque, latency, noise) without random pushes.

  python scripts/study/eval_push.py --runs lqr trk_g_s0 rob_a_s0 --config trk_d_if [--dr-params]
"""
import argparse
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

FORCES = (0, 10, 20, 30, 40, 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config", default="trk_d_if")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--dr-params", action="store_true", help="randomise physical parameters (no random pushes)")
    ap.add_argument("--dr-key", default="", help="randomise only this parameter (mass, friction, damping, torque, latency_p, obs_noise)")
    ap.add_argument("--forces", type=float, nargs="+", default=list(FORCES))
    ap.add_argument("--out", default=str(ROOT / "outputs/study/results/push.csv"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{a.config}.yaml"))
    cfg["camera"]["enabled"] = False
    cfg["env"]["cmd_resample_steps"] = 10 ** 9
    cfg["env"]["max_ep_steps"] = 10 ** 6
    cfg["env"]["init_yaw_random"] = False
    cfg["env"].pop("dr", None)
    if a.dr_params:
        dr = dict(yaml.safe_load(open(ROOT / "configs/env/rob_a.yaml"))["env"]["dr"])
        dr.pop("push_force", None)
        if a.dr_key:
            dr = {k: v for k, v in dr.items() if k in (a.dr_key, "push_every", "push_dur")}
        cfg["env"]["dr"] = dr
    rows = []
    for run in a.runs:
        env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                               env_config={k: cfg[k] for k in ("camera", "env", "logging")}, seed=0, eval_env=True)()
        ctrl, norm = load_controller(run, env)
        b = env.unwrapped
        base = b.model.body("base").id
        rng = np.random.default_rng(0)
        for F in a.forces:
            falls = 0
            for k in range(a.trials):
                cmd = np.array([0.0, 0.5, 0.0])
                b.cmd_fixed = cmd
                obs, _ = env.reset(seed=1000 + k)
                ang = rng.uniform(0, 2 * np.pi)
                fvec = F * np.array([np.cos(ang), np.sin(ang), 0.0])
                fell = False
                for t in range(500):
                    b.cmd = cmd
                    obs["cmd"] = cmd.astype(np.float32)
                    with torch.no_grad():
                        act, _ = ctrl.predict(norm(obs), deterministic=True)
                    if 200 <= t < 205:     # 50 ms push (applied for the next control step, cleared after)
                        b.data.xfrc_applied[base, :3] = fvec
                    obs, r, te, tr, info = env.step(act)
                    if te:
                        fell = True         # any fall counts (before or after the push)
                        break
                falls += fell
            rows.append(dict(run=run, force=F, falls=falls, trials=a.trials, dr_params=a.dr_params, dr_key=a.dr_key))
            print(f"{run:12s} {a.dr_key or 'all'} push {F:2.0f} N: falls {falls}/{a.trials}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~(old.run.isin(a.runs) & (old.dr_params == a.dr_params))]
        df = pd.concat([old, df])
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
