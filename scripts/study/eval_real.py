"""Evaluate model-based and learned controllers under realistic sensing and actuation (simulation only).

  python scripts/study/eval_real.py --ctrl lqr smc mpc --design nom --sensor none mild hard --motor 0 1 --out outputs/study/results/real_base.csv
Controllers: lqr, smc, mpc (classical, with SensorModel), or a run name (PPO; obs noise set to the same level via env dr.obs_noise).
Sensor levels: see ballbot_gym/controllers/realistic.py LEVELS.  Motor 1 = first-order lag, torque-speed curve and Coulomb friction (env.motor).
Metrics: speed ratio at {0.4,0.8,1.1} sqrt(s) in 4 headings (falls counted), push falls at {0.5,1,1.5} x m g.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
from eval_design import make_env, run_cmd, SPEEDS, ALPHAS  # noqa: E402
from policy_io import load_controller  # noqa: E402
from ballbot_gym.controllers.lqr import LQRPolicy  # noqa: E402
from ballbot_gym.controllers.realistic import SensorModel, SMCPolicy, MPCPolicy, LEVELS  # noqa: E402

OBS_SD = {"none": 0.0, "mild": 0.01, "hard": 0.03}


OVER = {}


def build(name, env, level, seed):
    b = env.unwrapped
    mj = getattr(b, "_env_settings_lqr_json", None)
    kw = {"model_json": mj} if mj else {}
    d = OVER.get("delay", LEVELS[level]["delay"])
    if name in ("lqr", "smc", "mpc"):
        ctrl = {"lqr": lambda: LQRPolicy(b, **kw), "smc": lambda: SMCPolicy(b, **kw),
                "mpc": lambda: MPCPolicy(b, delay=d, **kw)}[name]()
        ctrl.sensor = SensorModel(level, seed=seed, **OVER)
        return ctrl, (lambda o: o)
    b.dr = {"obs_noise": OBS_SD[level]} if OBS_SD[level] else None
    return load_controller(name, env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", nargs="+", required=True)
    ap.add_argument("--design", default="nom")
    ap.add_argument("--sensor", nargs="+", default=["none"])
    ap.add_argument("--motor", nargs="+", type=int, default=[0])
    ap.add_argument("--mode", default="ppo", help="env mode for PPO runs (ppo, ppo_ctx)")
    ap.add_argument("--over", default="{}", help="JSON overrides of SensorModel params, e.g. '{\"delay\":0}'")
    ap.add_argument("--tag", default="")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "outputs/study/results/real_eval.csv"))
    a = ap.parse_args()
    import json
    OVER.update(json.loads(a.over))
    rows = []
    for mo in a.motor:
        design = a.design + ("_motor" if mo else "")
        for lev in a.sensor:
            for c in a.ctrl:
                env, m_tot, f = make_env(design, "lqr" if c in ("lqr", "smc", "mpc") else a.mode)
                ctrl, norm = build(c, env, lev, 0)
                rng = np.random.default_rng(0)
                row = dict(tag=a.tag, ctrl=c, design=a.design, sensor=lev, motor=mo)
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
                pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
