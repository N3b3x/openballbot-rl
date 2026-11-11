"""Step-edge benchmark: drive straight (+y) at a commanded speed toward a raised floor that starts at y = 1.0 m
(clean step with a 2 cm bevel), heights 0-3 cm. Success = ball centre past y = 1.5 m within 8 s, then stop and hold balance on top for 2 s.

  python scripts/study/eval_step.py --runs lqr res_c_s0 "robs_c_s0@ki=1.0" --configs trk_d_if res_c robs_c
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

HEIGHTS = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)


def body_yaw(b):
    q = b.data.qpos[b.model.joint('base_free_joint').qposadr[0] + 3:][:4]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.4, 0.7])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--fric-scale", type=float, default=1.0, help="scale the wheel-ball cross-roller friction (traction study)")
    ap.add_argument("--init-yaws", type=float, nargs="+", default=None, help="fixed start headings [deg] (else random); trials are spread over these")
    ap.add_argument("--hold-yaw", type=float, default=None, help="align the body to this heading [deg] for 1.5 s before driving and hold it (P control on the yaw-rate command)")
    ap.add_argument("--heights", type=float, nargs="+", default=list(HEIGHTS))
    ap.add_argument("--out", default=str(ROOT / "outputs/study/results/step.csv"))
    a = ap.parse_args()
    rows = []
    for run, conf in zip(a.runs, a.configs):
        for h in a.heights:
            cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{conf}.yaml"))
            cfg["camera"]["enabled"] = False
            cfg["env"].update(cmd_resample_steps=10 ** 9, max_ep_steps=10 ** 6, init_yaw_random=a.init_yaws is None)  # varied trials
            cfg["env"].pop("dr", None)
            cfg["terrain"] = {"type": "book", "config": {"step_y": 1.0, "book_len": 0.0, "rough_amplitude": 0.0,
                                                         "step_height_m": float(h), "ramp_len": 0.02, "seed": 0}}
            env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                                   env_config={k: cfg[k] for k in ("camera", "env", "logging")}, seed=0, eval_env=True)()
            ctrl, norm = load_controller(run, env)
            b = env.unwrapped
            b.model.pair_friction[:, 1] *= a.fric_scale
            g = b.model.geom("the_ball").id
            for sp in a.speeds:
                ok = fell = 0
                for k in range(a.trials):
                    cmd = np.array([0.0, sp, 0.0])
                    b.cmd_fixed = cmd
                    obs, _ = env.reset(seed=k)
                    if a.init_yaws is not None:
                        y0 = np.radians(a.init_yaws[k % len(a.init_yaws)])
                        bq = b.model.joint('base_free_joint').qposadr[0]
                        b.data.qpos[bq + 3:bq + 7] = [np.cos(y0 / 2), 0, 0, np.sin(y0 / 2)]
                        import mujoco
                        mujoco.mj_forward(b.model, b.data)
                        obs = b._get_obs(np.zeros(3, dtype=np.float32))
                    if a.hold_yaw is not None:
                        tgt = np.radians(a.hold_yaw)
                        for _ in range(150):      # align first, standing still
                            e = np.arctan2(np.sin(tgt - body_yaw(b)), np.cos(tgt - body_yaw(b)))
                            c0 = np.array([0.0, 0.0, float(np.clip(2.0 * e, -1, 1))])
                            b.cmd = c0
                            obs['cmd'] = c0.astype(np.float32)
                            with torch.no_grad():
                                act, _ = ctrl.predict(norm(obs), deterministic=True)
                            obs, r, te, tr, info = env.step(act)
                            if te:
                                break
                        if te:
                            fell += 1
                            print(f'  trial {k} yaw {a.init_yaws[k % len(a.init_yaws)] if a.init_yaws else -1:.0f} -> fell while aligning', flush=True)
                            continue
                    for t in range(800):
                        b.cmd = cmd.copy()
                        if a.hold_yaw is not None:
                            e = np.arctan2(np.sin(tgt - body_yaw(b)), np.cos(tgt - body_yaw(b)))
                            b.cmd[2] = float(np.clip(2.0 * e, -1, 1))
                        cmd_t = b.cmd.copy()
                        obs["cmd"] = (cmd_t if a.hold_yaw is not None else cmd).astype(np.float32)
                        with torch.no_grad():
                            act, _ = ctrl.predict(norm(obs), deterministic=True)
                        obs, r, te, tr, info = env.step(act)
                        if te:
                            fell += 1
                            print(f'  trial {k} yaw {a.init_yaws[k % len(a.init_yaws)] if a.init_yaws else -1:.0f} -> fell before crossing', flush=True)
                            break
                        if b.data.geom_xpos[g][1] > 1.5:
                            # crossed: now stop on top and hold balance for 2 s
                            stop = np.zeros(3)
                            held = True
                            for _ in range(200):
                                b.cmd = stop
                                obs["cmd"] = stop.astype(np.float32)
                                with torch.no_grad():
                                    act, _ = ctrl.predict(norm(obs), deterministic=True)
                                obs, r, te, tr, info = env.step(act)
                                if te:
                                    held = False
                                    break
                            ok += held
                            fell += not held
                            print(f'  trial {k} yaw {a.init_yaws[k % len(a.init_yaws)] if a.init_yaws else -1:.0f} -> {"ok" if held else "fell on top"}', flush=True)
                            break
                rows.append(dict(run=run, height_cm=h * 100, speed=sp, success=ok, falls=fell, trials=a.trials))
                print(f"{run:20s} h={h * 100:.1f} cm v={sp}: success {ok}/{a.trials}, falls {fell}", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.run.isin(a.runs)], df])
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
