"""Evaluate trained PPO policies on held-out terrains and perception corruptions.

Held-out terrains use Perlin seeds >= 100000 (training seeds are < 10000) and fixed
difficulty levels (Perlin amplitude). Episodes run in parallel worker processes.

Metrics per episode: success (reached goal distance, no fall), fall, return,
steps/time-to-goal, mean speed along the target direction.
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROPRIO_KEYS = ["orientation", "angular_vel", "vel", "motor_state", "actions"]

# perception corruption conditions for H2
CONDITIONS = {
    "clean": {},
    "noise_0.05": {"depth_noise_std": 0.05},
    "noise_0.15": {"depth_noise_std": 0.15},
    "noise_0.30": {"depth_noise_std": 0.30},
    "dropout_0.2": {"depth_dropout": 0.2},
    "occlusion_0.25": {"depth_occlusion": 0.25},
    "occlusion_0.50": {"depth_occlusion": 0.50},
    "blank": {"depth_dropout": 1.0},  # all pixels invalid (sanity check of reliance on depth)
}


def _load_policy(run_dir, variant, ckpt):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_ppo import make_policy_kwargs
    run_dir = Path(run_dir)
    zip_path = run_dir / (ckpt + ".zip")
    # re-supply policy_kwargs so a repo-relative encoder path is used (portable across machines)
    model = PPO.load(zip_path, device="cpu",
                     custom_objects={"policy_kwargs": make_policy_kwargs(variant)})
    vn_path = run_dir / ("vecnormalize.pkl" if ckpt == "final_model" else ckpt.replace("ppo_", "ppo_vecnormalize_") + ".pkl")
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    return model, vn


def _normalize(obs, vn):
    out = dict(obs)
    for k in PROPRIO_KEYS:
        rms = vn.obs_rms[k]
        out[k] = np.clip((obs[k] - rms.mean) / np.sqrt(rms.var + vn.epsilon), -vn.clip_obs, vn.clip_obs).astype(np.float32)
    return out


def run_job(job):
    """Run a list of (seed, amplitude) episodes for one policy / condition."""
    import torch
    from ballbot_rl.training.utils import make_ballbot_env
    import ballbot_gym  # noqa: F401
    cfg = job["env_cfg"]
    job.setdefault("goal_task", bool(cfg["env"].get("goal_mode")))
    job.setdefault("eval_goal", 1.0)
    model, vn = _load_policy(job["run_dir"], job["variant"], job["ckpt"]) if job["run_dir"] else (None, None)
    env_config = {k: cfg[k] for k in ("camera", "env", "logging")}
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                           env_config=env_config, seed=0, eval_env=True)()
    base = env.unwrapped
    rows = []
    for (seed, amp) in job["episodes"]:
        tcfg = base.terrain_config["config"]
        tcfg["seed"] = int(seed)
        if base.terrain_type == "perlin":
            tcfg["amplitude"] = float(amp)
            tcfg.pop("amplitude_range", None)
        obs, info = env.reset(seed=int(seed))
        ret, steps, t_goal, fell, success = 0.0, 0, None, False, False
        d0 = float(np.linalg.norm(base.goal_xy)) if job["goal_task"] else 0.0
        min_gd = d0
        while True:
            if model is None:  # open-loop constant action baseline (never used for claims)
                action = np.zeros(3, dtype=np.float32)
            else:
                with torch.no_grad():
                    action, _ = model.predict(_normalize(obs, vn), deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            steps += 1
            if job["goal_task"]:
                min_gd = min(min_gd, info.get("goal_dist", min_gd))
            if (not job["goal_task"]) and t_goal is None and info["progress"] >= job["eval_goal"]:
                t_goal = steps  # first step reaching the evaluation goal distance
            if term or trunc:
                fell = bool(info.get("failure", False))
                if job["goal_task"]:
                    success = bool(info.get("success", False))
                    t_goal = steps if success else None
                else:
                    success = (t_goal is not None) and not fell
                break
        rows.append(dict(seed=seed, amplitude=amp, condition=job["condition"], variant=job["variant"],
                         run=job["run_name"], ret=ret, steps=steps, fell=fell, success=success,
                         time_to_goal=t_goal, progress=(d0 - min_gd) if job["goal_task"] else float(info["progress"]),
                         goal_dist0=d0, min_goal_dist=min_gd,
                         speed=float(info["progress"]) / (steps * 0.01 + 1e-9)))
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run_dir:variant pairs")
    ap.add_argument("--env-config", default="configs/env/perlin_directional_train.yaml")
    ap.add_argument("--amplitudes", nargs="+", type=float, default=[0.04, 0.08, 0.12, 0.18])
    ap.add_argument("--episodes", type=int, default=30, help="episodes per (run, condition, amplitude)")
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--ckpt", default="final_model")
    ap.add_argument("--eval-goal", type=float, default=1.0, help="progress [m] defining success (directional task)")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    jobs = []
    for spec in args.runs:
        run_dir, variant = spec.split(":")
        cfg = yaml.safe_load(open(args.env_config))
        cfg["camera"]["enabled"] = variant != "proprio"
        for cond in args.conditions:
            if variant == "proprio" and cond != "clean":
                continue  # a proprio-only policy has no depth input to corrupt
            ccfg = json.loads(json.dumps(cfg))
            ccfg["camera"].update(CONDITIONS[cond])
            for amp in args.amplitudes:
                eps = [(100000 + 7 * i, amp) for i in range(args.episodes)]
                for chunk in np.array_split(np.arange(len(eps)), max(1, min(len(eps), 4))):
                    jobs.append(dict(eval_goal=args.eval_goal, run_dir=run_dir, variant=variant, ckpt=args.ckpt, condition=cond,
                                     env_cfg=ccfg, run_name=Path(run_dir).name,
                                     episodes=[eps[i] for i in chunk]))
    print(f"{len(jobs)} jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(run_job, jobs)):
            rows.extend(res)
            if i % 10 == 0:
                print(f"  {i + 1}/{len(jobs)} jobs done", flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["variant", "condition", "amplitude"])[["success", "fell", "ret"]].mean().round(3))


if __name__ == "__main__":
    main()
