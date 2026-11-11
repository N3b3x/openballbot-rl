"""Flat-ground control benchmark for velocity-command policies (ctrl_*).

Tests (each 4 s, deterministic policy, 3 repeats):
  speed   : 8 directions x speeds up to the config's cmd_vmax          -> achieved speed, error, falls
  yaw     : spin in place at several rates, both signs                 -> achieved yaw rate, position drift
  step    : rest -> 0.5 m/s forward, time to 90 % of the command        -> rise time
  reverse : +y 0.5 m/s then -y 0.5 m/s (switch at 2 s)                 -> reversal time
  arc     : drive 0.5 m/s while turning at 1 rad/s                     -> speed / yaw error
Writes a CSV and prints a summary. Usage:
  python scripts/study/eval_control.py --run ctrl_base_s0 --config ctrl_base --out outputs/study/results/control_base.csv
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


def rollout(env, model, norm, cmd_fn, seed, steps=400):
    base = env.unwrapped
    base.cmd_fixed = cmd_fn(0)
    obs, info = env.reset(seed=seed)
    rows, fell = [], False
    for t in range(steps):
        base.cmd = np.asarray(cmd_fn(t), dtype=float)
        obs["cmd"] = base.cmd.astype(np.float32)
        with torch.no_grad():
            act, _ = model.predict(norm(obs), deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        # cmd_fixed resampling is disabled by overriding cmd each step; keep info consistent
        v = obs["vel"][:2]
        rows.append((t, float(v[0]), float(v[1]), float(obs["angular_vel"][2]), float(info["pos2d"][0]), float(info["pos2d"][1]), float(info.get("act_rate", 0.0)), float(info.get("off_ball_deg", 0.0))))
        if term:
            fell = True
            break
    return np.array(rows), fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="final_model")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--perlin-amp", type=float, default=0.0, help="evaluate on Perlin-noise terrain of this amplitude [m]")
    ap.add_argument("--dr", action="store_true", help="keep the config's domain randomisation / pushes")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{a.config}.yaml"))
    vmax, wmax = cfg["env"]["cmd_vmax"], cfg["env"]["cmd_wmax"]
    cfg["env"]["cmd_resample_steps"] = 10 ** 9
    if not a.dr:
        cfg["env"].pop("dr", None)   # benchmark the nominal robot unless asked otherwise
    if a.perlin_amp > 0:
        cfg["terrain"] = {"type": "perlin", "config": {"scale": 25.0, "octaves": 4, "persistence": 0.2,
                                                       "lacunarity": 2.0, "amplitude": a.perlin_amp, "seed": None}}
    ec = {k: cfg[k] for k in ("camera", "env", "logging")}
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config=ec, seed=0, eval_env=True)()
    model, norm = load_controller(a.run, env, a.ckpt)

    out = []

    def record(test, label, cmdv, rows, fell, extra=None):
        n = len(rows)
        late = rows[min(n // 2, n - 1):] if n else rows
        d = dict(test=test, label=label, cmd_x=cmdv[0], cmd_y=cmdv[1], cmd_w=cmdv[2], fell=fell, steps=n)
        if n:
            d.update(vx=late[:, 1].mean(), vy=late[:, 2].mean(), wz=late[:, 3].mean(),
                     drift=float(np.linalg.norm(rows[-1, 4:6] - rows[0, 4:6])), act_rate=float(rows[:, 6].mean()), off_ball=float(rows[:, 7].max()))
        d.update(extra or {})
        out.append(d)

    # speed grid
    speeds = [s for s in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0) if s <= vmax + 1e-9]
    for deg in range(0, 360, 45):
        for sp in speeds:
            cmd = (sp * np.sin(np.radians(deg)), sp * np.cos(np.radians(deg)), 0.0)  # deg 0 = +y
            for rep in range(a.repeats):
                rows, fell = rollout(env, model, norm, lambda t, c=cmd: c, seed=rep)
                record("speed", f"dir{deg}_v{sp}", cmd, rows, fell, dict(dir_deg=deg, speed_cmd=sp, rep=rep))
    # yaw
    for w in [x for x in (1.0, 1.5, 2.0, 3.0) if x <= wmax + 1e-9]:
        for sgn in (1, -1):
            cmd = (0.0, 0.0, sgn * w)
            for rep in range(a.repeats):
                rows, fell = rollout(env, model, norm, lambda t, c=cmd: c, seed=rep)
                record("yaw", f"w{sgn * w}", cmd, rows, fell, dict(rep=rep))
    # step response + reversal + arc
    for rep in range(a.repeats):
        cmd = (0.0, 0.5, 0.0)
        rows, fell = rollout(env, model, norm, lambda t: cmd if t >= 50 else (0, 0, 0), seed=rep)
        vy = rows[:, 2]
        hit = np.where(vy[50:] >= 0.45)[0]
        record("step", "rest_to_0.5", cmd, rows, fell, dict(rep=rep, rise_s=float(hit[0] * 0.01) if len(hit) else np.nan))
        rows, fell = rollout(env, model, norm, lambda t: (0, 0.5, 0) if t < 200 else (0, -0.5, 0), seed=rep)
        vy = rows[:, 2]
        hit = np.where(vy[200:] <= -0.45)[0]
        record("reverse", "+0.5_to_-0.5", (0, -0.5, 0), rows, fell, dict(rep=rep, reverse_s=float(hit[0] * 0.01) if len(hit) else np.nan))
        cmd = (0.0, 0.5, 1.0)
        rows, fell = rollout(env, model, norm, lambda t: cmd, seed=rep)
        record("arc", "v0.5_w1.0", cmd, rows, fell, dict(rep=rep))

    # heading robustness: same 0.5 m/s world-frame commands from 6 random starting headings
    env.unwrapped.init_yaw_random = True
    for deg in (0, 90, 180, 270):
        cmd = (0.5 * np.sin(np.radians(deg)), 0.5 * np.cos(np.radians(deg)), 0.0)
        for rep in range(6):
            rows, fell = rollout(env, model, norm, lambda t, c=cmd: c, seed=500 + rep)
            record("heading", f"dir{deg}_h{rep}", cmd, rows, fell, dict(dir_deg=deg, speed_cmd=0.5, rep=rep))
    env.unwrapped.init_yaw_random = False

    df = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    s = df[df.test == "speed"].copy()
    s["speed_ach"] = np.hypot(s.vx, s.vy)
    g = s.groupby("speed_cmd").agg(speed_ach=("speed_ach", "mean"), fall=("fell", "mean"))
    print("speed (mean over directions):\n", g.round(2))
    print("smoothness (mean act_rate, max off-ball deg):", round(df.act_rate.mean(), 4), round(df.off_ball.max(), 1))
    y = df[df.test == "yaw"].groupby("cmd_w").agg(wz=("wz", "mean"), fall=("fell", "mean"), drift=("drift", "mean"))
    print("yaw:\n", y.round(2))
    h = df[df.test == "heading"]
    if len(h):
        print("heading test (0.5 m/s, random start yaw): vel err", round(float(np.hypot(h.vx - h.cmd_x, h.vy - h.cmd_y).mean()), 3), "falls", round(h.fell.mean(), 2))
    for t in ("step", "reverse", "arc"):
        print(t, df[df.test == t][["fell"] + [c for c in ("rise_s", "reverse_s", "vx", "vy", "wz") if c in df.columns]].mean(numeric_only=True).round(2).to_dict())


if __name__ == "__main__":
    main()
