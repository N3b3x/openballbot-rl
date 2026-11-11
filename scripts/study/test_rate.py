"""Control-rate experiment: LQR on a design at several control rates (action_repeat of the 500 Hz physics).
  python scripts/study/test_rate.py --design nom --repeats 5 2 1 --speeds 0.5 1.0 1.3
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, torch, yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
from policy_io import load_controller
from ballbot_rl.training.utils import make_ballbot_env
import ballbot_gym  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--design", default="nom"); ap.add_argument("--repeats", type=int, nargs="+", default=[5, 2, 1])
ap.add_argument("--speeds", type=float, nargs="+", default=[0.5, 1.0, 1.3]); ap.add_argument("--run", default="lqr@d")
ap.add_argument("--fric", type=float, nargs="+", default=[1.0]); ap.add_argument("--roll-fric", type=float, default=1.0); ap.add_argument("--T", type=float, default=4.0); ap.add_argument("--trials", type=int, default=4)
a = ap.parse_args()
for ar, fs in [(r, fs) for r in a.repeats for fs in a.fric]:
    cfg = yaml.safe_load(open(ROOT / f"configs/env/dz_{a.design}.yaml")); cfg["camera"]["enabled"] = False
    e = cfg["env"]; e.update(action_repeat=ar, cmd_resample_steps=10**9, max_ep_steps=10**7, init_yaw_random=True); e.pop("dr", None)
    f = np.sqrt(json.load(open(e["lqr_model_json"]))["params"]["r"] / 0.09)
    env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config={k: cfg[k] for k in ("camera", "env", "logging")}, seed=0, eval_env=True)()
    b = env.unwrapped
    b.model.pair_friction[:, 1] *= fs; b.model.pair_friction[:, 0] *= a.roll_fric
    run = a.run
    ctrl, norm = load_controller(run, env)
    if hasattr(ctrl, "dt") is False and hasattr(ctrl, "inner"):
        pass
    if run.startswith("lqr@d"):   # discrete LQR at the actual period
        from ballbot_gym.controllers.lqr import LQRPolicy
        ctrl = LQRPolicy(b, model_json=e["lqr_model_json"], discrete_dt=0.002 * ar)
    dt = 0.002 * ar; n = int(a.T / dt)
    out = []
    for sp in a.speeds:
        ratios, falls = [], 0
        for k in range(a.trials):
            cmd = sp * f * np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2), 0.0]); b.cmd_fixed = cmd
            obs, _ = env.reset(seed=k); vs = []; fell = False
            for t in range(n):
                b.cmd = cmd; obs["cmd"] = cmd.astype(np.float32)
                act, _ = ctrl.predict(norm(obs), deterministic=True)
                obs, r, te, tr, info = env.step(act)
                if te: fell = True; break
                vs.append(obs["vel"][:2].astype(float))
            falls += fell
            if not fell: ratios.append(float(np.dot(np.mean(vs[-n // 3:], 0), cmd[:2]) / np.dot(cmd[:2], cmd[:2])))
        out.append(f"{sp * f:.2f}m/s: ratio {np.mean(ratios) if ratios else float('nan'):.2f} falls {falls}/{a.trials}")
    print(f"design {a.design} fric x{fs} roll x{a.roll_fric} rate {int(1 / dt)} Hz | " + " | ".join(out), flush=True)
