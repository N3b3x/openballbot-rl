"""Record deterministic reference rollouts of robs_c_s0 on flat ground in the real Python env (MuJoCo 3.13 +
ContactFrameStepper, implicitfast, 500 Hz physics, action_repeat 5 => 100 Hz control).

Domain randomisation (latency, pushes, obs noise, mass/friction/torque scaling) and random initial yaw are switched
OFF so the rollout is a clean, repeatable reference. Writes site/live/ref_rollout.json.
Run:  OPENBLAS_NUM_THREADS=1 python scripts/site/ref_rollout.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/study"))
RUN = "robs_c_s0"
CASES = [("fwd_diag", [0.4, 0.4, 0.0], 800, True), ("yaw_fwd", [0.0, 0.8, 0.5], 800, True), ("turn_back", [-0.6, 0.3, -0.8], 800, True),
         ("hold", [0.0, 0.0, 0.0], 300, True),
         ("fwd_diag_nofix", [0.4, 0.4, 0.0], 600, False)]   # last: contact-frame fix disabled, to show why it matters


def r(a, n=7):
    return np.round(np.asarray(a, float), n).tolist()


def main():
    import torch
    torch.set_num_threads(1)
    from evaluate import _load_policy, _normalize
    from ballbot_rl.training.utils import make_ballbot_env
    import ballbot_gym  # noqa: F401
    model, vn = _load_policy(ROOT / "outputs/study/runs" / RUN, "proprio", "final_model")
    cfg = yaml.safe_load(open(ROOT / "configs/env/robs_c.yaml"))
    cfg["env"]["dr"] = {}
    cfg["env"]["init_yaw_random"] = False
    out = {"run": RUN, "dt_control": 0.01, "cases": []}
    for name, cmd, n, fix in CASES:
        cfg["env"]["cmd_fixed"] = cmd
        cfg["env"]["fix_wheel_contacts"] = fix
        env_config = {k: cfg[k] for k in ("camera", "env", "logging") if k in cfg}
        env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config=env_config,
                               seed=0, eval_env=True)()
        base = env.unwrapped
        obs, _ = env.reset(seed=0)
        d = base.data
        rec = dict(name=name, cmd=cmd, qpos0=r(d.qpos, 9), qvel0=r(d.qvel, 9), obs=[], act=[], pos=[], ball=[], qpos=[], qvel=[], tilt=[])
        rec["obs0"] = {k: r(v, 7) for k, v in obs.items()}
        fell_at = None
        for t in range(n):
            with torch.no_grad():
                a, _ = model.predict(_normalize(obs, vn), deterministic=True)
            rec["obs"].append({k: r(v, 7) for k, v in obs.items()})
            rec["act"].append(r(a, 7))
            obs, rew, term, trunc, info = env.step(a)
            rec["pos"].append(r(d.xpos[base.model.body("base").id], 6))
            rec["ball"].append(r(d.xpos[base.model.body("ball").id], 6))
            rec["qpos"].append(r(d.qpos, 9))
            rec["qvel"].append(r(d.qvel, 9))
            if term:
                fell_at = t
                break
        rec["fell_at_step"] = fell_at
        rec["steps"] = len(rec["act"])
        rec["final_obs"] = {k: r(v, 7) for k, v in obs.items()}
        out["cases"].append(rec)
        env.close()
        print(name, cmd, "steps", rec["steps"], "fell", fell_at, "final xy", rec["pos"][-1][:2])
    p = ROOT / "site/live/ref_rollout.json"
    json.dump(out, open(p, "w"), separators=(",", ":"))
    print("wrote", p, p.stat().st_size // 1024, "KiB")
    # lean copy for the browser (closed-loop positions for every step, recorded observations/actions for the first 150 steps)
    web = {"run": RUN, "dt_control": 0.01, "cases": [
        dict(name=c["name"], cmd=c["cmd"], qpos0=c["qpos0"], qvel0=c["qvel0"], steps=c["steps"], fell_at_step=c["fell_at_step"],
             pos_xy=[q[:2] for q in c["pos"]], obs=c["obs"][:150], act=c["act"][:150]) for c in out["cases"]]}
    pw = ROOT / "site/live/ref_web.json"
    json.dump(web, open(pw, "w"), separators=(",", ":"))
    print("wrote", pw, pw.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
