"""Does a policy stay on the ball? Reports max/mean off-ball angle over the scripted command sequence."""
import sys, numpy as np, torch, yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
from evaluate import _load_policy, _normalize
from ballbot_rl.training.utils import make_ballbot_env
import ballbot_gym
from render_control import SEQ
run, conf = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{conf}.yaml")); cfg["camera"]["enabled"] = False
cfg["env"]["cmd_resample_steps"] = 10**9; cfg["env"].pop("max_ball_offset_deg", None)
env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config={k: cfg[k] for k in ("camera","env","logging")}, seed=0, eval_env=True)()
model, vn = _load_policy(ROOT / "outputs/study/runs" / run, "proprio", "final_model")
b = env.unwrapped; b.cmd_fixed = np.zeros(3); obs, _ = env.reset(seed=0)
for lab, cmd, secs in SEQ:
    cmd = np.asarray(cmd, float); o = []
    for _ in range(int(secs*100)):
        b.cmd = cmd; obs["cmd"] = cmd.astype(np.float32)
        with torch.no_grad(): a, _ = model.predict(_normalize(obs, vn), deterministic=True)
        obs, r, te, tr, info = env.step(a); o.append(info["off_ball_deg"])
        if te or tr: obs, _ = env.reset(seed=1)
    print(f"{lab:24s} off-ball max {max(o):5.1f} deg  mean {np.mean(o):5.1f}")
