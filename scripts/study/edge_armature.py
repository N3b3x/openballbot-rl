"""Does the ~1 m/s stability edge move with rotor inertia? LQR re-derived for each armature, run in a simulator
with that armature; 4 directions x speeds, 6 s each. Prints the fall rate and achieved speed."""
import sys
import numpy as np
import yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import ballbot_gym  # noqa: F401,E402
from ballbot_gym.controllers.lqr import LQRPolicy  # noqa: E402

J = sys.argv[1]
DT = float(sys.argv[2]) if len(sys.argv) > 2 else None   # 0.01 = discrete LQR at the 100 Hz control rate
cfg = yaml.safe_load(open(ROOT / "configs/env/trk_d_if.yaml"))
cfg["camera"]["enabled"] = False
cfg["env"].update(cmd_resample_steps=10 ** 9, max_ep_steps=10 ** 6, wheel_armature=float(J))
env = make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"],
                       env_config={k: cfg[k] for k in ("camera", "env", "logging")}, seed=0, eval_env=True)()
b = env.unwrapped
for sp in (0.75, 1.0, 1.25, 1.5):
    falls, ach = 0, []
    for deg in (0, 90, 180, 270):
        pol = LQRPolicy(b, model_json=ROOT / f"outputs/study/results/model_J{J}.json", discrete_dt=DT)
        c = np.array([sp * np.sin(np.radians(deg)), sp * np.cos(np.radians(deg)), 0.0])
        b.cmd_fixed = c
        obs, _ = env.reset(seed=deg)
        vs = []
        for t in range(600):
            b.cmd = c
            obs["cmd"] = c.astype(np.float32)
            a, _ = pol.predict(obs)
            obs, r, te, tr, info = env.step(a)
            vs.append(np.linalg.norm(obs["vel"][:2]))
            if te:
                falls += 1
                break
        ach.append(np.mean(vs[-150:]))
    print(f"J_w={J} dt={DT}: cmd {sp:.2f} m/s -> falls {falls}/4, speed {np.mean(ach):.2f}", flush=True)
