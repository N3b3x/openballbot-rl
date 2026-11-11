import sys, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"/"study")); sys.path.insert(0, str(Path(__file__).parent))
import eval_design as ed
import eval_policy as ep
pol, _ = ep.load("wp_nom_s0", "cpu")
env, m, f = ed.make_env("nom", "ppo")
b = env.unwrapped; b.init_yaw_random = False
cmd = np.array([0.4, 0, 0.0]); b.cmd_fixed = cmd
obs, _ = env.reset(seed=1)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for t in range(40):
    x = torch.tensor(np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in ep.KEYS]))[None]
    a = pol(x)[0].detach().numpy()
    obs, r, te, tr, info = env.step(a)
    if t < 4 or t % 10 == 0 or te:
        print(t, "a", a, "| ori", obs["orientation"], "ang", obs["angular_vel"], "vel", obs["vel"], "mot", obs["motor_state"], "vint", obs["vint"], "r", round(r, 3))
    if te: print("FELL at", t); break
