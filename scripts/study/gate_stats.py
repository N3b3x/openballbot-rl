"""Mean learned gate g=(1+clip(a[3]))/2 of a hybrid policy while tracking at several speeds (0 = pure LQR, 1 = pure PPO branch)."""
import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
from eval_design import make_env, load_controller  # noqa
run = sys.argv[1]
for d in sys.argv[2:]:
    env, m, f = make_env(d, "hybrid")
    ctrl, norm = load_controller(run, env)
    b = env.unwrapped
    out = []
    for sp in (0.0, 0.4, 0.8, 1.1):
        cmd = sp * f * np.array([1.0, 0, 0]); b.cmd_fixed = cmd
        obs, _ = env.reset(seed=1); gs = []
        for t in range(400):
            b.cmd = cmd; obs["cmd"] = cmd.astype(np.float32)
            with torch.no_grad():
                a, _ = ctrl.predict(norm(obs), deterministic=True)
            gs.append((1 + float(np.clip(a[3], -1, 1))) / 2)
            obs, r, te, tr, info = env.step(a)
            if te: break
        out.append(f"{sp}:{np.mean(gs):.2f}/{np.max(gs):.2f}")
    print(d, " ".join(out), flush=True)
