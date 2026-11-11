"""Is the ball-mass sensitivity of LQR real physics? CPU MuJoCo reference engine, nominal design, flat ground, no disturbances.
Changes the ball body mass by a factor s (inertia unchanged), with and without recomputing the model constants (mj_setConst: invweight0, subtree mass, ...),
and drives LQR at a fixed velocity command. Counts falls over `--steps` control steps.
  python scripts/study/ball_mass_cpu.py --speed 0.6 --trials 16"""
import argparse, sys
from pathlib import Path
import numpy as np, mujoco
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "study"))
import eval_design as ed
from ballbot_gym.controllers.lqr import LQRPolicy
ap = argparse.ArgumentParser(); ap.add_argument("--speed", type=float, default=0.6); ap.add_argument("--trials", type=int, default=16); ap.add_argument("--steps", type=int, default=800)
a = ap.parse_args()
env, mtot, rs = ed.make_env("nom", "lqr")
b = env.unwrapped
ctrl = LQRPolicy(b, discrete_dt=0.01)
ball = b.model.body("ball").id
m0 = float(b.model.body_mass[ball])
norm = lambda o: o
print(f"ball mass {m0:.3f} kg, total {mtot:.2f} kg")
for s in (1.0, 0.63, 1.37, 2.0):
    for recompute in (False, True):
        b.model.body_mass[ball] = m0 * s
        if recompute:
            mujoco.mj_setConst(b.model, b.data)
        falls = 0
        for k in range(a.trials):
            ang = 2 * np.pi * k / a.trials
            cmd = np.array([a.speed * np.cos(ang), a.speed * np.sin(ang), 0.0])
            fell, _ = ed.run_cmd(env, ctrl, norm, cmd, a.steps)
            falls += int(fell)
        print(f"ball mass x{s:.2f}  recompute constants {str(recompute):5s}: falls {falls}/{a.trials}", flush=True)
b.model.body_mass[ball] = m0
