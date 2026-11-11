import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train as T
for cfg in (None, "nom_real"):
    env = T.GpuBallbot(4, seed=0, dr=False, cfg_name=cfg)
    env.qpos[:] = env.q0; env.qvel[:] = 0
    for t in range(20):
        (env.motor_cmd if env.motor else env.ctrl)[:] = torch.tensor([-20.0, 0, 0], device="cuda"); env.be.step(5)
    print(cfg, "motor" if env.motor else "ideal", "wheel0 speed after 0.2 s:", float(env.qvel[0, 6]), "tau_act", float(env.tau_act[0, 0]) if env.motor else None)
