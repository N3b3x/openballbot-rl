import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train as T
env = T.GpuBallbot(8, seed=0, dr=True)
# same initial state in all worlds (yaw 0), identical constant torques, different physics only
env.qpos[:] = env.q0; env.qvel[:] = 0
for t in range(30):
    env.ctrl[:] = torch.tensor([3.0, -1.0, 2.0], device="cuda"); env.be.step(5)
print("mass ball", env.t_mass[:, env.body_ids[1]].cpu().numpy().round(3))
print("wheel speed after 30 steps (should differ across worlds):", env.qvel[:, 6].cpu().numpy().round(3))
print("qvel std across worlds", float(env.qvel[:, 6:9].std(0).mean()))
