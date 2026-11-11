import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import train as T, eval_policy as ep
pol, _ = ep.load("wp_nom_s0", "cuda")
n = 2
env = T.GpuBallbot(n, seed=1, dr=False); env.resample = env.max_steps = 10**9
env.cmd = torch.tensor([[0.4, 0, 0]], device="cuda").repeat(n, 1)
# force yaw 0
q = env.qpos; q[:, 3] = 1; q[:, 6] = 0
o = env.reset_obs()
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for t in range(40):
    with torch.no_grad(): a = pol(o)
    o, r, fell, tr, info = env.step(a)
    if t < 4 or t % 10 == 0 or fell.any():
        print(t, "a", a[0].cpu().numpy(), "| ori", o[0,0:3].cpu().numpy(), "ang", o[0,3:6].cpu().numpy(), "vel", o[0,6:9].cpu().numpy(), "mot", o[0,9:12].cpu().numpy(), "vint", o[0,20:22].cpu().numpy(), "r", round(float(r[0]),3))
    if fell.any(): print("FELL", t); break
