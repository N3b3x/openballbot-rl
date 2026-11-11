"""Does mjw.rays work on our ballbot + height-field, in many worlds, fast enough to sit inside a PPO loop?
8x8 forward-down ToF-like grid mounted on the base (pitch -35 deg, fov 45 deg). Compares against the analytic flat-floor distance."""
import sys, time
from pathlib import Path
import numpy as np, torch, warp as wp, mujoco_warp as mjw
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
env = T.GpuBallbot(N, seed=0, dr=False, terrain=True)
be = env.be
res, fov, pitch = 8, np.radians(45), np.radians(-35)
a = np.linspace(-fov / 2, fov / 2, res)
az, el = np.meshgrid(a, a + pitch)                   # body frame: x forward, z up
dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1).reshape(-1, 3)
R = dirs.shape[0]
dirs_t = torch.tensor(dirs, dtype=torch.float32, device="cuda")
off = torch.tensor([0.11, 0.0, 0.03], device="cuda")
dist = wp.empty((N, R), dtype=float); gid = wp.empty((N, R), dtype=int); nrm = wp.empty((N, R), dtype=wp.vec3)
bex = wp.array(np.full(R, be.base, dtype=np.int32), dtype=int)       # ignore the robot's own geoms
gg = mjw._src.types.vec6(-1, -1, -1, -1, -1, -1)
rc = None
if len(sys.argv) > 2 and sys.argv[2] == "bvh":
    rc = mjw.create_render_context(env.mjm, nworld=N, cam_res=(8, 8), render_rgb=False, render_depth=False, enabled_geom_groups=[0, 1, 2])
    print("render context ok")
def cast():
    xm = wp.to_torch(be.d.xmat)[:, be.base]           # (N,3,3)
    xp = wp.to_torch(be.d.xpos)[:, be.base]
    p = (xp + xm @ off)[:, None, :].expand(N, R, 3).contiguous()
    v = torch.einsum("nij,rj->nri", xm, dirs_t).contiguous()
    mjw.rays(be.m, be.d, wp.from_torch(p, dtype=wp.vec3), wp.from_torch(v, dtype=wp.vec3), gg, True, bex, dist, gid, nrm, rc)
    return wp.to_torch(dist)
mjw.forward(be.m, be.d)
d0 = cast(); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50): d = cast()
torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 50 * 1e3
print(f"N={N} rays/world={R}: {ms:.2f} ms per cast ({N*R/ms/1e3:.1f} M rays/s); hit fraction {(d>0).float().mean():.3f}")
# analytic check on flat start worlds: base z ~ height of sensor above floor
hit = d[0].reshape(res, res).cpu().numpy()
h = float(wp.to_torch(be.d.xpos)[0, be.base, 2] + 0.03)
print("sensor height", round(h, 3), " centre-row measured", np.round(hit[res // 2], 3), " analytic", round(h / -np.sin(pitch), 3))
# step response: place worlds at y=0.35 facing +y? report distance min/mean over all worlds as sanity
print("distance stats over worlds (m): min %.3f median %.3f max %.3f" % (d[d > 0].min(), d[d > 0].median(), d.max()))
step = time.perf_counter()
for _ in range(20): env.be.step(5)
torch.cuda.synchronize(); print(f"physics only: {(time.perf_counter()-step)/20*1e3:.2f} ms per control step")
