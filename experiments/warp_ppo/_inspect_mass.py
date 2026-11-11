import os, sys, torch
os.environ["TILES_FLAT"] = "1"
sys.path.insert(0, ".")
import train_dev as T
env = T.GpuBallbot(256, seed=3, dr=True, design="nom", tiles=True, tile_seed=0)
env.set_dr_scale(0.0); env.dr_m = 2.5; env.dr_mbody = [0.0, 1.0]
env.reset(torch.ones(256, dtype=torch.bool, device="cuda"))
bb, kb = env.body_ids
r = env.t_mass / env.mass_tab[env.didx]
print("base body id", bb, "ball body id", kb, "nbody", env.mass_tab.shape[1])
print("nominal masses (kg):", {i: round(float(env.mass_tab[0, i]), 4) for i in range(env.mass_tab.shape[1])})
print("ratio per body: min/max over worlds")
for i in range(env.mass_tab.shape[1]):
    print(f"  body {i}: {float(r[:, i].min()):.3f} .. {float(r[:, i].max()):.3f}")
import mujoco_warp as mjw
print("model body_mass shape", tuple(env.be.m.body_mass.shape), "is t_mass the model array:", env.be.m.body_mass.ptr == __import__('warp').from_torch(env.t_mass).ptr if hasattr(env.be.m.body_mass, 'ptr') else 'n/a')
print("body_subtreemass shape", tuple(env.be.m.body_subtreemass.shape), " body_invweight0", tuple(env.be.m.body_invweight0.shape), " dof_invweight0", tuple(env.be.m.dof_invweight0.shape))
