import sys, numpy as np, mujoco
from ring import *
cfg = DesignCfg("nom")
m0 = mujoco.MjModel.from_xml_path(str(SRC)); print("ideal total mass", m0.body_mass[1:].sum())
xml, name = make_xml(12, 0.012, 0.0, False, 0.0, tag="diag")
be = RingBackend(cfg, xml)
S.run_drive(be, cfg, np.array([0.4*cfg.f,0,0]), 150)
d, m = be.d, be.m
print("ncon", d.ncon, [(m.geom(c.geom1).name, m.geom(c.geom2).name, round(c.dist,4)) for c in d.contact])
print("wheel qvel", d.qvel[be.wdofs], "roller qvel", np.round(d.qvel[[m.joint(f'roll_joint_0_{i}').dofadr[0] for i in range(12)]],2))
f=np.zeros(6); 
for i in range(d.ncon):
    mujoco.mj_contactForce(m,d,i,f); print(i, np.round(f[:3],2))
