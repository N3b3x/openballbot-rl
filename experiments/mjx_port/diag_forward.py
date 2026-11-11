import sys; import numpy as np
from warp_sim import *
import cpu_ref
lowprio()
m=load_cpu_model(); q,v=reset_state(m)
np.set_printoptions(precision=5,suppress=True,linewidth=200)
for fix in (True,False):
    be=cpu_ref.CpuBackend(m,1,fix); be.reset(q,v)
    d=be.ds[0]
    be.st.forward(d)     # forward with hook
    w=WarpBallbot(m,1,fix_frames=fix); w.reset(q,v)
    print('fix',fix)
    print(' cpu qacc', d.qacc[:15]); print(' wrp qacc', w.d.qacc.numpy()[0])
    n=int(w.d.nacon.numpy()[0])
    fc=np.array([d.efc_force[d.contact[i].efc_address:d.contact[i].efc_address+3] for i in range(d.ncon)])
    ef=w.d.efc.force.numpy()[0]; ad=w.d.contact.efc_address.numpy()[:n]
    fw=np.array([ef[a[:3]] for a in ad])
    print(' cpu contact force (n,t1,t2)\n',fc); print(' wrp\n',fw)
    print(' frames t1 cpu',[d.contact[i].frame[3:6] for i in range(d.ncon)]); print(' t1 wrp',w.d.contact.frame.numpy()[:n,1])
    # one RK-free step comparison
    be.st.step(d); w.step(1)
    print(' after 1 step qvel cpu', d.qvel[:15]); print(' wrp', w.d.qvel.numpy()[0])
print('opt', m.opt.iterations, m.opt.ls_iterations, m.opt.tolerance, m.opt.ls_tolerance, m.opt.impratio, m.opt.cone, m.opt.jacobian)
