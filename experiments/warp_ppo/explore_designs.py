import sys; sys.path.insert(0,'../mjx_port')
import numpy as np, warp as wp, mujoco_warp as mjw, dataclasses
from common import *
names=["nom","s0.5","s1.5","zhigh","zlow","kb4","bal4","J02","tau3","tau6","bal025","kb025","J0005","zeta41","zeta66"]
models={}
for n in names:
    d=Design(n); m=load_cpu_model(design=d); models[n]=(m,mjw.put_model(m),d)
ref=models["nom"][1]
diff={}
for n in names[1:]:
    mw=models[n][1]
    for f in dataclasses.fields(ref):
        a=getattr(ref,f.name); b=getattr(mw,f.name)
        if isinstance(a,wp.array) and isinstance(b,wp.array) and a.shape==b.shape:
            try:
                if not np.allclose(a.numpy(),b.numpy(),equal_nan=True): diff.setdefault(f.name,[]).append(n)
            except Exception as e: pass
        elif isinstance(a,wp.array) and isinstance(b,wp.array): diff.setdefault(f.name+"(shape)",[]).append(n)
for k,v in diff.items(): print(k, len(v), v[:6])
print("ctrlrange etc per design tau:", {n:models[n][2].tau for n in names})
