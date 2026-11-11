import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PFX = os.environ.get("STEP_PREFIX", "stepsize")      # "stepsize" = friction 0.001 files (S95); "stepsize_mu01" = friction 0.01 (S98/S99)
OUT = os.environ.get("OUT_NAME", "6_ballsize_vs_step")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
R={'s0.75':(0.0675,'#d62728'),'nom':(0.09,'#ff7f0e'),'s1.5':(0.135,'#2ca02c'),'s2.0':(0.18,'#1f77b4')}
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
for ax,kind,bins in zip(axes,('step_up','bump'),(np.arange(0.01,0.0651,0.005),np.arange(0.01,0.0551,0.005))):
    for d,(r,c) in R.items():
        per=[]
        for seed in ('s0','s1'):
            j=json.load(open(f'outputs/study/warp_ppo/g24_scale_rs025_{seed}/{PFX}_{d}.json'))['per_world']
            k=np.array(j['kind']); s=np.array(j['severity']); fl=np.array(j['fell'])
            m=k==kind; hs=s[m]; f=fl[m]
            xs,ys=[],[]
            for lo,hi in zip(bins[:-1],bins[1:]):
                mm=(hs>=lo)&(hs<hi)
                if mm.sum()>=8: xs.append((lo+hi)/2*100); ys.append(f[mm].mean())
            per.append((xs,ys))
        n=min(len(p[0]) for p in per); xs=per[0][0][:n]
        Y=np.array([p[1][:n] for p in per]); mu=Y.mean(0); err=np.abs(Y-mu).max(0)
        ax.errorbar(xs,mu,yerr=err,fmt='o-',color=c,capsize=3,label=f'ball r = {r*100:g} cm')
        ax.axvline(0.29*r*100,color=c,ls=':',lw=1.2)
    ax.set_xlabel(f"{kind.replace('_',' ')} height (cm)"); ax.set_ylim(0,1.02); ax.grid(alpha=0.3)
    ax.set_title(f"Fall probability vs {kind.replace('_',' ')} height, by ball size")
axes[0].set_ylabel("fall probability (20 s at 0.3 m/s, 640 worlds per kind)")
axes[0].legend(loc="lower right",fontsize=9)
fig.text(0.5,-0.02,"Dotted lines: predicted traction limit h = 0.29 r (friction about 1). One policy trained on all sizes, mean of 2 training seeds (bars: seed spread); simulation only, MuJoCo-Warp, DR on, no pushes.",ha="center",fontsize=9)
fig.tight_layout(); fig.savefig(f"outputs/study/warp_ppo/review/{OUT}.png",dpi=150,bbox_inches="tight"); print("ok", OUT)
