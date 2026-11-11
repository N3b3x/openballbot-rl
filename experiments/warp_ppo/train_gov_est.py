"""Train the sensor-to-height-grid estimator that drives the speed governor (simulation only).

Input: the noisy 18x14 ToF zone ranges plus the 25-D proprioception, from collect_percep.py. Output: the 7x5 height grid ahead (env.scan_feat, divided by 6 cm),
which is exactly what governor.py reads from the privileged map, so eval_governor.py --source enc:NAME can swap the oracle for this estimate.
Split by world, as in train_percep.py. Reported: validation MSE, and for the governor's own decision (any |step| > thr within d_near ahead in the
centre columns) precision and recall against the oracle.
  python train_gov_est.py --data percep_g26 --out tof18
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--width", type=int, default=512); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=4096); ap.add_argument("--nworld", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--thr", type=float, default=0.01, help="hazard threshold in m (governor default)")
a = ap.parse_args()
torch.manual_seed(a.seed)
DEV = "cuda"
RUN = Path(__file__).resolve().parents[2] / "outputs" / "study" / "warp_ppo" / "percep"
d = torch.load(RUN / f"{a.data}.pt")
depth, prop, scan = d["depth"].float(), d["proprio"].float(), d["scan"].float()
nrow, ncol = len(d["fwd"]), len(d["lat"])
n = depth.shape[0]
world = torch.arange(n) % a.nworld
va = world % 5 == 0; tr = ~va
dm, ds = depth[tr].mean(0), depth[tr].std(0).clamp_min(1e-3)
pm, ps = prop[tr].mean(0), prop[tr].std(0).clamp_min(1e-3)
X = torch.cat([((depth - dm) / ds).clamp(-10, 10), ((prop - pm) / ps).clamp(-6, 6)], 1)
Y = scan
nin = X.shape[1]
net = nn.Sequential(nn.Linear(nin, a.width), nn.GELU(), nn.Linear(a.width, a.width), nn.GELU(), nn.Linear(a.width, nrow * ncol)).to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
Xtr, Ytr = X[tr].to(DEV), Y[tr].to(DEV); Xva, Yva = X[va].to(DEV), Y[va].to(DEV)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)


def hazard(s):
    """Governor-style decision: any |height| > thr (m) in the first three rows (0.1 to 0.5 m ahead), centre three columns."""
    g = s.reshape(-1, nrow, ncol)[:, :3, ncol // 2 - 1:ncol // 2 + 2] * 0.06
    return (g.abs() > a.thr).any(2).any(1)


for ep in range(a.epochs):
    net.train(); perm = torch.randperm(Xtr.shape[0], device=DEV)
    for i in range(0, len(perm), a.bs):
        b = perm[i:i + a.bs]
        loss = ((net(Xtr[b]) - Ytr[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if ep % 10 == 9 or ep == a.epochs - 1:
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xva[i:i + 65536]) for i in range(0, Xva.shape[0], 65536)])
            mse = float(((pv - Yva) ** 2).mean()); base = float(((Yva - Ytr.mean(0)) ** 2).mean())
            hp, ht = hazard(pv), hazard(Yva)
            prec = float((hp & ht).sum() / hp.sum().clamp_min(1)); rec = float((hp & ht).sum() / ht.sum().clamp_min(1))
        print(f"epoch {ep + 1}: val MSE {mse:.4f} (predict-mean {base:.4f}); hazard precision {prec:.2f} recall {rec:.2f} (base rate {float(ht.float().mean()):.2f})", flush=True)
res = dict(val_mse=mse, mean_baseline_mse=base, hazard_precision=prec, hazard_recall=rec, hazard_base_rate=float(ht.float().mean()), thr=a.thr)
torch.save(dict(net=net.state_dict(), nin=nin, width=a.width, dmean=dm, dstd=ds, pmean=pm, pstd=ps, mount="head", res=(18, 14), results=res, data=a.data),
           RUN / f"{a.out}_gov.pt")
print("saved", RUN / f"{a.out}_gov.pt", res)
