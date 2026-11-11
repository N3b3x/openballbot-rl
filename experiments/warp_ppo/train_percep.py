"""Pretrain the perception encoder on labelled depth data (collect_percep.py), then freeze it for the policy.

  depth (noisy 18x14 zone ranges) + proprio  ->  encoder -> 32-D latent  ->  decoder -> 7x5 height grid ahead (the label), plus hazard heads (rise / drop in cm within 0.5 m)
Train/validation are split by WORLD (not by frame), so the validation set is terrain the encoder has never seen from a neighbouring frame.
Baselines for honesty: predict the mean; predict from proprioception only (no depth). If the encoder beats both clearly, the latent carries terrain information.
  python train_percep.py --data percep_v1 --out enc_v1
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--latent", type=int, default=32); ap.add_argument("--width", type=int, default=256)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--bs", type=int, default=4096)
ap.add_argument("--nworld", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
DEV = "cuda"
RUN = Path(__file__).resolve().parents[2] / "outputs" / "study" / "warp_ppo" / "percep"
d = torch.load(RUN / f"{a.data}.pt")
depth, prop, scan, haz = d["depth"].float(), d["proprio"].float(), d["scan"].float(), d["haz"].float()
nrow, ncol = len(d["fwd"]), len(d["lat"])
n = depth.shape[0]
world = torch.arange(n) % a.nworld
tr, va = world < int(0.9 * a.nworld), world >= int(0.9 * a.nworld)
X_d = (depth / 4.0).clamp(0, 1)
pm, ps = prop[tr].mean(0), prop[tr].std(0).clamp_min(1e-3)
X_p = ((prop - pm) / ps).clamp(-6, 6)
Y, H = scan.clamp(-6, 6), (haz / 3.0).clamp(0, 6)


class Enc(nn.Module):
    def __init__(self, nd, npr, w, z):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nd + npr, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, z))

    def forward(self, xd, xp):
        return self.net(torch.cat([xd, xp], 1))


class Dec(nn.Module):
    def __init__(self, z, w, ny):
        super().__init__()
        self.net = nn.Sequential(nn.GELU(), nn.Linear(z, w), nn.GELU(), nn.Linear(w, ny + 2))

    def forward(self, z):
        o = self.net(z)
        return o[:, :-2], o[:, -2:]


def fit(use_depth, label):
    enc = Enc(X_d.shape[1] if use_depth else 0, X_p.shape[1], a.width, a.latent).to(DEV); dec = Dec(a.latent, a.width, Y.shape[1]).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=a.lr, weight_decay=1e-4)
    xd, xp, y, h = X_d[tr].to(DEV), X_p[tr].to(DEV), Y[tr].to(DEV), H[tr].to(DEV)
    m = xd.shape[0]; steps = a.epochs * (m // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    for ep in range(a.epochs):
        perm = torch.randperm(m, device=DEV)
        for k in range(m // a.bs):
            ix = perm[k * a.bs:(k + 1) * a.bs]
            z = enc(xd[ix] if use_depth else xd[ix][:, :0], xp[ix]); ys, hs = dec(z)
            loss = nn.functional.smooth_l1_loss(ys, y[ix]) + 0.5 * nn.functional.smooth_l1_loss(hs, h[ix])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    enc.eval(); dec.eval()
    with torch.no_grad():
        vd, vp = X_d[va].to(DEV), X_p[va].to(DEV)
        z = enc(vd if use_depth else vd[:, :0], vp); ys, hs = dec(z)
    return enc, ys.cpu(), hs.cpu()


def report(ys, hs, label):
    y, h = Y[va], H[va]
    r2 = 1 - ((ys - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-9)
    r2r = r2.reshape(nrow, ncol).mean(1)
    mae_cm = (ys - y).abs().mean() * 6.0
    rise_true, rise_pred = haz[va][:, 0] >= 2, hs[:, 0] * 3.0 >= 2
    drop_true, drop_pred = haz[va][:, 1] >= 2, hs[:, 1] * 3.0 >= 2
    def f1(t, p):
        tp = float((t & p).sum()); return 2 * tp / max(1.0, float(t.sum() + p.sum()))
    res = dict(model=label, R2_mean=float(r2.mean()), R2_by_forward_row=[round(float(v), 3) for v in r2r], MAE_cm=float(mae_cm),
               rise_ge2cm_F1=f1(rise_true, rise_pred), drop_ge2cm_F1=f1(drop_true, drop_pred),
               rise_rate=float(rise_true.float().mean()), drop_rate=float(drop_true.float().mean()))
    print(json.dumps(res), flush=True)
    return res


results = []
mean_pred = Y[tr].mean(0, keepdim=True).expand_as(Y[va])
results.append(report(mean_pred, torch.zeros(Y[va].shape[0], 2), "mean_baseline"))
_, ys, hs = fit(False, "proprio_only"); results.append(report(ys, hs, "proprio_only"))
enc, ys, hs = fit(True, "depth+proprio"); results.append(report(ys, hs, "depth+proprio"))

# policy-shaped depth-only encoder: Linear(252,128)-Tanh-Linear(128,32)-Tanh on per-zone standardised depth, exactly the layer the policy uses, so it can be frozen into the policy
dm, ds = depth[tr].mean(0), depth[tr].std(0).clamp_min(0.05)
Xn = ((depth - dm) / ds).clamp(-10, 10)
net_enc = nn.Sequential(nn.Linear(depth.shape[1], 128), nn.Tanh(), nn.Linear(128, a.latent), nn.Tanh()).to(DEV); dec2 = Dec(a.latent, a.width, Y.shape[1]).to(DEV)
opt2 = torch.optim.AdamW(list(net_enc.parameters()) + list(dec2.parameters()), lr=a.lr, weight_decay=1e-4)
xn, y_, h_ = Xn[tr].to(DEV), Y[tr].to(DEV), H[tr].to(DEV); m_ = xn.shape[0]
sch2 = torch.optim.lr_scheduler.OneCycleLR(opt2, a.lr, total_steps=a.epochs * (m_ // a.bs))
for ep in range(a.epochs):
    perm = torch.randperm(m_, device=DEV)
    for k in range(m_ // a.bs):
        ix = perm[k * a.bs:(k + 1) * a.bs]
        ys_, hs_ = dec2(net_enc(xn[ix]))
        loss = nn.functional.smooth_l1_loss(ys_, y_[ix]) + 0.5 * nn.functional.smooth_l1_loss(hs_, h_[ix])
        opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step(); sch2.step()
with torch.no_grad():
    ys2, hs2 = dec2(net_enc(Xn[va].to(DEV)))
results.append(report(ys2.cpu(), hs2.cpu(), "depth_only_policy_shaped"))
torch.save(dict(net=net_enc.state_dict(), dmean=dm, dstd=ds, latent=a.latent, results=results), RUN / f"{a.out}_net.pt")
torch.save(dict(enc=enc.state_dict(), nd=X_d.shape[1], npr=X_p.shape[1], width=a.width, latent=a.latent, pmean=pm, pstd=ps, results=results), RUN / f"{a.out}.pt")
json.dump(results, open(RUN / f"{a.out}.json", "w"), indent=1)
print("saved", a.out)
