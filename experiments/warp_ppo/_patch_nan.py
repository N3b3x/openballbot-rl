p='train_dev.py'
s=open(p).read()
def rep(o,n):
    global s
    assert o in s, o[:50]
    s=s.replace(o,n,1)
rep("        bad = ~torch.isfinite(self.qpos).all(1)\n","        bad = ~(torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1) & torch.isfinite(o).all(1))\n        o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)      # a non-finite world is a fall and gets reset; keep its values out of the running statistics and history\n")
open(p,'w').write(s); print('ok')
