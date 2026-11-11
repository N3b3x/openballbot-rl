p='train_dev.py'
s=open(p).read()
def rep(o,n):
    global s
    assert o in s, o[:50]
    s=s.replace(o,n,1)
rep("        tilt = torch.rad2deg(torch.acos(up_z.clamp(-1, 1)))\n        bad =","        tilt = torch.nan_to_num(torch.rad2deg(torch.acos(up_z.clamp(-1, 1))), nan=90.0)\n        bad =")
rep("        return torch.cat(parts, 1) if len(parts) > 1 else o\n","        return torch.nan_to_num(torch.cat(parts, 1) if len(parts) > 1 else o, nan=0.0, posinf=0.0, neginf=0.0)\n")
open(p,'w').write(s); print('ok')
