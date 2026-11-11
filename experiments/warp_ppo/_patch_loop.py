p='train_dev.py'
s=open(p).read()
def rep(o,n):
    global s
    assert o in s, o[:50]
    s=s.replace(o,n,1)
rep("        falls, rsum, errs, tilts = 0, 0.0, 0.0, 0.0\n","        z0 = torch.zeros((), device=DEV)\n        falls, rsum, errs, tilts = z0.clone(), z0.clone(), z0.clone(), z0.clone()\n")
rep("""            if trunc.any():       # bootstrap time-limit truncations
                with torch.no_grad():
                    vb = ac.v(on(o_next_n))[:, 0]
                r = torch.where(trunc, r + a.gamma * vb, r)""","""            with torch.no_grad():      # bootstrap time-limit truncations (computed every step: no host sync)
                vb = ac.v(on(o_next_n))[:, 0]
            r = torch.where(trunc, r + a.gamma * vb, r)""")
rep("falls += int(fell.sum()); rsum += float(r_true.mean()); errs += float(info[\"speed_err\"].mean()); tilts += float(info[\"tilt\"].mean())","falls += fell.sum(); rsum += r_true.mean(); errs += info[\"speed_err\"].mean(); tilts += info[\"tilt\"].mean()")
rep("{rsum/T:.4f},{falls/(N*T):.5f},{errs/T:.4f},{tilts/T:.3f}","{float(rsum)/T:.4f},{float(falls)/(N*T):.5f},{float(errs)/T:.4f},{float(tilts)/T:.3f}")
open(p,'w').write(s); print('ok')
