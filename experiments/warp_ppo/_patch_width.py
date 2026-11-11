p='train_dev.py'
s=open(p).read()
def rep(o,n,c=1):
    global s
    assert o in s, o[:60]
    s=s.replace(o,n,c)
rep("def __init__(self, sdim, out, enc=32, hdim=0, pdim=0, use_priv=False, ddim=0):","def __init__(self, sdim, out, enc=32, hdim=0, pdim=0, use_priv=False, ddim=0, width=256, nlayers=2):")
rep("""        self.body = nn.Sequential(nn.Linear(OBS_DIM + (enc if sdim else 0) + hdim + ddim + (pdim if self.use_priv else 0), 256), nn.Tanh(),
                                  nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, out))""","""        layers, d_in = [], OBS_DIM + (enc if sdim else 0) + hdim + ddim + (pdim if self.use_priv else 0)
        for _ in range(nlayers):
            layers += [nn.Linear(d_in, width), nn.Tanh()]; d_in = width
        self.body = nn.Sequential(*layers, nn.Linear(d_in, out))""")
rep("def __init__(self, log_std=-1.0, sdim=0, hdim=0, pdim=0, ddim=0):","def __init__(self, log_std=-1.0, sdim=0, hdim=0, pdim=0, ddim=0, width=256, nlayers=2):")
rep("self.pi, self.v = Net(sdim, ACT_DIM, hdim=hdim, pdim=pdim, use_priv=False, ddim=ddim), Net(sdim, 1, hdim=hdim, pdim=pdim, use_priv=True, ddim=ddim)","self.pi, self.v = Net(sdim, ACT_DIM, hdim=hdim, pdim=pdim, use_priv=False, ddim=ddim, width=width, nlayers=nlayers), Net(sdim, 1, hdim=hdim, pdim=pdim, use_priv=True, ddim=ddim, width=width, nlayers=nlayers)")
rep("ac = AC(sdim=env.sdim, hdim=env.hdim, pdim=env.pdim, ddim=env.ddim).to(DEV)","ac = AC(sdim=env.sdim, hdim=env.hdim, pdim=env.pdim, ddim=env.ddim, width=a.width, nlayers=a.layers).to(DEV)")
rep("ddim=env.ddim, hist=env.hist_k,","ddim=env.ddim, width=a.width, nlayers=a.layers, hist=env.hist_k,")
rep('ap.add_argument("--hist"','ap.add_argument("--width", type=int, default=256); ap.add_argument("--layers", type=int, default=2)\n    ap.add_argument("--hist"')
open(p,'w').write(s); print("ok")
