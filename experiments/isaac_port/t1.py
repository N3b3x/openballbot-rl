import time
import numpy as np
from common import DesignCfg
from backend_mj import MJBackend
import scenarios as S
cfg = DesignCfg("nom"); be = MJBackend(cfg)
st = be.state(); print("init ball z", st["pball"], "body", st["pbody"], cfg.m_total)
tr = S.settle_trace(be, 0.5); print(tr[::10])
t = time.time()
print(S.speed_table(be, cfg, yaw_dirs=2, steps=400), time.time() - t)
t = time.time(); print(S.push_table(be, cfg, trials=4), time.time() - t)
