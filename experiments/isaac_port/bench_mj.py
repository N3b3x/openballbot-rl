"""CPU MuJoCo throughput: one process, LQR closed loop (physics steps/s), with and without the Python contact-frame stepper."""
import json
import time

import numpy as np

from backend_mj import MJBackend
from common import ROOT, DesignCfg, LQRCore
from scenarios import FALL_DEG

cfg = DesignCfg("nom")
out = {}
for name, ff in (("contact_frame_stepper(env default)", True), ("mj_step_plain(no contact-frame fix)", False)):
    be = MJBackend(cfg, fix_frames=ff)
    if not ff:
        import mujoco
        be.stepper = type("S", (), {"step": staticmethod(lambda d, m=be.m: mujoco.mj_step(m, d))})()
    ctrl = LQRCore(cfg)
    be.reset(0.0)
    cmd = np.array([0.4 * cfg.f, 0, 0])
    n = 1500
    t0 = time.perf_counter()
    for _ in range(n):
        st = be.state()
        be.step_ctrl(ctrl.act(st, cmd))
    dt = time.perf_counter() - t0
    out[name] = dict(control_steps_per_s=n / dt, physics_steps_per_s=5 * n / dt)
    print(name, out[name], flush=True)
json.dump(out, open(ROOT / "outputs/study/isaac_port/bench_mj.json", "w"), indent=1)
