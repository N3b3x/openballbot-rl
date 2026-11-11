"""CPU MuJoCo reference (with the repo's ContactFrameStepper). Usage: python cpu_ref.py <xml> <tag> [--nolog]"""
import sys, importlib.util, time
import numpy as np
import mujoco
from pathlib import Path
from common import *  # noqa
import runner

spec_ = importlib.util.spec_from_file_location("cf", ROOT / "ballbot_gym" / "utils" / "contact_frame.py")
cf = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(cf)


class CpuBackend:
    """N independent MjData stepped sequentially with ContactFrameStepper."""
    def __init__(self, m, N=1, fix_frames=True):
        self.m = m; self.N = N
        self.ds = [mujoco.MjData(m) for _ in range(N)]
        self.st = cf.ContactFrameStepper(m, fix_frames=fix_frames)
        self.base = m.body("base").id
        self.bg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "the_ball")

    def reset(self, qpos, qvel):
        for d in self.ds:
            mujoco.mj_resetData(self.m, d)
            d.qpos[:] = qpos; d.qvel[:] = qvel
            mujoco.mj_forward(self.m, d)

    def set_ctrl(self, c):
        c = np.broadcast_to(c, (self.N, 3))
        for d, ci in zip(self.ds, c): d.ctrl[:] = ci

    def set_push(self, F):
        F = np.broadcast_to(F, (self.N, 3))
        for d, f in zip(self.ds, F): d.xfrc_applied[self.base, :3] = f

    def step(self, n=1):
        for d in self.ds:
            for _ in range(n): self.st.step(d)

    def state(self):
        qp = np.array([d.qpos for d in self.ds]); qv = np.array([d.qvel for d in self.ds])
        cv = np.array([d.cvel[self.base] for d in self.ds]); bx = np.array([d.xpos[self.base] for d in self.ds])
        gx = np.array([d.geom_xpos[self.bg] for d in self.ds])
        return qp, qv, cv, bx, gx


def run_all(be_factory, m, specs, log=print, design=None):
    out = {}
    be = be_factory(1)
    out["settle"] = runner.run_settle(be, m)
    out["step"] = runner.run_step(be, m, (design.tau / TAU_MAX) if design else 1.0)
    for name, sp in specs.items():
        be = be_factory(sp["cmd"].shape[0])
        t0 = time.time()
        r = runner.run_lqr(be, m, sp, design=design)
        out[name] = r
        log(name, "falls", int(r["fell"].sum()), "wall %.1fs" % (time.time() - t0))
    return out


def flatten(out):
    f = {"settle": out["settle"], "step": out["step"]}
    for k, r in out.items():
        if isinstance(r, dict):
            for kk, vv in r.items(): f[f"{k}__{kk}"] = vv
    return f


if __name__ == "__main__":
    lowprio()
    xml, tag = sys.argv[1], sys.argv[2]
    m = load_cpu_model(xml)
    specs = test_specs(total_mass(m))
    print(tag, "mujoco", mujoco.__version__, "mass", total_mass(m), flush=True)
    out = run_all(lambda N: CpuBackend(m, N), m, specs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"traj_cpu_{tag}.npz", **flatten(out))
