"""MuJoCo-Warp backend for the ballbot, with the omni-wheel contact-frame fix as a Warp kernel hooked after collision."""
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src import collision_driver as _cd


def _quiet_hfield(m):
    """Silence the per-step 'height field collision overflow' print (>= 50 hfield contacts per pair).
    The overflow itself is unchanged (extra contacts are still dropped); without this the warning
    writes ~15 GB per training run to the .out file."""
    try:
        m.opt.warn_overflow &= ~mjw.OverflowType.HFIELD
    except Exception:
        pass
    # diagnostics knobs (nan_hunt.py): solver settings from the environment; defaults are the model's own
    import os
    try:
        if os.environ.get("SOLVER_ITER"):
            m.opt.iterations = int(os.environ["SOLVER_ITER"])
        if os.environ.get("LS_ITER"):
            m.opt.ls_iterations = int(os.environ["LS_ITER"])
        if os.environ.get("NO_WARMSTART"):
            m.opt.disableflags |= mjw.DisableBit.WARMSTART
        if os.environ.get("SOLVER"):
            m.opt.solver = getattr(mjw.SolverType, os.environ["SOLVER"].upper())
        if os.environ.get("CONE"):
            m.opt.cone = {"pyramidal": mjw.ConeType.PYRAMIDAL, "elliptic": mjw.ConeType.ELLIPTIC}[os.environ["CONE"].lower()]
        if os.environ.get("IMPRATIO"):
            m.opt.impratio = float(os.environ["IMPRATIO"])
    except Exception as e:
        print("solver knob failed:", e)

from common import *  # noqa

_FIX = {}


@wp.kernel
def _fix_frames(nacon: wp.array[int], geom: wp.array[wp.vec2i], worldid: wp.array[int], is_wheel: wp.array[int],
                ball: int, xmat: wp.array2d[wp.mat33], frame: wp.array[wp.mat33]):
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    w = int(-1)
    if g[0] == ball and is_wheel[g[1]] == 1:
        w = g[1]
    elif g[1] == ball and is_wheel[g[0]] == 1:
        w = g[0]
    if w < 0:
        return
    f = frame[i]
    n = wp.vec3(f[0, 0], f[0, 1], f[0, 2])
    R = xmat[worldid[i], w]
    a = wp.vec3(R[0, 2], R[1, 2], R[2, 2])       # capsule axis (local z) in world
    t = a - wp.dot(a, n) * n
    lt = wp.length(t)
    if lt < 1.0e-4:
        # wheel axis (nearly) parallel to the contact normal: the capsule's end cap touches the ball. The tangent is undefined;
        # keep the solver's own frame. Overwriting it with zero tangents made the constraint rows singular and the state went
        # non-finite within one step (found 2026-11-09 by nan_hunt.py; this was every "standing fall" in S94 to S97).
        return
    t = t / lt
    s = wp.cross(n, t)
    frame[i] = wp.mat33(n[0], n[1], n[2], t[0], t[1], t[2], s[0], s[1], s[2])


_orig_collision = _cd.collision


def _collision_fixed(m, d, awake_prev=None):
    _orig_collision(m, d, awake_prev) if awake_prev is not None else _orig_collision(m, d)
    if _FIX.get("on"):
        wp.launch(_fix_frames, dim=d.naconmax, inputs=[d.nacon, d.contact.geom, d.contact.worldid, _FIX["is_wheel"],
                                                       _FIX["ball"], d.geom_xmat, d.contact.frame])


_cd.collision = _collision_fixed


class WarpBallbot:
    def __init__(self, mjm, nworld=1, fix_frames=True, nconmax=16, njmax=64):
        self.mjm = mjm
        self.use_graph, self._graph = True, None
        self.nworld = nworld
        gid = lambda n: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, n)
        isw = np.zeros(mjm.ngeom, dtype=np.int32)
        for k in range(3):
            isw[gid(f"wheel_mesh_{k}")] = 1
        _FIX.update(on=fix_frames, is_wheel=wp.array(isw, dtype=int), ball=gid("the_ball"))
        self.m = mjw.put_model(mjm)
        _quiet_hfield(self.m)
        mjd = mujoco.MjData(mjm)
        self.d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
        self.base = mjm.body("base").id
        self.ball_gid = gid("the_ball")
        mjw.step(self.m, self.d)      # eager step: loads/compiles all kernels before any CUDA-graph capture

    def reset(self, qpos, qvel):
        """qpos (N,nq) or (nq,), qvel likewise; runs a forward pass to populate derived quantities."""
        N = self.nworld
        qp = np.broadcast_to(qpos, (N, self.mjm.nq)).astype(np.float32).copy()
        qv = np.broadcast_to(qvel, (N, self.mjm.nv)).astype(np.float32).copy()
        wp.copy(self.d.qpos, wp.array(qp, dtype=float))
        wp.copy(self.d.qvel, wp.array(qv, dtype=float))
        self.d.time.zero_()
        self.d.qacc_warmstart.zero_()
        self.d.ctrl.zero_()
        self.d.xfrc_applied.zero_()
        mjw.forward(self.m, self.d)

    def set_ctrl(self, c):
        wp.copy(self.d.ctrl, wp.array(np.broadcast_to(c, (self.nworld, 3)).astype(np.float32).copy(), dtype=float))

    def set_push(self, F):
        x = np.zeros((self.nworld, self.mjm.nbody, 6), dtype=np.float32)
        x[:, self.base, :3] = F
        wp.copy(self.d.xfrc_applied, wp.array(x, dtype=wp.spatial_vector))

    def step(self, n=1):
        if n == ACTION_REPEAT and self.use_graph:
            if self._graph is None:
                with wp.ScopedCapture() as cap:
                    for _ in range(ACTION_REPEAT):
                        mjw.step(self.m, self.d)
                self._graph = cap.graph
                # capture records but does not execute: run once now
            wp.capture_launch(self._graph)
            return
        for _ in range(n):
            mjw.step(self.m, self.d)

    def state(self):
        qp = self.d.qpos.numpy().astype(np.float64)
        qv = self.d.qvel.numpy().astype(np.float64)
        cv = self.d.cvel.numpy()[:, self.base].astype(np.float64)   # (N,6) ang, lin
        bx = self.d.xpos.numpy()[:, self.base].astype(np.float64)
        gx = self.d.geom_xpos.numpy()[:, self.ball_gid].astype(np.float64)
        return qp, qv, cv, bx, gx
