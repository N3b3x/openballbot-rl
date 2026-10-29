"""Omniwheel contact-frame fix for unpatched MuJoCo.

The ballbot model uses capsules with anisotropic ball-wheel friction
(``<pair friction="0.001 1.0">``). That only works if the contact's first
tangent axis is the capsule axis, which stock MuJoCo does not guarantee
(see ``tools/mujoco_fix.patch``; the patch needs a from-source MuJoCo build).

``ContactFrameStepper`` reproduces the patch from Python: it re-implements
``mj_step`` (RK4, as in the model XML) split into its stages so the contact
frames can be rewritten between ``mj_collision`` and ``mj_makeConstraint``.
With ``fix_frames=False`` it is numerically identical to ``mujoco.mj_step``
(verified in ``tests/unit/test_contact_frame.py``).
"""
import math

import numpy as np
import mujoco

_RK4_C = (0.5, 0.5, 1.0)
_RK4_A = ((0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0))
_RK4_B = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


class ContactFrameStepper:

    def __init__(self, model, ball_geom="the_ball", wheel_geoms=None,
                 fix_frames=True):
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        assert model.na == 0, "activation states are not supported"
        assert model.ntendon == 0 and model.nflex == 0, "tendons/flex are not supported"
        assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
        self.m = model
        self.fix_frames = fix_frames
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.ball = gid(ball_geom)
        wheel_geoms = wheel_geoms or [f"wheel_mesh_{i}" for i in range(3)]
        self.wheels = np.array([gid(n) for n in wheel_geoms])
        self._is_wheel = np.zeros(model.ngeom, dtype=bool)
        self._is_wheel[self.wheels] = True
        self._qvel0 = np.zeros(model.nv)
        self._qpos0 = np.zeros(model.nq)

    # -- one forward evaluation with the frame hook -------------------------
    def forward(self, d):
        m = self.m
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_factorM(m, d)
        mujoco.mj_collision(m, d)
        if self.fix_frames and d.ncon:
            self._align_contacts(d)
        mujoco.mj_makeConstraint(m, d)
        mujoco.mj_island(m, d)
        mujoco.mj_transmission(m, d)
        mujoco.mj_fwdVelocity(m, d)
        mujoco.mj_fwdActuation(m, d)
        mujoco.mj_fwdAcceleration(m, d)
        mujoco.mj_fwdConstraint(m, d)

    def _align_contacts(self, d):
        c = d.contact
        geom = c.geom
        sel = np.flatnonzero((geom == self.ball).any(1) & self._is_wheel[geom].any(1))
        if sel.size == 0:
            return
        frame = c.frame
        xmat = d.geom_xmat
        for i in sel.tolist():
            g0, g1 = geom[i].tolist()
            w = g1 if self._is_wheel[g1] else g0
            ax, ay, az = xmat[w, 2::3].tolist()  # capsule axis (local z) in world frame
            nx, ny, nz = frame[i, :3].tolist()
            k = ax * nx + ay * ny + az * nz
            tx, ty, tz = ax - k * nx, ay - k * ny, az - k * nz
            inv = 1.0 / (math.sqrt(tx * tx + ty * ty + tz * tz) + 1e-12)
            tx, ty, tz = tx * inv, ty * inv, tz * inv
            frame[i, 3:6] = (tx, ty, tz)
            frame[i, 6:9] = (ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx)

    # -- RK4 identical to mj_RungeKutta(m, d, 4) ----------------------------
    def step(self, d):
        m = self.m
        h = m.opt.timestep
        t0 = d.time
        self.forward(d)
        qpos0 = d.qpos.copy()
        qvel0 = d.qvel.copy()
        F_vel = [qvel0.copy()]
        F_acc = [d.qacc.copy()]
        for i in range(3):
            dvel = sum(_RK4_A[i][j] * F_vel[j] for j in range(i + 1))
            dacc = sum(_RK4_A[i][j] * F_acc[j] for j in range(i + 1))
            d.qpos[:] = qpos0
            mujoco.mj_integratePos(m, d.qpos, dvel, h)
            d.qvel[:] = qvel0 + h * dacc
            d.time = t0 + _RK4_C[i] * h
            self.forward(d)
            F_vel.append(d.qvel.copy())
            F_acc.append(d.qacc.copy())
        bvel = sum(b * v for b, v in zip(_RK4_B, F_vel))
        bacc = sum(b * a for b, a in zip(_RK4_B, F_acc))
        d.qpos[:] = qpos0
        mujoco.mj_integratePos(m, d.qpos, bvel, h)
        d.qvel[:] = qvel0 + h * bacc
        d.time = t0 + h
        d.qacc_warmstart[:] = d.qacc  # as mj_advance does
