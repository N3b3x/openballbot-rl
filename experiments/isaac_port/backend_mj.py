"""CPU MuJoCo backend (same model, implicitfast integrator, contact-frame stepper as the study environment)."""
import sys

import mujoco
import numpy as np

from common import ROOT, REPEAT, DesignCfg

sys.path.insert(0, str(ROOT))
from ballbot_gym.utils.contact_frame import ContactFrameStepper  # noqa: E402


class MJBackend:
    name = "MuJoCo"

    def __init__(self, cfg: DesignCfg, fix_frames=True, integrator="implicitfast"):
        self.cfg = cfg
        m = self.m = mujoco.MjModel.from_xml_path(str(cfg.xml))
        if integrator == "implicitfast":
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.stepper = ContactFrameStepper(m, fix_frames=fix_frames)
        m.actuator_ctrlrange[:] = np.array([-1.0, 1.0]) * cfg.tau_max
        for k in range(3):
            j = m.joint(f"wheel_joint_{k}").dofadr[0]
            m.dof_armature[j] = cfg.armature
            m.dof_damping[j] = cfg.damping
        self.d = mujoco.MjData(m)
        self.bb, self.kb = m.body("base").id, m.body("ball").id
        self.bq = m.joint("base_free_joint").qposadr[0]
        self.bd = m.joint("base_free_joint").dofadr[0]
        self.kd = m.joint("ball_free_joint").dofadr[0]
        self.kq = m.joint("ball_free_joint").qposadr[0]
        self.wq = np.array([m.joint(f"wheel_joint_{k}").qposadr[0] for k in range(3)])
        self.wdofs = np.array([m.joint(f"wheel_joint_{k}").dofadr[0] for k in range(3)])
        self.ball_geom = m.geom("the_ball").id
        self.wheel_geoms = [m.geom(f"wheel_mesh_{k}").id for k in range(3)]
        self._settled = None
        self.reset(0.0)

    def reset(self, yaw=0.0, drop=0.0):
        """Start from the settled configuration (base/ball heights found by a 0.1 s zero-torque settle), lifted by `drop`."""
        m, d = self.m, self.d
        if self._settled is None:
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
            for _ in range(120):
                self.stepper.step(d)
            self._settled = (d.qpos[self.bq + 2], d.qpos[self.kq + 2])
        mujoco.mj_resetData(m, d)
        d.qpos[self.bq + 3:self.bq + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[self.bq + 2] = self._settled[0] + drop
        d.qpos[self.kq + 2] = self._settled[1] + drop
        mujoco.mj_forward(m, d)

    def state(self):
        d = self.d
        R = d.xmat[self.bb].reshape(3, 3).copy()
        Rb = d.xmat[self.kb].reshape(3, 3)
        return dict(R=R, wb_w=R @ d.qvel[self.bd + 3:self.bd + 6], wk_w=Rb @ d.qvel[self.kd + 3:self.kd + 6],
                    vel=d.cvel[self.bb][3:].copy(), pbody=d.xpos[self.bb].copy(),
                    pball=d.xpos[self.kb] + Rb @ self.m.geom_pos[self.ball_geom],
                    wheel_w=d.qvel[self.wdofs].copy(), t=float(d.time))

    def step_ctrl(self, tau, push=None):
        d = self.d
        d.ctrl[:] = tau
        d.xfrc_applied[self.bb, :3] = push if push is not None else 0.0
        for _ in range(REPEAT):
            self.stepper.step(d)
        d.xfrc_applied[self.bb, :3] = 0.0

    def pose(self):
        d = self.d
        return dict(base_pos=d.qpos[self.bq:self.bq + 3].copy(), base_quat=d.qpos[self.bq + 3:self.bq + 7].copy(),
                    ball_pos=d.qpos[self.kq:self.kq + 3].copy(), ball_quat=d.qpos[self.kq + 3:self.kq + 7].copy(),
                    wheel_q=d.qpos[self.wq].copy(),
                    ball_center=d.xpos[self.kb] + d.xmat[self.kb].reshape(3, 3) @ self.m.geom_pos[self.ball_geom])

    def wheel_ball_forces(self):
        """List of (wheel index, point, world force on the ball side, normal N, friction N) for ball-wheel contacts."""
        m, d = self.m, self.d
        out = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            w = pair & set(self.wheel_geoms)
            if self.ball_geom in pair and w:
                mujoco.mj_contactForce(m, d, i, f6)
                F = c.frame.reshape(3, 3).T @ f6[:3]
                out.append((self.wheel_geoms.index(w.pop()), c.pos.copy(), F, float(f6[0]), float(np.hypot(f6[1], f6[2]))))
        return out
