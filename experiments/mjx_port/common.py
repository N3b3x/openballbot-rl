"""Shared helpers for the MJX / MuJoCo-Warp feasibility study (simulation only).

Plant constants mirror the dz_nom design as run by the gym env (configs/env/dz_nom.yaml):
implicitfast, dt 2 ms, action_repeat 5 (100 Hz control), motor torque range +-30 N m, no domain randomisation.
"""
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "study" / "mjx_port"
XML = ROOT / "ballbot_gym" / "models" / "variants" / "dz_nom.xml"
MODEL_JSON = ROOT / "ballbot_gym" / "models" / "variants" / "dz_nom.json"
TAU_MAX = 30.0
ACTION_REPEAT = 5
G = 9.81

# state layout (qpos/qvel) of the compiled model
BQ, BD = 0, 0        # base free joint (qpos 0:7, dof 0:6)
WQ, WD = 7, 6        # wheel hinges (qpos 7:10, dof 6:9)
KQ, KD = 10, 9       # ball free joint (qpos 10:17, dof 9:15)


def lowprio():
    """Idle priority on Windows (CPU is shared with training), nice 19 elsewhere."""
    try:
        if os.name == "nt":
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x40)
        else:
            os.nice(19)
    except Exception:
        pass


class Design:
    """One robot design as evaluated by scripts/study/eval_design.py: xml + LQR json + tau + damping + armature (+ optional grip scale)."""
    def __init__(self, name, mu=1.0):
        import yaml
        base = name.split("+")[0]
        cfg = yaml.safe_load(open(ROOT / "configs" / "env" / f"dz_{base}.yaml"))["env"]
        self.name = name
        self.xml = ROOT / "ballbot_gym" / "models" / "variants" / f"dz_{base}.xml"
        self.json = ROOT / "ballbot_gym" / "models" / "variants" / f"dz_{base}.json"
        self.tau, self.damp, self.arm = float(cfg["max_wheel_velocity"]), cfg["wheel_damping"], cfg["wheel_armature"]
        P = json.load(open(self.json))["params"]
        self.m_total = float(P["m_a"] + P["m_k"])
        self.f = float(np.sqrt(P["r"] / 0.09))
        self.mu = mu
        if "+mu" in name:
            self.mu = float(name.split("+mu")[1])


def load_cpu_model(xml=XML, design=None):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(design.xml if design else xml))
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    tau = design.tau if design else TAU_MAX
    m.actuator_ctrlrange[:] = np.array([-1.0, 1.0]) * tau
    if design:
        for k in range(3):
            a = m.joint(f"wheel_joint_{k}").dofadr[0]
            m.dof_damping[a] = design.damp; m.dof_armature[a] = design.arm
        m.pair_friction[:, 1] *= design.mu          # grip (transverse) friction coefficient
    if os.environ.get("PAIR_MU1"):      # diagnostics (nan_hunt.py): raise the free-rolling coefficient of the anisotropic wheel-ball pairs (0.001 in the xml)
        sel = (m.pair_friction[:, 0] < 0.01) & (m.pair_friction[:, 1] > 0.1)
        m.pair_friction[sel, 0] = float(os.environ["PAIR_MU1"])
    return m


def reset_state(m, yaw=0.0):
    """Initial qpos/qvel as the env does it: mj_resetData, +0.01 height offset on both free joints, yaw about z."""
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[BQ + 2] += 0.01
    d.qpos[KQ + 2] += 0.01
    d.qpos[BQ + 3:BQ + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    eps = float(os.environ.get("MJX_EPS", "0"))      # optional tiny initial roll (rad), for noise-floor estimates
    if eps:
        d.qpos[BQ + 3:BQ + 7] = np.array([np.cos(yaw / 2), np.cos(yaw / 2) * eps / 2, np.sin(yaw / 2) * eps / 2, np.sin(yaw / 2)])
    return d.qpos.copy(), d.qvel.copy()


def total_mass(m):
    return float(m.body_mass[1:].sum())


# ----------------------------------------------------------------------------------------------------------------
# LQR law (numpy, batched over N worlds). Same gains / structure as ballbot_gym/controllers/lqr.py, LQRPolicy with
# its default arguments (q_theta=50, q_phidot=1, q_thetadot=1, r=1, ki=1, k_yaw=5, a_max=0.5), dt = 0.01.
# ----------------------------------------------------------------------------------------------------------------
def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class LQRBatch:
    def __init__(self, n, model_json=MODEL_JSON, tau_max=TAU_MAX, q_theta=50.0, q_phidot=1.0, q_thetadot=1.0, r=1.0,
                 ki=1.0, k_yaw=5.0, a_max=0.5, dt=0.002 * ACTION_REPEAT):
        M = json.load(open(model_json))
        self.A, self.B = np.array(M["A"]), np.array(M["B"])
        self.W = np.array(M["params"]["W"])
        self.rball = M["params"]["r"]
        Q = np.diag([1e-6, q_theta, q_phidot, q_thetadot])
        X = la.solve_continuous_are(self.A, self.B, Q, np.array([[r]]))
        self.K = (self.B.T @ X / r).ravel()
        self.ki, self.k_yaw, self.a_max, self.dt, self.tau_max = ki, k_yaw, a_max, dt, tau_max
        self.WinvT = np.linalg.inv(self.W.T)
        self.int_v = np.zeros((n, 2))
        self.vref = np.zeros((n, 2))

    def __call__(self, qpos, qvel, cvel_lin, cmd):
        """qpos (N,17), qvel (N,15), cvel_lin (N,3) = cvel[base][3:], cmd (N,3) -> wheel torques (N,3) (= data.ctrl)."""
        R = quat_to_mat(qpos[:, BQ + 3:BQ + 7])
        up = R[:, :, 2]
        tilt_y, tilt_x = np.arctan2(up[:, 0], up[:, 2]), np.arctan2(up[:, 1], up[:, 2])
        wb = np.einsum("nij,nj->ni", R, qvel[:, BD + 3:BD + 6])
        Rb = quat_to_mat(qpos[:, KQ + 3:KQ + 7])
        wk = np.einsum("nij,nj->ni", Rb, qvel[:, KD + 3:KD + 6])
        dv = cmd[:, :2] - self.vref
        nv = np.linalg.norm(dv, axis=1, keepdims=True)
        lim = self.a_max * self.dt
        dv = np.where(nv > lim, dv * lim / np.maximum(nv, 1e-12), dv)
        self.vref += dv
        planes = [(tilt_y, wb[:, 1], wk[:, 1], self.vref[:, 0]), (tilt_x, -wb[:, 0], -wk[:, 0], self.vref[:, 1])]
        taus = []
        for i, (th, thd, phd, vref) in enumerate(planes):
            self.int_v[:, i] = np.clip(self.int_v[:, i] + (vref - cvel_lin[:, i]) * self.dt, -2, 2)
            phd_ref = (vref + self.ki * self.int_v[:, i]) / self.rball
            taus.append(-(self.K[1] * th + self.K[2] * (phd - phd_ref) + self.K[3] * thd))
        tau_z = -self.k_yaw * (cmd[:, 2] - wb[:, 2])
        T = np.stack([-taus[1], taus[0], tau_z], axis=1)
        Tb = np.einsum("nji,nj->ni", R, T)            # R^T T
        tau_w = Tb @ self.WinvT.T                     # solve(W^T, Tb)
        return np.clip(tau_w, -self.tau_max, self.tau_max)


def tilt_deg(qpos):
    R = quat_to_mat(qpos[..., BQ + 3:BQ + 7])
    return np.degrees(np.arccos(np.clip(R[..., 2, 2], -1, 1)))


def off_ball_deg(base_xpos, ball_geom_xpos):
    rel = base_xpos - ball_geom_xpos
    return np.degrees(np.arctan2(np.hypot(rel[..., 0], rel[..., 1]), np.maximum(rel[..., 2], 1e-6)))


# ----------------------------------------------------------------------------------------------------------------
# Test protocol shared by all backends
# ----------------------------------------------------------------------------------------------------------------
def test_specs(m_total, f=1.0, speeds=(0.4, 0.8, 1.1)):
    """name -> dict(cmd (N,3), push (N,3), steps, push_steps). Control step = 10 ms. Speeds are Froude-scaled by f."""
    specs = {}
    for sp in speeds:
        cmds = np.array([[sp * f * np.cos(a), sp * f * np.sin(a), 0.0] for a in np.arange(4) * np.pi / 2])
        specs[f"speed_{sp}"] = dict(cmd=cmds, push=np.zeros((4, 3)), steps=400)
    nd = 16
    phis = np.arange(nd) * 2 * np.pi / nd
    for al in (0.5, 1.0, 1.5):
        F = al * m_total * G * np.stack([np.cos(phis), np.sin(phis), np.zeros(nd)], 1)
        specs[f"push_{al}"] = dict(cmd=np.tile([0.0, 0.4 * f, 0.0], (nd, 1)), push=F, steps=500, push_steps=(200, 205))
    return specs
