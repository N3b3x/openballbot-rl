"""Engine-agnostic pieces for the MuJoCo vs PhysX comparison (simulation only).

LQRCore reproduces ballbot_gym.controllers.lqr.LQRPolicy on plain arrays, so the identical control law runs on any
backend.  DesignCfg reads configs/env/dz_<design>.yaml for torque limit, wheel damping/armature and the planar-model json.
"""
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
import yaml

ROOT = Path(__file__).resolve().parents[2]
DT_PHYS = 0.002
REPEAT = 5          # 100 Hz control
DT_CTRL = DT_PHYS * REPEAT
_OLD = "C:\\Users\\Owner\\Documents\\GitHub\\openballbot-rl"


def _fix(p):
    return Path(p.replace(_OLD, str(ROOT)).replace("\\", "/"))


class DesignCfg:
    def __init__(self, design):
        c = yaml.safe_load(open(ROOT / "configs/env" / f"dz_{design}.yaml"))["env"]
        self.design = design
        self.xml = _fix(c["xml_path"])
        self.json = _fix(c["lqr_model_json"])
        self.tau_max = float(c["max_wheel_velocity"])
        self.damping = float(c["wheel_damping"])
        self.armature = float(c["wheel_armature"])
        self.cmd_vmax = float(c["cmd_vmax"])
        p = json.load(open(self.json))["params"]
        self.m_total = float(p["m_a"] + p["m_k"])
        self.f = float(np.sqrt(p["r"] / 0.09))        # Froude speed scale sqrt(s)
        self.params = p


class LQRCore:
    def __init__(self, cfg, q_theta=50.0, q_phidot=1.0, q_thetadot=1.0, r=1.0, ki=1.0, k_yaw=5.0, a_max=0.5):
        M = json.load(open(cfg.json))
        A, B = np.array(M["A"]), np.array(M["B"])
        self.W = np.array(M["params"]["W"])
        self.rball = M["params"]["r"]
        Q = np.diag([1e-6, q_theta, q_phidot, q_thetadot])
        X = la.solve_continuous_are(A, B, Q, np.array([[r]]))
        self.K = np.linalg.solve(np.array([[r]]), B.T @ X).ravel()
        self.ki, self.k_yaw, self.a_max, self.dt = ki, k_yaw, a_max, DT_CTRL
        self.tau_max = cfg.tau_max
        self.reset()

    def reset(self):
        self.int_v = np.zeros(2)
        self.vref = np.zeros(2)

    def act(self, st, cmd):
        """st: R (body->world), wb_w, wk_w, vel (world, body COM). Returns wheel motor torques [N m] (3,)."""
        R = st["R"]
        up = R[:, 2]
        tilt_y, tilt_x = np.arctan2(up[0], up[2]), np.arctan2(up[1], up[2])
        wb, wk = st["wb_w"], st["wk_w"]
        dv = cmd[:2] - self.vref
        nv = np.linalg.norm(dv)
        if nv > self.a_max * self.dt:
            dv *= self.a_max * self.dt / nv
        self.vref += dv
        planes = [(tilt_y, wb[1], wk[1], self.vref[0]), (tilt_x, -wb[0], -wk[0], self.vref[1])]
        taus = []
        for i, (th, thd, phd, vref) in enumerate(planes):
            self.int_v[i] = np.clip(self.int_v[i] + (vref - st["vel"][i]) * self.dt, -2, 2)
            phd_ref = (vref + self.ki * self.int_v[i]) / self.rball
            s = np.array([0.0, th, phd - phd_ref, thd])
            taus.append(float(-self.K @ s))
        tau_z = -self.k_yaw * (cmd[2] - wb[2])
        T = np.array([-taus[1], taus[0], tau_z])
        tau_w = np.linalg.solve(self.W.T, R.T @ T)
        return np.clip(tau_w, -self.tau_max, self.tau_max)      # == env ctrl after the -act*wmax / -ctrl sign flips


def tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
