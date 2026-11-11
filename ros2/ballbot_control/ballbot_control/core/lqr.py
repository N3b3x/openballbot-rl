"""LQR base controller (single robot), the law of experiments/warp_ppo/lqr_gpu.py GpuLQR.act.

Per sagittal plane a discrete LQR on the planar ballbot model (state: [unused, tilt, ball rate - ball rate ref, tilt rate]), a
velocity reference that slews toward the command at a_max, a slow integral of the reference minus the measured COM velocity, and a
proportional yaw-rate loop. The plane torques act on the ball (world axes), are rotated into the body frame and mapped to the three
wheels through the planar model's W matrix. Output in normalised action units (-tau_wheel / tau_max), clipped to [-1, 1].
"""
import numpy as np

from .math3d import quat_to_mat


class LQRController:
    def __init__(self, lqr_cfg, design):
        L = design["lqr"]
        self.K = np.asarray(L["K"], dtype=np.float64)                 # (4,) discrete gain at dt
        self.WT = np.asarray(L["W"], dtype=np.float64).T              # W maps wheel torques to body-frame ball torque: Tb = W^T tau_w
        self.rb = float(L["r_ball"])
        self.tau = float(design["tau"])
        self.ki, self.k_yaw, self.a_max = float(lqr_cfg["ki"]), float(lqr_cfg["k_yaw"]), float(lqr_cfg["a_max"])
        self.dt, self.int_clip = float(lqr_cfg["dt"]), float(lqr_cfg["int_clip"])
        self.reset()

    def reset(self):
        self.vref = np.zeros(2)
        self.int_v = np.zeros(2)
        self.first = True       # the simulator's first tick after a reset has no measured velocity yet (zero in the integral)

    def act(self, s, cmd):
        """s: RobotState, cmd (3,) shaped command (vx, vy world, wz). Returns the LQR action (3,) in [-1, 1]."""
        R = quat_to_mat(np.asarray(s.quat, dtype=np.float64))
        up = R[:, 2]
        tilt_y, tilt_x = np.arctan2(up[0], up[2]), np.arctan2(up[1], up[2])
        wb = R @ np.asarray(s.omega_body, dtype=np.float64)            # base angular velocity, world
        wk = np.asarray(s.ball_omega, dtype=np.float64)                # ball angular velocity, world
        dv = np.asarray(cmd[:2], dtype=np.float64) - self.vref
        nv = max(np.linalg.norm(dv), 1e-9)
        self.vref = self.vref + dv * min(self.a_max * self.dt / nv, 1.0)
        v_meas = np.zeros(2) if self.first else np.clip(np.asarray(s.v_com[:2], dtype=np.float64), -8.0, 8.0)
        self.first = False
        self.int_v = np.clip(self.int_v + (self.vref - v_meas) * self.dt, -self.int_clip, self.int_clip)
        taus = []
        for i, (th, thd, phd) in enumerate([(tilt_y, wb[1], wk[1]), (tilt_x, -wb[0], -wk[0])]):
            phd_ref = (self.vref[i] + self.ki * self.int_v[i]) / self.rb
            st = np.array([0.0, th, phd - phd_ref, thd])
            taus.append(-float(self.K @ st))
        tau_z = -self.k_yaw * (cmd[2] - wb[2])
        T = np.array([-taus[1], taus[0], tau_z])                       # torque on the ball, world axes
        Tb = R.T @ T
        tau_w = np.linalg.solve(self.WT, Tb)
        return np.clip(-tau_w / self.tau, -1.0, 1.0)
