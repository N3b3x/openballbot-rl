"""Model-based baseline: LQR designed on the derived planar ballbot model (scripts/study/ballbot_model.py),
applied per vertical plane, with a velocity reference and integral action, plus a yaw-rate loop.

Exposes the same predict(obs, deterministic) interface as an SB3 policy so the benchmark scripts can run it.
State is read from the simulator (body tilt and rates from the IMU-equivalent orientation/gyro, ball angular
velocity as a wheel-odometry equivalent).
"""
import json
from pathlib import Path

import numpy as np
import quaternion
import scipy.linalg as la

MODEL_JSON = Path(__file__).resolve().parents[1] / "models" / "ballbot_planar_model.json"


class LQRPolicy:
    def __init__(self, env, model_json=MODEL_JSON, q_theta=50.0, q_phidot=1.0,
                 q_thetadot=1.0, r=1.0, ki=1.0, k_yaw=5.0, tau_max=None, a_max=0.5, discrete_dt=None):
        M = json.load(open(model_json))
        self.A, self.B = np.array(M["A"]), np.array(M["B"])
        self.W = np.array(M["params"]["W"])
        self.rball = M["params"]["r"]
        Q = np.diag([1e-6, q_theta, q_phidot, q_thetadot])
        if discrete_dt:
            # discrete-time LQR at the actual control period (zero-order hold): avoids sampled-data instability
            # when the continuous design has poles faster than the control rate can follow
            from scipy.signal import cont2discrete
            Ad, Bd, *_ = cont2discrete((self.A, self.B, np.eye(4), np.zeros((4, 1))), discrete_dt)
            X = la.solve_discrete_are(Ad, Bd, Q * discrete_dt, np.array([[r * discrete_dt]]))
            self.K = np.linalg.solve(np.array([[r * discrete_dt]]) + Bd.T @ X @ Bd, Bd.T @ X @ Ad).ravel()
        else:
            X = la.solve_continuous_are(self.A, self.B, Q, np.array([[r]]))
            self.K = np.linalg.solve(np.array([[r]]), self.B.T @ X).ravel()   # u = -K s
        self.env = env
        self.ki, self.k_yaw = ki, k_yaw
        self.tau_max = tau_max
        self.int_v = np.zeros(2)
        self.vref = np.zeros(2)
        self.a_max = a_max
        self._last_step = -1
        self.dt = float(env.model.opt.timestep) * float(getattr(env, 'action_repeat', 5))   # control period
        self.sensor = None      # optional realistic sensing layer (ballbot_gym.controllers.realistic.SensorModel)

    def _state(self):
        e = self.env
        m, d = e.model, e.data
        bq = m.joint("base_free_joint").qposadr[0]
        bd = m.joint("base_free_joint").dofadr[0]
        kd = m.joint("ball_free_joint").dofadr[0]
        R = quaternion.as_rotation_matrix(quaternion.from_float_array(d.qpos[bq + 3:bq + 7]))
        up = R[:, 2]                                     # body z axis in world
        tilt_y, tilt_x = np.arctan2(up[0], up[2]), np.arctan2(up[1], up[2])   # lean toward +x, toward +y
        w_body = d.qvel[bd + 3:bd + 6]                   # MuJoCo free joint: angular velocity (body frame)
        w_body_w = R @ w_body
        w_ball = d.qvel[kd + 3:kd + 6]
        w_ball_w = m.body("ball").id and w_ball          # ball free joint angular velocity (local == world here, ball frame)
        Rb = quaternion.as_rotation_matrix(quaternion.from_float_array(d.qpos[m.joint("ball_free_joint").qposadr[0] + 3:m.joint("ball_free_joint").qposadr[0] + 7]))
        w_ball_w = Rb @ w_ball
        v_ball = d.qvel[kd:kd + 3]                      # ball centre linear velocity (world)
        return tilt_x, tilt_y, w_body_w, w_ball_w, R, v_ball

    def law(self, i, s):
        """Planar control law: torque on the ball for plane i given s=[0, tilt, ball-rate error, tilt-rate]. Override in subclasses."""
        return float(-self.K @ s)

    def predict(self, obs, deterministic=True):
        e = self.env
        if e.step_counter <= 1 or e.step_counter < self._last_step:
            self.int_v[:] = 0
            self.vref[:] = 0
        self._last_step = e.step_counter
        cmd = np.asarray(obs.get("cmd", np.zeros(3)), dtype=float)
        tilt_x, tilt_y, wb, wk, R, vb = self._state()
        if self.sensor is not None:
            tilt_x, tilt_y, wb, wk, obs = self.sensor(tilt_x, tilt_y, wb, wk, obs, reset=e.step_counter <= 1)
        # reference governor: move the velocity reference toward the command at a feasible acceleration
        dv = cmd[:2] - self.vref
        nv = np.linalg.norm(dv)
        if nv > self.a_max * self.dt:
            dv *= self.a_max * self.dt / nv
        self.vref += dv
        # plane "x" (motion along +x): rotation about +y.  plane "y" (motion along +y): rotation about -x.
        # rolling-equivalent ball rate from the measured ball velocity (robust to contact creep)
        planes = [(tilt_y, wb[1], wk[1], self.vref[0]), (tilt_x, -wb[0], -wk[0], self.vref[1])]
        v_meas = np.asarray(obs["vel"][:2], dtype=float)
        taus = []
        for i, (th, thd, phd, vref) in enumerate(planes):
            # outer loop: slow integral on the measured body velocity corrects the rolling-rate reference
            self.int_v[i] = np.clip(self.int_v[i] + (vref - v_meas[i]) * self.dt, -2, 2)
            phd_ref = (vref + self.ki * self.int_v[i]) / self.rball
            s = np.array([0.0, th, phd - phd_ref, thd])
            taus.append(self.law(i, s))
        tau_y_axis, tau_minus_x_axis = taus
        tau_z = -self.k_yaw * (cmd[2] - wb[2])          # reaction on the body spins it opposite to the ball
        T = np.array([-tau_minus_x_axis, tau_y_axis, tau_z])   # torque on the ball (world axes), relative to body
        # wheel torques: ball torque = W^T tau_w (virtual power); W is expressed in the body frame at upright
        Tb = R.T @ T
        tau_w = np.linalg.solve(self.W.T, Tb)
        wmax = e.max_wheel_velocity
        act = -tau_w / wmax                             # env applies ctrl = -action * wmax
        return np.clip(act, -1, 1).astype(np.float32), None
