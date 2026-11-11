"""Realistic sensing plus two more model-based baselines for the ballbot study.

SensorModel  : IMU tilt noise, gyro white noise and random-walk bias, wheel-encoder quantisation and noise, velocity-estimate noise,
               and a fixed sensing delay (control steps). Wraps the state read by LQRPolicy.
SMCPolicy    : sliding-mode control. LQR gain K plus a switching term on the sliding variable sigma = c s, where c is the left
               eigenvector of (A - B K) of its fastest mode (so sigma decays on its own and the sliding dynamics are the other two LQR poles).
               u = -K s - eta sgn(cB) sat(sigma / phi).
MPCPolicy    : constraint-aware linear MPC per plane (condensed QP, OSQP). Same model and weights as the discrete LQR, with hard
               input limits, soft tilt limits, and a model-based delay compensation of the measured state (Smith-style prediction).
Both subclass LQRPolicy, so the governor, outer integral, yaw loop and wheel mapping are identical; only the planar law differs.
"""
from collections import deque

import numpy as np
import scipy.linalg as la
from scipy.signal import cont2discrete

from .lqr import LQRPolicy

LEVELS = {
    # tilt_std [rad], gyro_std [rad/s], gyro_bias [rad/s] (1 sigma, per episode), enc_std [rad/s], enc_quant [rad/s], vel_std [m/s], delay [steps of 10 ms]
    "none": dict(tilt_std=0.0, gyro_std=0.0, gyro_bias=0.0, enc_std=0.0, enc_quant=0.0, vel_std=0.0, delay=0),
    "mild": dict(tilt_std=0.003, gyro_std=0.01, gyro_bias=0.005, enc_std=0.05, enc_quant=0.05, vel_std=0.01, delay=1),
    "hard": dict(tilt_std=0.01, gyro_std=0.03, gyro_bias=0.02, enc_std=0.2, enc_quant=0.2, vel_std=0.03, delay=2),
}


class SensorModel:
    def __init__(self, level="mild", seed=0, **over):
        self.p = {**LEVELS[level], **over}
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.bias = self.rng.normal(0, self.p["gyro_bias"], 3)
        self.buf = deque(maxlen=int(self.p["delay"]) + 1)

    def __call__(self, tx, ty, wb, wk, obs, reset=False):
        p, r = self.p, self.rng
        if reset:
            self.reset()
        tx, ty = tx + r.normal(0, p["tilt_std"]), ty + r.normal(0, p["tilt_std"])
        wb = wb + self.bias + r.normal(0, p["gyro_std"], 3)
        wk = wk + r.normal(0, p["enc_std"], 3)
        if p["enc_quant"] > 0:
            wk = np.round(wk / p["enc_quant"]) * p["enc_quant"]
        obs = dict(obs)
        obs["vel"] = np.asarray(obs["vel"], dtype=float) + r.normal(0, p["vel_std"], np.shape(obs["vel"]))
        rec = (tx, ty, wb, wk, obs["vel"])
        self.buf.append(rec)
        tx, ty, wb, wk, vel = self.buf[0]            # oldest = delayed by len-1 steps (fills up after reset)
        obs["vel"] = vel
        return tx, ty, wb, wk, obs


class SMCPolicy(LQRPolicy):
    def __init__(self, env, eta=0.25, phi=0.05, **kw):
        super().__init__(env, **kw)
        Acl = self.A - self.B @ self.K[None, :]
        w, V = la.eig(Acl.T)                         # left eigenvectors of Acl
        w = w.real
        i = int(np.argmin(w))                        # fastest (most negative) stable mode
        c = V[:, i].real
        self.c = c / np.abs(c @ self.B).max() if abs(c @ self.B).max() > 0 else c
        self.sgn = float(np.sign((self.c @ self.B).item()))
        self.eta, self.phi = eta, phi

    def law(self, i, s):
        sigma = float(self.c @ s)
        return float(-self.K @ s - self.eta * self.sgn * np.clip(sigma / self.phi, -1, 1))


class MPCPolicy(LQRPolicy):
    def __init__(self, env, horizon=25, theta_max_deg=12.0, u_frac=1.0, delay=0, q_theta=50.0, q_phidot=1.0, q_thetadot=1.0, r=1.0, **kw):
        super().__init__(env, q_theta=q_theta, q_phidot=q_phidot, q_thetadot=q_thetadot, r=r, **kw)
        import osqp
        import scipy.sparse as sp
        dt, N = self.dt, int(horizon)
        A3, B3 = self.A[1:, 1:], self.B[1:]
        Ad, Bd, *_ = cont2discrete((A3, B3, np.eye(3), np.zeros((3, 1))), dt)
        Q = np.diag([q_theta, q_phidot, q_thetadot]) * dt
        R = np.array([[r * dt]])
        P = la.solve_discrete_are(Ad, Bd, Q, R)
        self.Ad, self.Bd, self.N = Ad, Bd, N
        # prediction matrices: s_k = Phi_k s0 + sum_j G_kj u_j  (k = 1..N)
        Phi = np.zeros((3 * N, 3))
        G = np.zeros((3 * N, N))
        Apow = np.eye(3)
        for k in range(N):
            Apow = Ad @ Apow
            Phi[3 * k:3 * k + 3] = Apow
            for j in range(k + 1):
                G[3 * k:3 * k + 3, j] = (np.linalg.matrix_power(Ad, k - j) @ Bd).ravel()
        Qb = np.kron(np.eye(N), Q)
        Qb[-3:, -3:] = P
        Rb = np.kron(np.eye(N), R)
        H = G.T @ Qb @ G + Rb
        self.fq = G.T @ Qb @ Phi
        self.Phi, self.G = Phi, G
        # limits
        W = np.array(self.W)
        wmax = float(env.max_wheel_velocity)
        self.u_max = u_frac * wmax * float(np.linalg.svd(W.T, compute_uv=False).min())
        self.th_max = np.deg2rad(theta_max_deg)
        Gth = G[0::3]                                 # tilt rows
        self.Phith = Phi[0::3]
        nv = N + N                                    # u and slack
        Pm = sp.block_diag([sp.csc_matrix(H), 1e-3 * sp.eye(N)], format="csc")
        A_ = sp.vstack([sp.hstack([sp.csc_matrix(Gth), -sp.eye(N)]), sp.hstack([sp.csc_matrix(-Gth), -sp.eye(N)]),
                        sp.hstack([sp.eye(N), sp.csc_matrix((N, N))]), sp.hstack([sp.csc_matrix((N, N)), sp.eye(N)])], format="csc")
        self._qslack = 1e4 * np.ones(N)
        self._l0 = np.concatenate([-np.inf * np.ones(2 * N), -self.u_max * np.ones(N), np.zeros(N)])
        self.qp = osqp.OSQP()
        self.qp.setup(Pm, np.zeros(nv), A_, self._l0, np.concatenate([np.zeros(2 * N), self.u_max * np.ones(N), np.inf * np.ones(N)]),
                      verbose=False, eps_abs=1e-5, eps_rel=1e-5, polish=False, max_iter=400)
        self.delay = int(delay)
        self._hist = [deque([0.0] * self.delay, maxlen=max(self.delay, 1)) for _ in range(2)]
        self.last_u = [0.0, 0.0]

    def law(self, i, s):
        s3 = np.asarray(s[1:], dtype=float)
        for u in self._hist[i]:                       # delay compensation: roll the state forward over the commands still in flight
            s3 = self.Ad @ s3 + self.Bd.ravel() * u
        N = self.N
        q = np.concatenate([self.fq @ s3, self._qslack])
        ub = np.concatenate([self.th_max - self.Phith @ s3, self.th_max + self.Phith @ s3, self.u_max * np.ones(N), np.inf * np.ones(N)])
        # rows: [G u - e <= th_max - Phi s] ; [-G u - e <= th_max + Phi s]
        self.qp.update(q=q, l=self._l0, u=ub)
        res = self.qp.solve()
        u = float(res.x[0]) if res.x is not None and np.isfinite(res.x[0]) else float(-self.K @ s)
        if self.delay:
            self._hist[i].append(u)
        return u
