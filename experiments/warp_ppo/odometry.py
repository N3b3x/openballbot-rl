"""Ballbot state estimator from on-board sensors only: three wheel encoders and a 6-axis IMU (simulation study; numpy, vectorised over
robots so one call evaluates many logged worlds, but every operation is per robot and runs as-is on one robot).

Pipeline per 10 ms step (the same structure a real robot would run on its main computer):
  1. attitude   Mahony-style complementary filter: integrate the bias-corrected gyro, pull roll and pitch toward the accelerometer's
                gravity direction (gain k_acc). Yaw has no absolute reference (no magnetometer, no vision), so it drifts with the
                remaining gyro bias.
  2. gyro bias  estimated only while the robot is still (all wheel rates and the gyro below thresholds): a slow average of the gyro.
                Zero-velocity updates are the standard way to observe IMU bias without an external reference.
  3. ball rate  ball rotation relative to the body from the encoders, w_rel = C @ wheel_rate, with C a calibration matrix
                (calibrate() fits it from a log; the planar model's W^-1 is the uncalibrated start, off by a factor of about 0.65,
                log S108). The ball's world rotation is R (w_rel + gyro) and, for rolling without slip, the ball centre moves at
                v = (w_y r, -w_x r).
  4. Kalman     state (x, y, vx, vy) per robot: predict with the horizontal acceleration from the accelerometer (tilt-compensated,
                gravity removed), update with the encoder velocity of step 3. The filter smooths encoder quantisation and noise;
                it cannot remove a scale or heading error, which is what makes position drift.
Outputs: position, velocity, roll, pitch, yaw, and the filter covariance.
"""
import numpy as np

G = np.array([0.0, 0.0, -9.81])


def _skew(v):
    z = np.zeros(v.shape[:-1])
    return np.stack([np.stack([z, -v[..., 2], v[..., 1]], -1), np.stack([v[..., 2], z, -v[..., 0]], -1), np.stack([-v[..., 1], v[..., 0], z], -1)], -2)


def _expm_so3(phi):
    th = np.linalg.norm(phi, axis=-1, keepdims=True)[..., None]
    K = _skew(phi / np.maximum(th[..., 0], 1e-12))
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _orthonormalise(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


class BallbotOdometry:
    def __init__(self, n, C, ball_r, dt=0.01, k_acc=0.5, still_w=0.05, still_g=0.03, bias_tau=2.0,
                 q_acc=0.5, r_enc=0.02, R0=None, p0=None):
        self.n, self.C, self.r, self.dt = n, np.asarray(C), ball_r, dt
        self.k_acc, self.still_w, self.still_g, self.bias_tau = k_acc, still_w, still_g, bias_tau
        self.q_acc, self.r_enc = q_acc, r_enc
        self.R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy() if R0 is None else R0.copy()
        self.bg = np.zeros((n, 3))
        self.x = np.zeros((n, 4)) if p0 is None else np.concatenate([p0, np.zeros((n, 2))], 1)
        self.P = np.broadcast_to(np.diag([1e-6, 1e-6, 1e-2, 1e-2]), (n, 4, 4)).copy()
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.B = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.H = np.array([[0, 0, 1, 0], [0, 0, 0, 1.0]])

    def step(self, wheel_rate, gyro, acc):
        dt = self.dt
        # 2. gyro bias from zero-velocity periods
        still = (np.abs(wheel_rate).max(1) < self.still_w) & (np.abs(gyro).max(1) < self.still_g)
        a = dt / self.bias_tau
        self.bg = np.where(still[:, None], (1 - a) * self.bg + a * gyro, self.bg)
        w = gyro - self.bg
        # 1. attitude: gyro integration + accelerometer tilt correction
        g_body_meas = -acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-6)       # direction of gravity in the body frame (specific force points up)
        g_body_pred = np.einsum("nji,j->ni", self.R, np.array([0, 0, -1.0]))
        corr = self.k_acc * np.cross(g_body_meas, g_body_pred)
        self.R = _orthonormalise(self.R @ _expm_so3((w + corr) * dt))
        # 3. ball rate from the encoders
        w_rel = wheel_rate @ self.C.T
        wk = np.einsum("nij,nj->ni", self.R, w_rel + w)
        v_enc = np.stack([wk[:, 1] * self.r, -wk[:, 0] * self.r], 1)
        # 4. Kalman filter on (x, y, vx, vy)
        a_world = np.einsum("nij,nj->ni", self.R, acc) + G
        self.x = self.x @ self.F.T + a_world[:, :2] @ self.B.T
        Q = self.q_acc ** 2 * (self.B @ self.B.T)
        self.P = self.F @ self.P @ self.F.T + Q
        S = self.H @ self.P @ self.H.T + self.r_enc ** 2 * np.eye(2)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innov = v_enc - self.x[:, 2:]
        self.x = self.x + np.einsum("nij,nj->ni", K, innov)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:, :2], self.x[:, 2:], v_enc

    def yaw(self):
        return np.arctan2(self.R[:, 1, 0], self.R[:, 0, 0])


def true_ball_rel_rate(log, sl=slice(None)):
    """Ground truth: ball angular velocity relative to the base, in the base frame (T, N, 3)."""
    from scipy.spatial.transform import Rotation as Rot
    q, qk = log["quat"][sl], log["ball_quat"][sl]
    T_, N_ = q.shape[:2]
    Rb = Rot.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    Rk = Rot.from_quat(qk.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    wk = np.einsum("nij,nj->ni", Rk, log["ball_w"][sl].reshape(-1, 3))
    wb = np.einsum("nij,nj->ni", Rb, log["gyro"][sl].reshape(-1, 3))
    return np.einsum("nji,nj->ni", Rb, wk - wb).reshape(T_, N_, 3)


def calibrate(log, worlds):
    """Least-squares calibration C (3x3) with w_rel = C @ wheel_rate, fitted on the given worlds while alive."""
    wr = true_ball_rel_rate(log)[:, worlds]
    ww = log["wheel_w"][:, worlds]
    m = log["alive"][:, worlds]
    A, *_ = np.linalg.lstsq(ww[m], wr[m], rcond=None)
    return A.T
