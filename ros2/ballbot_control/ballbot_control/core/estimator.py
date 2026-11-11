"""State estimator for one robot: IMU (gyro, accelerometer) + three wheel encoders -> RobotState for the controller and odometry.

Filter: experiments/warp_ppo/odometry.py BallbotOdometry (same equations, one robot):
  attitude   Mahony-style complementary filter (gyro integration, accelerometer tilt correction with gain k_acc); yaw drifts
  gyro bias  slow average of the gyro while the robot is still (zero-velocity update)
  ball rate  w_rel = C @ wheel_rate (calibration matrix C); ball world rate w_k = R (w_rel + gyro); rolling: v = (w_k,y r, -w_k,x r)
  Kalman     (x, y, vx, vy) of the ball centre: predict with the tilt-compensated accelerometer, update with the encoder velocity
Added here for the controller:
  goal point        track_point "ball_centre" (goal_centre runs, g39 and later): the Kalman ball-centre position and velocity directly.
                    track_point "ball_body_origin" (older runs): the ball orientation is integrated from w_k (starting at identity, as
                    in every simulator episode) to reproduce the simulator's ball BODY origin, 0.14 m from the centre (nominal)
  point velocities  base origin and base-subtree COM velocities from the ball-centre velocity and the base rotation (rigid offsets)
"""
import numpy as np

from .math3d import expm_so3, mat_to_quat, orthonormalise
from .state import RobotState

G = np.array([0.0, 0.0, -9.81])


class BallbotOdometry:
    def __init__(self, C, ball_r, dt=0.01, k_acc=0.5, still_w=0.05, still_g=0.03, bias_tau=2.0, q_acc=0.5, r_enc=0.02, R0=None, p0=None):
        self.C, self.r, self.dt = np.asarray(C, dtype=np.float64), float(ball_r), float(dt)
        self.k_acc, self.still_w, self.still_g, self.bias_tau = k_acc, still_w, still_g, bias_tau
        self.q_acc, self.r_enc = q_acc, r_enc
        self.R = np.eye(3) if R0 is None else np.array(R0, dtype=np.float64)
        self.bg = np.zeros(3)
        self.x = np.zeros(4) if p0 is None else np.concatenate([np.asarray(p0, dtype=np.float64), np.zeros(2)])
        self.P = np.diag([1e-6, 1e-6, 1e-2, 1e-2])
        dt = self.dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.B = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        self.H = np.array([[0, 0, 1, 0], [0, 0, 0, 1.0]])
        self.w = np.zeros(3)          # bias-corrected body rate of the last step
        self.wk = np.zeros(3)         # ball angular velocity, world, of the last step

    def step(self, wheel_rate, gyro, acc):
        dt = self.dt
        wheel_rate, gyro, acc = (np.asarray(v, dtype=np.float64) for v in (wheel_rate, gyro, acc))
        still = (np.abs(wheel_rate).max() < self.still_w) and (np.abs(gyro).max() < self.still_g)
        if still:
            a = dt / self.bias_tau
            self.bg = (1 - a) * self.bg + a * gyro
        w = gyro - self.bg
        g_body_meas = -acc / max(np.linalg.norm(acc), 1e-6)
        g_body_pred = self.R.T @ np.array([0.0, 0.0, -1.0])
        corr = self.k_acc * np.cross(g_body_meas, g_body_pred)
        self.R = orthonormalise(self.R @ expm_so3((w + corr) * dt))
        w_rel = self.C @ wheel_rate
        wk = self.R @ (w_rel + w)
        v_enc = np.array([wk[1] * self.r, -wk[0] * self.r])
        a_world = self.R @ acc + G
        self.x = self.F @ self.x + self.B @ a_world[:2]
        Q = self.q_acc ** 2 * (self.B @ self.B.T)
        self.P = self.F @ self.P @ self.F.T + Q
        S = self.H @ self.P @ self.H.T + self.r_enc ** 2 * np.eye(2)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (v_enc - self.x[2:])
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.w, self.wk = w, wk
        return self.x[:2].copy(), self.x[2:].copy(), v_enc

    def yaw(self):
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))


class StateEstimator:
    """Odometry plus the derived quantities the controller observes (RobotState), using the bundle's kinematic offsets."""

    def __init__(self, est_cfg, kin, dt):
        self.cfg, self.kin, self.dt = est_cfg, kin, float(dt)
        self.r_origin = np.asarray(kin["base_origin_from_ball_centre_body"], dtype=np.float64)
        self.r_com = np.asarray(kin["base_com_from_ball_centre_body"], dtype=np.float64)
        self.ball_off = np.asarray(kin["ball_body_origin_from_centre_ball"], dtype=np.float64)
        self.track = est_cfg.get("track_point") or "ball_body_origin"
        if self.track not in ("ball_body_origin", "ball_centre"):
            raise ValueError("track_point must be 'ball_centre' (goal_centre runs) or 'ball_body_origin' (runs before S118)")
        self.reset()

    def reset(self, R0=None, p0=None):
        c = self.cfg
        self.odo = BallbotOdometry(c["C"], self.kin["ball_radius"], dt=self.dt, k_acc=c["k_acc"], still_w=c["still_w"], still_g=c["still_g"],
                                   bias_tau=c["bias_tau"], q_acc=c["q_acc"], r_enc=c["r_enc"], R0=R0, p0=p0)
        self.Rk = np.eye(3)          # ball orientation (dead-reckoned)
        self.q_prev = None

    def step(self, wheel_rate, gyro, acc):
        pos_c, vel_c, _ = self.odo.step(wheel_rate, gyro, acc)
        R, w, wk = self.odo.R, self.odo.w, self.odo.wk
        if self.track == "ball_body_origin":
            self.Rk = orthonormalise(expm_so3(wk * self.dt) @ self.Rk)
        q = mat_to_quat(R, self.q_prev)
        self.q_prev = q
        om_w = R @ w
        vc3 = np.array([vel_c[0], vel_c[1], 0.0])
        v_origin = vc3 + np.cross(om_w, R @ self.r_origin)
        v_com = vc3 + np.cross(om_w, R @ self.r_com)
        if self.track == "ball_body_origin":
            off_w = self.Rk @ self.ball_off
            p_track = pos_c + off_w[:2]
            v_track = vel_c + np.cross(wk, off_w)[:2]
        else:
            # goal point = ball centre, no orientation dead-reckoning. The authority dead band read the ball BODY origin velocity
            # in training; here it gets the centre velocity (a deliberate approximation, see README open issues).
            p_track, v_track = pos_c, vel_c
        st = RobotState(quat=q, omega_body=w.copy(), v_com=v_com, v_origin=v_origin, ball_omega=wk.copy(), ball_pos=p_track, ball_vel=v_track)
        return st, dict(pos=pos_c, vel=vel_c, R=R.copy(), yaw=self.odo.yaw(), P=self.odo.P.copy(), gyro_bias=self.odo.bg.copy())
