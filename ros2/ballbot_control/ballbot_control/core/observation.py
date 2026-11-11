"""Policy observation and actor-input vector, reproducing GpuBallbot.obs / reset_obs / noisy / step of train_dev.py.

25-D observation (order and scales from the bundle's observation.layout):
   0:3   rotvec      rotation vector of the base quaternion (includes yaw; w < 0 is not wrapped)
   3:6   ang_vel     base angular velocity, world frame, clipped to +-8
   6:9   lin_vel     base-subtree COM velocity, world frame, clipped to +-8
   9:12  motor       (v_origin_y, v_origin_z, omega_body_x) / tau, clipped to +-2. This is a quirk of the training environment, kept
                     on purpose: it reads qvel[1:4] of the base free joint, not the wheel rates. The policy was trained on it.
  12:15  last_action previous policy action (clipped to [-1, 1]; the raw residual, not the applied action)
  15:18  cmd         shaped command (vx, vy world, wz)
  18:20  heading     sin(yaw), cos(yaw)
  20:22  vint        velocity-error integral, clip(vint + (cmd_xy - lin_vel_xy) dt, +-1)
  22:25  goal        clip((goal - ball_pos) / 2, +-2) * goal_on, goal_on

Actor input (255-D): [obs (25), history (hist_k x (25 + 3)), design vector (6)]. History entry j (j = 0 newest) is the
(observation, action) pair of control tick t-1-j: the observation the policy saw at that tick and the action it returned.
The training vector also carries 46 privileged critic columns after these; the actor never reads them, so they are not built.

Episode start: the first tick after reset() uses the simulator's reset observation (ang_vel, lin_vel and motor zero, vint not yet
integrated), exactly as every training episode started. The controller only resets while the robot is held still, so this is exact.
"""
import numpy as np

from .math3d import quat_to_mat, rotvec, yaw_of


class ObservationBuilder:
    def __init__(self, obs_cfg, tau, dvec, dt):
        self.cfg = obs_cfg
        self.obs_dim, self.act_dim, self.hist_k = int(obs_cfg["obs_dim"]), int(obs_cfg["act_dim"]), int(obs_cfg["hist_k"])
        self.vel_clip, self.motor_clip = float(obs_cfg["vel_clip"]), float(obs_cfg["motor_clip"])
        self.vint_clip, self.goal_scale, self.goal_clip = float(obs_cfg["vint_clip"]), float(obs_cfg["goal_scale"]), float(obs_cfg["goal_clip"])
        self.tau, self.dt = float(tau), float(dt)
        self.dvec = np.asarray(dvec, dtype=np.float64)
        self.reset()

    @property
    def actor_dim(self):
        return self.obs_dim + self.hist_k * (self.obs_dim + self.act_dim) + len(self.dvec)

    def reset(self):
        self.hist = np.zeros((self.hist_k, self.obs_dim + self.act_dim))
        self.last_a = np.zeros(self.act_dim)
        self.vint = np.zeros(2)
        self.prev_obs = None          # observation of the previous tick (None right after reset)

    def build(self, s, cmd, goal=None, goal_on=False):
        """Observation (25,) and actor input (actor_dim,) for this tick. s: RobotState; cmd (3,) shaped command; goal (2,) or None."""
        first = self.prev_obs is None
        if not first:      # shift in the (observation, action) pair of the previous tick, newest first
            self.hist = np.concatenate([np.concatenate([self.prev_obs, self.last_a])[None], self.hist[:-1]], 0)
        q = np.asarray(s.quat, dtype=np.float64)
        cmd = np.asarray(cmd, dtype=np.float64)
        if first:
            ang = np.zeros(3); vel = np.zeros(3); motor = np.zeros(3)
        else:
            om_w = s.omega_world if s.omega_world is not None else quat_to_mat(q) @ np.asarray(s.omega_body, dtype=np.float64)
            ang = np.clip(om_w, -self.vel_clip, self.vel_clip)
            vel = np.clip(np.asarray(s.v_com, dtype=np.float64), -self.vel_clip, self.vel_clip)
            raw = np.array([s.v_origin[1], s.v_origin[2], s.omega_body[0]], dtype=np.float64)      # see module doc: qvel[1:4]
            motor = np.clip(raw / self.tau, -self.motor_clip, self.motor_clip)
            self.vint = np.clip(self.vint + (cmd[:2] - vel[:2]) * self.dt, -self.vint_clip, self.vint_clip)
        yaw = yaw_of(q)
        if goal_on and goal is not None:
            e = np.clip((np.asarray(goal, dtype=np.float64) - np.asarray(s.ball_pos, dtype=np.float64)) / self.goal_scale, -self.goal_clip, self.goal_clip)
            gf = np.array([e[0], e[1], 1.0])
        else:
            gf = np.zeros(3)
        o = np.concatenate([rotvec(q), ang, vel, motor, self.last_a, cmd, [np.sin(yaw), np.cos(yaw)], self.vint, gf])
        self.prev_obs = o
        x = np.concatenate([o, self.hist.ravel(), self.dvec])
        return o, x

    def commit_action(self, a):
        """Record the policy action of this tick (becomes last_action and enters the history next tick)."""
        self.last_a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
