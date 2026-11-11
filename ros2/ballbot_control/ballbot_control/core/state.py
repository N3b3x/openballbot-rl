"""The measured robot state the tracking controller consumes once per control tick.

Frames: "world" is the odometry frame (z up, gravity along -z, origin and yaw fixed at estimator start); "body" is base_link (the
base frame of the MuJoCo model: z along the tower, origin at the base body origin). Units SI (m, m/s, rad, rad/s).

Each field states which simulator quantity it reproduces, because the trained policy observes those quantities, quirks included.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RobotState:
    quat: np.ndarray          # (4,) base orientation (w, x, y, z), body -> world               sim: qpos[3:7]
    omega_body: np.ndarray    # (3,) base angular velocity, body frame (bias-corrected gyro)    sim: qvel[3:6]
    v_com: np.ndarray         # (3,) velocity of the base-subtree centre of mass, world         sim: cvel[base][3:6]
    v_origin: np.ndarray      # (3,) velocity of the base frame origin, world                   sim: qvel[0:3]
    ball_omega: np.ndarray    # (3,) ball angular velocity, world                               sim: R_ball @ qvel[12:15]
    ball_pos: np.ndarray      # (2,) goal point, world xy (goal rule, goal observation, hold)   sim: GpuBallbot.ball_xy(): the ball centre
                              #      for goal_centre runs (g39 and later), else the ball BODY origin qpos[10:12]
    ball_vel: np.ndarray      # (2,) velocity read by the authority dead band, world xy         sim: qvel[9:11] (ball BODY origin, all runs)
    omega_world: Optional[np.ndarray] = None   # (3,) observed base angular velocity, world. None = R(quat) @ omega_body.
    # The simulator's observation reads cvel, which MuJoCo computes one 2 ms physics substep before the end of the control step, so
    # it differs slightly from R @ qvel[3:6]; the parity test passes the cvel value here. On hardware leave it None.

    @staticmethod
    def from_sim(qpos, qvel, cvel_base, goal_centre=False, ball_origin_from_centre=(0.0, 0.0, 0.14)):
        """Build the state from one world's MuJoCo arrays (qpos 17, qvel 15, cvel[base] 6) exactly as the simulator's controller sees it.
        goal_centre: goal point = ball centre = ball origin - R_ball @ ball_origin_from_centre (kinematics.ball_body_origin_from_centre_ball)."""
        from .math3d import quat_to_mat
        qpos, qvel, cvel_base = (np.asarray(v, dtype=np.float64) for v in (qpos, qvel, cvel_base))
        Rk = quat_to_mat(qpos[13:17])
        if goal_centre:
            ball_pos = (qpos[10:13] - Rk @ np.asarray(ball_origin_from_centre, dtype=np.float64))[:2]
        else:
            ball_pos = qpos[10:12].copy()
        return RobotState(quat=qpos[3:7].copy(), omega_body=qvel[3:6].copy(), v_com=cvel_base[3:6].copy(), v_origin=qvel[0:3].copy(),
                          ball_omega=Rk @ qvel[12:15], ball_pos=ball_pos, ball_vel=qvel[9:11].copy(),
                          omega_world=cvel_base[0:3].copy())
