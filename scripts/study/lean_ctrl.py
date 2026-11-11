"""Hand-designed cascaded controller: velocity error -> lean setpoint -> PD balance -> wheel torques.
Used as a physical ceiling check and as a teacher for RL."""
import sys
from pathlib import Path
import numpy as np
import quaternion
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ballbot_rl.training.utils import make_ballbot_env  # noqa: E402
import ballbot_gym  # noqa: F401,E402

TH = np.deg2rad([0, 120, 240])


class LeanCtrl:
    def __init__(self, wmax, kp=20.0, kd=1.5, ki=0.0, kv=0.15, kvi=0.0, tilt_max=0.25, wz_k=1.0, sgn=(1, -1), dt=0.01, gs=1.0):
        self.gs = gs
        self.wmax, self.kp, self.kd, self.ki, self.kv, self.kvi = wmax, kp, kd, ki, kv, kvi
        self.tilt_max, self.wz_k, self.sgn, self.dt = tilt_max, wz_k, sgn, dt
        self.vint = np.zeros(2)
        self.iint = np.zeros(2)

    def act(self, obs, cmd):
        R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(obs["orientation"][-3:]))
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
        w_loc = R.T @ obs["angular_vel"]            # body-frame angular velocity
        v = obs["vel"][:2]
        verr = np.asarray(cmd[:2]) - v
        self.vint = np.clip(self.vint + verr * self.dt, -1, 1)
        a_w = self.kv * verr + self.kvi * self.vint  # desired lean (rad) in world frame
        a_l = R.T @ np.array([a_w[0], a_w[1], 0.0])
        a_l = np.clip(a_l[:2], -self.tilt_max, self.tilt_max)
        pitch_sp, roll_sp = self.sgn[0] * a_l[0], self.sgn[1] * a_l[1]
        e_p, e_r = pitch_sp - pitch, roll_sp - roll
        self.iint = np.clip(self.iint + np.array([e_p, e_r]) * self.dt, -0.3, 0.3)
        u_p = self.kp * e_p + self.ki * self.iint[0] - self.kd * w_loc[1]
        u_r = self.kp * e_r + self.ki * self.iint[1] - self.kd * w_loc[0]
        yaw_t = self.wz_k * (cmd[2] - w_loc[2])      # equal torque on all wheels spins the body
        ctrl = self.gs * (u_r * np.cos(TH) + u_p * np.sin(TH)) + yaw_t
        return np.clip(ctrl / self.wmax, -1, 1).astype(np.float32), (pitch, roll)


def make_env(conf="ctrl_ball_free_d", fixed=True):
    cfg = yaml.safe_load(open(ROOT / "configs/env" / f"{conf}.yaml"))
    cfg["camera"]["enabled"] = False
    cfg["env"]["cmd_resample_steps"] = 10 ** 9
    ec = {k: cfg[k] for k in ("camera", "env", "logging")}
    return make_ballbot_env(terrain_config=cfg["terrain"], reward_config=cfg["reward"], env_config=ec, seed=0, eval_env=True)(), cfg["env"]["max_wheel_velocity"]


def run(ctrl_kwargs, cmd, secs=4.0, seed=0, conf="ctrl_ball_free_d"):
    env, wmax = make_env(conf)
    b = env.unwrapped
    b.cmd_fixed = np.zeros(3)
    obs, _ = env.reset(seed=seed)
    c = LeanCtrl(wmax, **ctrl_kwargs)
    cmd = np.asarray(cmd, float)
    vs, tl, off = [], [], []
    for t in range(int(secs * 100)):
        b.cmd = cmd; obs["cmd"] = cmd.astype(np.float32)
        a, (p, r) = c.act(obs, cmd)
        obs, rew, te, tr, info = env.step(a)
        vs.append(np.r_[obs["vel"][:2], obs["angular_vel"][2]]); tl.append(np.hypot(p, r)); off.append(info["off_ball_deg"])
        if te:
            break
    vs = np.array(vs)
    return dict(steps=len(vs), fell=bool(te), v=vs[len(vs)//2:].mean(0).round(2), tilt_deg=round(float(np.degrees(max(tl))), 1), off=round(max(off), 1))


if __name__ == "__main__":
    import itertools
    for gs, kp, kd in itertools.product((1, -1), (60, 150, 300), (3, 10, 25)):
        r = run(dict(kv=0.0, gs=gs, kp=kp, kd=kd), (0, 0, 0), secs=5)
        print(gs, kp, kd, r["steps"], r["tilt_deg"], r["off"], flush=True)
