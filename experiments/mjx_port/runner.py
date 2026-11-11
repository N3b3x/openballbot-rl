"""Backend-agnostic test runner. A backend exposes reset(qpos,qvel), set_ctrl(c), set_push(F), step(n), state()."""
import numpy as np
from common import *  # noqa

STEP_T0 = 0.5     # s settle before the torque step
STEP_TAU = np.array([2.0, 0.0, 0.0])
STEP_LEN = 0.3
STEP_TOTAL = 1.0


def run_settle(be, m, T=1.5):
    q, v = reset_state(m)
    be.reset(q, v)
    out = []
    for _ in range(int(T / 0.01)):
        be.step(ACTION_REPEAT)
        qp, qv, cv, bx, gx = be.state()
        out.append(np.concatenate([qp[0, [2]], qp[0, [KQ + 2]], [tilt_deg(qp[0])], qv[0, KD:KD + 3], qp[0, WQ:WQ + 3]]))
    return np.array(out)    # cols: base z, ball z, tilt, ball vx vy vz, wheel angles


def run_step(be, m, tau_scale=1.0):
    q, v = reset_state(m)
    be.reset(q, v)
    out = []
    n = int(STEP_TOTAL / 0.01)
    for k in range(n):
        t = k * 0.01
        be.set_ctrl(STEP_TAU * tau_scale if STEP_T0 <= t < STEP_T0 + STEP_LEN else np.zeros(3))
        be.step(ACTION_REPEAT)
        qp, qv, cv, bx, gx = be.state()
        out.append(np.concatenate([qv[0, KD:KD + 3], qv[0, WD:WD + 3], [tilt_deg(qp[0])], qp[0, KQ:KQ + 2]]))
    return np.array(out)    # ball vx vy vz, wheel w0..2, tilt, ball x y


def run_lqr(be, m, spec, fall_tilt=70.0, fall_off=30.0, record=True, yaw=0.0, design=None):
    """Closed-loop LQR over N worlds. Returns dict(vel (T,N,2) base lin vel (cvel), tilt (T,N), pos (T,N,2), fell (N,), fall_t (N,))."""
    N = spec["cmd"].shape[0]
    cmd = spec["cmd"]
    q, v = reset_state(m, yaw)
    be.reset(q, v)
    ctl = LQRBatch(N, model_json=design.json, tau_max=design.tau) if design else LQRBatch(N)
    vel = np.zeros((spec["steps"], N, 2)); til = np.zeros((spec["steps"], N)); pos = np.zeros((spec["steps"], N, 2))
    fell = np.zeros(N, bool); fall_t = np.full(N, np.nan)
    ps = spec.get("push_steps")
    qp, qv, cv, bx, gx = be.state()
    for t in range(spec["steps"]):
        tau = ctl(qp, qv, cv[:, 3:6], cmd)
        be.set_ctrl(tau)
        if ps is not None:
            be.set_push(spec["push"] if ps[0] <= t < ps[1] else np.zeros((N, 3)))
        be.step(ACTION_REPEAT)
        qp, qv, cv, bx, gx = be.state()
        vel[t] = cv[:, 3:5]; til[t] = tilt_deg(qp); pos[t] = qp[:, :2]
        bad = (til[t] > fall_tilt) | (off_ball_deg(bx, gx) > fall_off) | ~np.isfinite(qp).all(1)
        newf = bad & ~fell
        fall_t[newf] = t * 0.01
        fell |= bad
    return dict(vel=vel, tilt=til, pos=pos, fell=fell, fall_t=fall_t)


def summarize(res, cmd, tail=150):
    """speed ratio like eval_design.py: mean over non-fallen runs of <v_tail_mean, cmd>/<cmd,cmd>."""
    ratios = []
    for i in range(cmd.shape[0]):
        if not res["fell"][i]:
            vm = res["vel"][-tail:, i].mean(0)
            ratios.append(float(vm @ cmd[i, :2] / (cmd[i, :2] @ cmd[i, :2])))
    return (float(np.mean(ratios)) if ratios else float("nan")), int(res["fell"].sum())
