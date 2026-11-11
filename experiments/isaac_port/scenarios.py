"""Engine-agnostic scenarios and metrics.  A backend provides reset(yaw), state(), step_ctrl(tau, push)."""
import numpy as np

from common import DT_CTRL, LQRCore, tilt_deg

FALL_DEG = 70.0   # same as env max_allowed_tilt
SPEEDS = (0.4, 0.8, 1.1)
ALPHAS = (0.5, 1.0, 1.5)


def run_drive(be, cfg, cmd, steps, push=None, push_at=200, push_n=5, rec=None, yaw=0.0, drop=0.0):
    """LQR drive at constant world command. push: world force vector held for push_n control steps (50 ms)."""
    ctrl = LQRCore(cfg)
    be.reset(yaw, drop)
    vel = []
    for t in range(steps):
        st = be.state()
        if tilt_deg(st["R"]) > FALL_DEG or not np.isfinite(st["pbody"]).all():
            return True, np.array(vel)
        tau = ctrl.act(st, cmd)
        F = push if (push is not None and push_at <= t < push_at + push_n) else None
        if rec is not None:
            rec(t, st, tau, F)
        be.step_ctrl(tau, F)
        vel.append(st["vel"][:2].copy())
    return False, np.array(vel)


def speed_table(be, cfg, yaw_dirs=4, steps=400, speeds=SPEEDS):
    row = {}
    for sp in speeds:
        ratios, falls = [], 0
        for k in range(yaw_dirs):
            ang = k * np.pi / 2
            cmd = sp * cfg.f * np.array([np.cos(ang), np.sin(ang), 0.0])
            fell, v = run_drive(be, cfg, cmd, steps)
            falls += fell
            if not fell:
                vm = v[-150:].mean(0)
                ratios.append(float(vm @ cmd[:2] / (cmd[:2] @ cmd[:2])))
        row[f"ratio_{sp}"] = float(np.mean(ratios)) if ratios else np.nan
        row[f"falls_{sp}"] = falls
    return row


def push_table(be, cfg, trials=8, alphas=ALPHAS, seed=0):
    rng = np.random.default_rng(seed)
    row = {}
    for al in alphas:
        falls = 0
        for k in range(trials):
            phi = rng.uniform(0, 2 * np.pi)
            F = al * cfg.m_total * 9.81 * np.array([np.cos(phi), np.sin(phi), 0.0])
            fell, _ = run_drive(be, cfg, np.array([0.0, 0.4 * cfg.f, 0.0]), 500, push=F)
            falls += fell
        row[f"pushfall_{al}"] = falls / trials
    return row


def cmd_step_response(be, cfg, steps=300, sp=0.4):
    """Velocity-command step 0 -> sp*f along +y under LQR (reference governor active)."""
    return run_drive(be, cfg, np.array([0.0, sp * cfg.f, 0.0]), steps)


def torque_pulse(be, cfg, tau0=0.5, dur=0.1, T=0.4):
    """Open loop (no feedback): zero torque 0.1 s, then a torque pulse on wheels 0 and 2 for dur, then zero.
    Returns rows [t, tilt_x, tilt_y, ball w (3, world), body w (3, world)] for T seconds."""
    be.reset(0.0)
    out = []
    for t in range(int(T / DT_CTRL)):
        st = be.state()
        on = 10 <= t < 10 + int(dur / DT_CTRL)
        tau = np.array([tau0, 0.0, -tau0]) * (1.0 if on else 0.0)
        up = st["R"][:, 2]
        out.append([st["t"], np.arctan2(up[1], up[2]), np.arctan2(up[0], up[2]), *st["wk_w"], *st["wb_w"]])
        be.step_ctrl(tau)
    return np.array(out)


def settle_trace(be, T=1.0):
    """Zero torque: ball/body settling.  Rows [t, ball centre z, body z, tilt deg, wheel-ball normal force proxy none]."""
    be.reset(0.0)
    out = []
    for t in range(int(T / DT_CTRL)):
        st = be.state()
        out.append([st["t"], st["pball"][2], st["pbody"][2], tilt_deg(st["R"])])
        be.step_ctrl(np.zeros(3))
    return np.array(out)
