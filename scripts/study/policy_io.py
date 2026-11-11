"""Load a controller for the benchmark scripts: a trained PPO run (with its observation normaliser), the
model-based LQR baseline (run name "lqr"), or either wrapped in an outer velocity integral ("<run>@ki=<gain>").

The outer integral is the classical cascade fix for a steady-state velocity gain below one: the command passed to
the inner controller is c' = c + ki * integral(c - v) dt (planar components only, clipped).
"""
from pathlib import Path

import numpy as np

from evaluate import _load_policy, _normalize

ROOT = Path(__file__).resolve().parents[2]


class CmdIntegral:
    def __init__(self, inner, norm, env, ki, lim=1.5, vcap=1.5):
        self.inner, self.norm, self.env, self.ki, self.lim, self.vcap = inner, norm, env.unwrapped, ki, lim, vcap
        self.acc = np.zeros(2)
        self._last = -1

    def predict(self, obs, deterministic=True):
        e = self.env
        if e.step_counter <= 1 or e.step_counter < self._last:
            self.acc[:] = 0
        self._last = e.step_counter
        cmd = np.asarray(obs["cmd"], dtype=float).copy()
        err = cmd[:2] - np.asarray(obs["vel"][:2], dtype=float)
        c2 = cmd.copy()
        c2[:2] = cmd[:2] + self.ki * self.acc
        sp = float(np.linalg.norm(c2[:2]))
        if sp > self.vcap:                       # saturate the corrected command and freeze the integral (anti-windup; vcap = the policy's trained command range)
            c2[:2] *= self.vcap / sp
        else:
            self.acc = np.clip(self.acc + err * 0.01, -self.lim, self.lim)
        o = dict(obs)
        o["cmd"] = c2.astype(np.float32)
        return self.inner.predict(self.norm(o), deterministic=deterministic)


def load_controller(run, env, ckpt="final_model"):
    """Returns (controller, norm) where controller.predict(norm(obs), deterministic=True) gives an action."""
    ki = 0.0
    run_opts = ""
    if run.startswith("lqr@d"):
        run, run_opts = run.replace("@d", "", 1), "@d"
    if "@ki=" in run:
        run, k = run.split("@ki=")
        ki = float(k)
    if run == "lqr":
        from lqr_ctrl import LQRPolicy
        mj = getattr(env.unwrapped, "_env_settings_lqr_json", None)
        kw = {"model_json": mj} if mj else {}
        if "@d" in run_opts:      # discrete-time LQR at the control period (light rotors)
            kw["discrete_dt"] = 0.01
        ctrl, norm = LQRPolicy(env.unwrapped, **kw), (lambda obs: obs)
    else:
        run_dir = ROOT / "outputs/study/runs" / run
        model, vn = _load_policy(run_dir if ckpt == "final_model" else run_dir / "checkpoints", "proprio", ckpt)
        ctrl, norm = model, (lambda obs: _normalize(obs, vn))
    if ki:
        return CmdIntegral(ctrl, norm, env, ki), (lambda obs: obs)
    return ctrl, norm
