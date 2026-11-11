"""Export the trained SB3 PPO policy (run robs_c_s0) to a compact JSON for the in-browser forward pass.

Writes site/live/policy_robs_c_s0.json with: observation key order, per-key sizes, which keys are
VecNormalize-normalised (mean, var, eps, clip), the MLP weights (pi network + action head), activation and the
action scaling used by BallbotEnv.step. Then verifies numerically against model.predict(deterministic=True)
on random observations (max abs error must be < 1e-5).

Run:  OPENBLAS_NUM_THREADS=1 python scripts/site/export_policy.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/study"))
RUN = "robs_c_s0"
OUT = ROOT / "site/live" / f"policy_{RUN}.json"


def main():
    import torch
    torch.set_num_threads(1)
    from evaluate import _load_policy, _normalize, PROPRIO_KEYS
    model, vn = _load_policy(ROOT / "outputs/study/runs" / RUN, "proprio", "final_model")
    pol = model.policy
    cfg = json.load(open(ROOT / "outputs/study/runs" / RUN / "run_config.json"))
    e = cfg["env"]["env"]
    keys = list(pol.features_extractor.extractors.keys())          # order in which features are concatenated
    sizes = {k: int(model.observation_space.spaces[k].shape[0]) for k in keys}
    assert pol.mlp_extractor.policy_net[1].__class__.__name__ == "Tanh"
    layers = []
    lin = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)] + [pol.action_net]
    for m in lin:
        layers.append({"W": m.weight.detach().double().numpy().round(9).tolist(),   # [out][in]
                       "b": m.bias.detach().double().numpy().round(9).tolist()})
    norm = {}
    for k in PROPRIO_KEYS:
        r = vn.obs_rms[k]
        norm[k] = {"mean": np.asarray(r.mean, float).round(9).tolist(), "var": np.asarray(r.var, float).round(9).tolist()}
    asp = model.action_space
    out = {
        "run": RUN, "source": "outputs/study/runs/robs_c_s0/{final_model.zip,vecnormalize.pkl}",
        "obs_keys": keys, "obs_sizes": sizes,
        "obs_description": {
            "orientation": "base orientation as rotation vector (rad), world frame",
            "angular_vel": "base angular velocity (rad/s), world frame, clipped to +-8",
            "vel": "base linear velocity (m/s), world frame, clipped to +-8",
            "motor_state": "3 wheel speeds divided by max_wheel_velocity (30), clipped to +-2",
            "actions": "previous action (3, in [-1,1])",
            "cmd": "velocity command [vx, vy (m/s), yaw rate (rad/s)], world frame",
            "heading": "[sin(yaw), cos(yaw)] of the base",
            "vint": "clipped integral of the planar velocity error, += (cmd - vel) * 0.01 each control step",
        },
        "normalised_keys": PROPRIO_KEYS, "norm_eps": float(vn.epsilon), "clip_obs": float(vn.clip_obs), "norm": norm,
        "activation": "tanh", "layers": layers,
        "action": {"low": asp.low.tolist(), "high": asp.high.tolist(), "deterministic": "clip(action_net(pi_net(x)), low, high)",
                   "to_ctrl": "ctrl = -clip(a * max_wheel_velocity, +-max_wheel_velocity) (motor torque in N m, applied to actuators 0..2)",
                   "max_wheel_velocity": e["max_wheel_velocity"]},
        "timing": {"mujoco_timestep_s": 0.002, "action_repeat": e["action_repeat"], "control_period_s": 0.002 * e["action_repeat"],
                   "integrator": e["integrator"], "obs_vel_clip": e["obs_vel_clip"], "cmd_vmax": e["cmd_vmax"], "cmd_wmax": e["cmd_wmax"]},
    }
    # numerical check: PyTorch SB3 predict vs a numpy re-implementation of exactly what the JSON encodes
    rng = np.random.default_rng(0)
    scale = {"orientation": .3, "angular_vel": 1, "vel": 1, "motor_state": .5, "actions": .7, "cmd": 1, "heading": .7, "vint": .5}
    worst = 0.0
    for _ in range(500):
        obs = {k: (rng.normal(size=sizes[k]) * scale[k]).astype(np.float32) for k in keys}
        with torch.no_grad():
            a_ref, _ = model.predict(_normalize(obs, vn), deterministic=True)
        x = []
        for k in keys:
            v = obs[k].astype(np.float64)
            if k in norm:
                mu, var = np.array(norm[k]["mean"]), np.array(norm[k]["var"])
                v = np.clip((v - mu) / np.sqrt(var + out["norm_eps"]), -out["clip_obs"], out["clip_obs"])
            x.append(v)
        x = np.concatenate(x)
        for i, L in enumerate(layers):
            x = np.array(L["W"]) @ x + np.array(L["b"])
            if i < len(layers) - 1:
                x = np.tanh(x)
        a = np.clip(x, asp.low, asp.high)
        worst = max(worst, float(np.abs(a - a_ref).max()))
    print(f"max |action_json - action_sb3| over 500 random observations: {worst:.3e}")
    assert worst < 1e-5
    out["verification"] = {"n_random_obs": 500, "max_abs_err": worst, "reference": "stable_baselines3 PPO.predict(deterministic=True) on VecNormalize-normalised observations"}
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print("wrote", OUT, OUT.stat().st_size // 1024, "KiB; keys", keys, sizes)


if __name__ == "__main__":
    main()
