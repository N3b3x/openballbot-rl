"""Train a PPO policy for the perception-guided ballbot study.

Variants
    proprio       MLP on proprioception only (no cameras are rendered)
    depth_frozen  proprioception + frozen pretrained depth encoder (Salehi 2025)
    depth_e2e     proprioception + depth CNN trained end-to-end with PPO

Example
    python scripts/study/train_ppo.py --name proprio_s0 --variant proprio \
        --env-config configs/env/perlin_directional_train.yaml --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import ballbot_gym  # noqa: F401  (registers the env)
from ballbot_rl.policies import Extractor
from ballbot_rl.training.utils import make_ballbot_env

PROPRIO_KEYS = ["orientation", "angular_vel", "vel", "motor_state", "actions"]


class FrozenEncoderExtractor(Extractor):
    """Extractor whose frozen depth encoder stays in eval mode (fixed BatchNorm stats)."""

    def train(self, mode: bool = True):
        super().train(mode)
        for key, ext in self.extractors.items():
            if "rgbd_" in key:
                ext.eval()
        return self


def make_policy_kwargs(variant, encoder_path="outputs/encoders/encoder_epoch_53", log_std_init=-1.0):
    """Policy kwargs shared by training and evaluation (the encoder path is resolved at load time)."""
    extractor_cls, extractor_kwargs = Extractor, {}
    if variant == "depth_frozen":
        extractor_cls = FrozenEncoderExtractor
        extractor_kwargs = {"frozen_encoder_path": str(encoder_path)}
    elif variant == "depth_e2e":
        extractor_kwargs = {"frozen_encoder_path": ""}
    return dict(features_extractor_class=extractor_cls,
                features_extractor_kwargs=extractor_kwargs,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.Tanh,
                log_std_init=log_std_init)


def load_env_cfg(path, variant, overrides=None):
    cfg = yaml.safe_load(open(path))
    cfg["camera"]["enabled"] = variant != "proprio"
    for k, v in (overrides or {}).items():
        sect, key = k.split(".")
        cfg[sect][key] = v
    return cfg


def make_vec_env(cfg, n_envs, seed):
    env_config = {k: cfg[k] for k in ("camera", "env", "logging", "tof", "height_scan") if k in cfg}
    fns = [
        make_ballbot_env(terrain_config=cfg["terrain"],
                         reward_config=cfg["reward"],
                         env_config=env_config,
                         seed=seed * 1000 + i,
                         eval_env=True) for i in range(n_envs)
    ]
    return SubprocVecEnv(fns)


def lin_schedule(start, end):
    return lambda progress_remaining: end + (start - end) * progress_remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", choices=["proprio", "depth_frozen", "depth_e2e"], required=True)
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--total-timesteps", type=float, default=3e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=20)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1280)
    ap.add_argument("--n-epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--encoder", default="outputs/encoders/encoder_epoch_53")
    ap.add_argument("--out-root", default="outputs/study/runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init-from", default="", help="resume weights from a model zip")
    ap.add_argument("--zero-init-actor", action="store_true", help="zero the policy output layer (residual control)")
    ap.add_argument("--gate-bias", type=float, default=None,
                    help="hybrid_gate envs: zero the output layer and set the gate logit bias (-0.8 -> gate 0.1, i.e. start as ~LQR)")
    args = ap.parse_args()

    set_random_seed(args.seed)
    out = Path(args.out_root) / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_env_cfg(args.env_config, args.variant)
    json.dump({"args": vars(args), "env": cfg}, open(out / "run_config.json", "w"), indent=1)

    venv = make_vec_env(cfg, args.n_envs, args.seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        norm_obs_keys=PROPRIO_KEYS, clip_obs=10.0, clip_reward=10.0)

    policy_kwargs = make_policy_kwargs(args.variant, args.encoder, args.log_std_init)
    model = PPO("MultiInputPolicy", venv, policy_kwargs=policy_kwargs,
                learning_rate=lin_schedule(args.lr, args.lr_end),
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                gamma=args.gamma, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef,
                vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03, seed=args.seed,
                device=args.device, verbose=1)
    if args.zero_init_actor:
        # policy mean starts at exactly zero: with residual_lqr the initial controller is the LQR itself
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
    if args.gate_bias is not None:
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[3] = args.gate_bias
    if args.init_from:
        model.set_parameters(args.init_from, exact_match=False)
        vn_src = Path(args.init_from).parent / "vecnormalize.pkl"
        if not vn_src.exists():  # checkpoint: ppo_<n>_steps.zip -> ppo_vecnormalize_<n>_steps.pkl
            vn_src = Path(args.init_from).with_name(Path(args.init_from).name.replace("ppo_", "ppo_vecnormalize_").replace(".zip", ".pkl"))
        if vn_src.exists():
            import pickle
            with open(vn_src, "rb") as f:
                old = pickle.load(f)
            for k, v in old.obs_rms.items():
                if k in venv.obs_rms:
                    venv.obs_rms[k] = v
            venv.ret_rms = old.ret_rms
            print("restored VecNormalize stats from", vn_src, flush=True)
    model.set_logger(configure(str(out), ["stdout", "csv"]))

    n_params = sum(p.numel() for p in model.policy.parameters())
    n_train = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"params total={n_params} trainable={n_train}", flush=True)

    ckpt = CheckpointCallback(save_freq=max(1, 500_000 // args.n_envs), save_path=str(out / "checkpoints"),
                              name_prefix="ppo", save_vecnormalize=True)
    t0 = time.time()
    model.learn(total_timesteps=int(args.total_timesteps), callback=ckpt)
    model.save(out / "final_model")
    venv.save(str(out / "vecnormalize.pkl"))
    json.dump({"wall_time_s": time.time() - t0, "params": n_params, "trainable": n_train},
              open(out / "summary.json", "w"))
    venv.close()


if __name__ == "__main__":
    main()
