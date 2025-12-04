import numpy as np
import json
import pdb
import torch
import random
import argparse
import shutil
import platform
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

# Set multiprocessing start method for macOS compatibility
if platform.system() == 'Darwin':  # macOS
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import VectorizedActionNoise, NormalActionNoise
from stable_baselines3.common.logger import configure
from termcolor import colored

from ballbot_rl.training.utils import make_ballbot_env
from ballbot_rl.training.schedules import lr_schedule
from ballbot_rl.training.interactive import confirm
from ballbot_rl.training.callbacks import create_training_callbacks
from ballbot_rl.policies import Extractor
from ballbot_gym.core.config import get_component_config, load_training_config


def main(config, seed):

    policy_kwargs = dict(
        activation_fn=torch.nn.LeakyReLU,
        net_arch=dict(pi=[
            config["hidden_sz"], config["hidden_sz"], config["hidden_sz"],
            config["hidden_sz"]
        ],
                      vf=[
                          config["hidden_sz"], config["hidden_sz"],
                          config["hidden_sz"], config["hidden_sz"]
                      ]),
        features_extractor_class=Extractor,  #note that this will be shared by the policy and the value networks
        features_extractor_kwargs={
            "frozen_encoder_path": config["frozen_cnn"]
        },
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "weight_decay": float(config["algo"]["weight_decay"])
        },
    )

    N_ENVS = int(config["num_envs"])

    # Use DummyVecEnv on macOS by default to avoid multiprocessing issues
    # SubprocVecEnv can hang on macOS due to multiprocessing and MuJoCo initialization
    use_subproc = config.get("use_subproc_vec_env", platform.system() != 'Darwin')
    if use_subproc:
        VecEnvClass = SubprocVecEnv
        print(colored("Using SubprocVecEnv (multi-process)", "cyan"))
    else:
        VecEnvClass = DummyVecEnv
        print(colored("Using DummyVecEnv (single process) for compatibility", "yellow"))

    # Extract component configs from merged config
    terrain_config = get_component_config(config, "terrain")
    reward_config = get_component_config(config, "reward")
    
    # Extract env config (camera, env, logging settings)
    env_config = {
        "camera": config.get("camera", {}),
        "env": config.get("env", {}),
        "logging": config.get("logging", {})
    }
    
    try:
        vec_env = VecEnvClass([
            make_ballbot_env(
                terrain_config=terrain_config,
                reward_config=reward_config,
                env_config=env_config,
                seed=seed,
                eval_env=True) for _ in range(N_ENVS)
        ])
        eval_env = VecEnvClass([
            make_ballbot_env(
                terrain_config=terrain_config,
                reward_config=reward_config,
                env_config=env_config,
                seed=seed + N_ENVS + env_i,
                eval_env=True) for env_i in range(N_ENVS)
        ])
    except Exception as e:
        if VecEnvClass == SubprocVecEnv:
            print(colored(
                f"SubprocVecEnv failed: {e}. Falling back to DummyVecEnv (single process).",
                "yellow",
                attrs=["bold"]
            ))
            VecEnvClass = DummyVecEnv
            vec_env = VecEnvClass([
                make_ballbot_env(
                    terrain_config=terrain_config,
                    reward_config=reward_config,
                    seed=seed) for _ in range(N_ENVS)
            ])
            eval_env = VecEnvClass([
                make_ballbot_env(
                    terrain_config=terrain_config,
                    reward_config=reward_config,
                    seed=seed + N_ENVS + env_i,
                    eval_env=True) for env_i in range(N_ENVS)
            ])
        else:
            raise

    device = "cuda"
    if not config["resume"]:

        normalize_advantage = bool(config["algo"]["normalize_advantage"])
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=1,
                    ent_coef=float(config["algo"]["ent_coef"]),
                    device=device,
                    clip_range=float(config["algo"]["clip_range"]),
                    target_kl=float(config["algo"]["target_kl"]),
                    vf_coef=float(config["algo"]["vf_coef"]),
                    learning_rate=float(config["algo"]["learning_rate"])
                    if config["algo"]["learning_rate"] != -1 else lr_schedule,
                    policy_kwargs=policy_kwargs,
                    n_steps=int(config["algo"]["n_steps"]),
                    batch_size=int(config["algo"]["batch_sz"]),
                    n_epochs=int(config["algo"]["n_epochs"]),
                    normalize_advantage=normalize_advantage,
                    seed=seed)

    else:
        print(
            colored(f"loading model from {config['resume']}...",
                    "yellow",
                    attrs=["bold"]))

        custom_objects = dict(
            ent_coef=float(config["algo"]["ent_coef"]),
            device=device,
            clip_range=float(config["algo"]["clip_range"]),
            vf_coef=float(config["algo"]["vf_coef"]),
            learning_rate=float(config["algo"]["learning_rate"])
            if config["algo"]["learning_rate"] != -1 else lr_schedule,
            n_steps=int(config["algo"]["n_steps"]),
            seed=seed)

        for k, v in custom_objects.items():
            print(k, v)

        model = PPO.load(config["resume"],
                         device=device,
                         env=vec_env,
                         custom_objects=custom_objects)

    total_timesteps = int(float(config["total_timesteps"]))

    # Create experiment directory with descriptive name
    # Format: {timestamp}_{algo}_{env_config_name}_{seed}
    # Example: 20241203_143022_ppo_perlin_directional_seed10
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    algo_name = config["algo"]["name"]
    
    # Extract environment config name from problem config (terrain + reward types)
    # This works because env_config is merged into problem section during loading
    terrain_type = config.get("problem", {}).get("terrain", {}).get("type", "unknown")
    reward_type = config.get("problem", {}).get("reward", {}).get("type", "unknown")
    env_config_name = f"{terrain_type}_{reward_type}"
    
    seed_str = f"seed{config.get('seed', 'unknown')}"
    experiment_name = f"{timestamp}_{algo_name}_{env_config_name}_{seed_str}"
    
    if config.get('out') and config['out'].strip():
        # If out path is specified, use it as base (can be full path or relative)
        out_path = Path(config['out']).resolve()
        # If it's a directory, append experiment name
        if out_path.is_dir() or not out_path.suffix:
            out_path = out_path / experiment_name
    else:
        # Default to outputs/experiments/runs/{experiment_name}
        out_path = Path("outputs/experiments/runs") / experiment_name
    
    # Check if directory exists and is not empty
    if out_path.exists() and any(out_path.iterdir()):
        if confirm(
                colored(
                    f"The output directory ({out_path}) already exists and is not empty. Overwrite?",
                    "red",
                    attrs=["bold"])):
            shutil.rmtree(out_path)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            print(colored("Okay, aborted! Exiting.", "red"))
            exit(1)
    else:
        out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "config.yaml").open("w") as fl:
        yaml.dump(config, fl)
    with (out_path / "info.txt").open("w") as fl:
        json.dump(
            {
                "algo": config["algo"]["name"],
                "num_envs": config["num_envs"],
                "out": str(out_path),
                "resume": config["resume"],
                "seed": config["seed"]
            }, fl)

    logger_path = str(out_path) + "/"
    logger = configure(logger_path, ["stdout", "csv"])
    model.set_logger(logger)

    callback = create_training_callbacks(
        eval_env=eval_env,
        out_path=out_path,
        config=config
    )

    print(model.policy)
    num_params_total = sum(
        [param.numel() for param in model.policy.parameters()])
    num_params_learnable = sum([
        param.numel() for param in model.policy.parameters()
        if param.requires_grad
    ])
    print(
        colored(f"num_total_params={num_params_total}", "cyan",
                attrs=["bold"]))
    print(
        colored(f"num_learnable_params={num_params_learnable}",
                "cyan",
                attrs=["bold"]))
    print(model.policy.optimizer)
    print(colored(f"total_timesteps={total_timesteps}", "yellow"))
    num_updates_per_rollout = (
        config["algo"]["n_epochs"] * config["num_envs"] *
        config["algo"]["n_steps"]) / config["algo"]["batch_sz"]
    if not confirm(
            colored(
                f"the current config results in {num_updates_per_rollout} number of updates per rollout. Continue? ",
                "green",
                attrs=["bold"])):
        print("Aborting.")
        os._exit(1)
    else:
        print("Okay.")

    model.terrain_type = config["problem"]["terrain_type"]
    model.learn(total_timesteps=total_timesteps, callback=callback)

    vec_env.close()


def cli_main():
    """CLI entry point for training."""
    _parser = argparse.ArgumentParser(description="Train a policy.")
    _parser.add_argument("--config",
                         help="your yaml config file",
                         required=True)

    _args = _parser.parse_args()

    config_path = Path(_args.config).resolve()
    
    # Load training config (merges with env config)
    _config = load_training_config(str(config_path))
    
    # Update frozen_cnn path if specified
    if _config.get("frozen_cnn"):
        encoder_path = Path(_config["frozen_cnn"]).resolve()
        _config["frozen_cnn"] = str(encoder_path)

    repeatable = True if int(_config["seed"]) != -1 else False
    if repeatable:
        _seed = int(_config["seed"])
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(_seed)
            torch.cuda.manual_seed_all(_seed)

        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(_seed)

        main(_config, seed=_seed)
    else:
        raise Exception("Seed must be specified for reproducibility")


if __name__ == "__main__":
    cli_main()
