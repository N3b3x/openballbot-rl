"""Callback utilities for training."""
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from pathlib import Path
from typing import Dict


def create_training_callbacks(
    eval_env,
    out_path: Path,
    config: Dict
) -> CallbackList:
    """
    Create callbacks for training.
    
    Args:
        eval_env: Evaluation environment (vectorized).
        out_path: Path to save outputs (checkpoints, best model, etc.).
        config: Training configuration dictionary.
    
    Returns:
        CallbackList: List of callbacks for training.
    """
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_path / "best_model"),
        log_path=str(out_path / "results"),
        eval_freq=5000 if config["algo"]["name"] == "ppo" else 500,
        n_eval_episodes=8,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([
        eval_callback,
        CheckpointCallback(20000,
                           save_path=str(out_path / "checkpoints"),
                           name_prefix=f"{config['algo']['name']}_agent")
    ])
    
    return callback

