#!/usr/bin/env python3
"""
Evaluate all archived models and save evaluation results.

This script evaluates all models in outputs/experiments/archived_models/
and saves evaluation results (evaluations.npz) for each model.

Usage:
    # Evaluate all archived models
    python scripts/utils/evaluate_archived_models.py
    
    # Evaluate specific model
    python scripts/utils/evaluate_archived_models.py --model legacy_salehi-2025-original
    
    # Custom evaluation settings
    python scripts/utils/evaluate_archived_models.py --n_episodes 10 --n_test 5
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import yaml
import json

from stable_baselines3 import PPO, SAC
from termcolor import colored

from ballbot_rl.training.utils import make_ballbot_env
from ballbot_gym.core.config import load_config, get_component_config


def evaluate_model(model_path: Path, n_episodes: int = 8, n_test: int = 5) -> Dict:
    """
    Evaluate a model and return results.
    
    Args:
        model_path: Path to best_model.zip
        n_episodes: Number of episodes per evaluation
        n_test: Number of test runs
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"   Loading model: {model_path}")
    
    # Try to load model (auto-detect algorithm)
    model = None
    try:
        model = PPO.load(str(model_path))
        algo = "ppo"
    except:
        try:
            model = SAC.load(str(model_path))
            algo = "sac"
        except Exception as e:
            print(f"   ✗ Failed to load model: {e}")
            return None
    
    # Load config from archive folder
    archive_dir = model_path.parent
    config_path = archive_dir / "config.yaml"
    
    if config_path.exists():
        config = load_config(str(config_path))
        terrain_config = get_component_config(config, "terrain")
        reward_config = get_component_config(config, "reward")
        env_config = {
            "camera": config.get("camera", {}),
            "env": config.get("env", {}),
            "logging": config.get("logging", {})
        }
    else:
        # Use defaults
        terrain_config = {"type": "perlin", "config": {}}
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        env_config = None
    
    # Create evaluation environment
    env_factory = make_ballbot_env(
        terrain_config=terrain_config,
        reward_config=reward_config,
        env_config=env_config,
        gui=False,
        log_options={"cams": False, "reward_terms": False},
        seed=42,
        eval_env=True
    )
    env = env_factory()
    
    # Run evaluations
    all_rewards = []
    all_lengths = []
    
    for test_run in range(n_test):
        test_rewards = []
        test_lengths = []
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=42 + test_run * n_episodes + episode)
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)
            test_lengths.append(episode_length)
        
        all_rewards.extend(test_rewards)
        all_lengths.extend(test_lengths)
    
    env.close()
    
    # Calculate statistics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    mean_length = np.mean(all_lengths)
    std_length = np.std(all_lengths)
    
    results = {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(mean_length),
        "std_length": float(std_length),
        "all_rewards": all_rewards,
        "all_lengths": all_lengths,
        "n_episodes": n_episodes * n_test
    }
    
    print(f"   ✓ Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   ✓ Mean length: {mean_length:.1f} ± {std_length:.1f}")
    
    return results


def save_evaluation_results(archive_dir: Path, results: Dict):
    """Save evaluation results to archive directory."""
    results_dir = archive_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save as npz (compatible with training format)
    npz_path = results_dir / "evaluations.npz"
    np.savez(
        npz_path,
        timesteps=[0],  # Single evaluation point
        results=[results["all_rewards"]],
        ep_lengths=[results["all_lengths"]]
    )
    
    # Also save as JSON for easy reading
    json_path = results_dir / "evaluation_summary.json"
    summary = {
        "mean_reward": results["mean_reward"],
        "std_reward": results["std_reward"],
        "mean_length": results["mean_length"],
        "std_length": results["std_length"],
        "n_episodes": results["n_episodes"],
        "min_reward": float(np.min(results["all_rewards"])),
        "max_reward": float(np.max(results["all_rewards"]))
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✓ Saved evaluation results to {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all archived models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model folder to evaluate (e.g., 'legacy_salehi-2025-original')"
    )
    
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=8,
        help="Number of episodes per evaluation run (default: 8)"
    )
    
    parser.add_argument(
        "--n_test",
        type=int,
        default=5,
        help="Number of test runs (default: 5)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have evaluation results"
    )
    
    args = parser.parse_args()
    
    archived_dir = Path("outputs/experiments/archived_models")
    
    if not archived_dir.exists():
        print(f"Error: Archived models directory not found: {archived_dir}")
        return 1
    
    print("=" * 80)
    print("Evaluating Archived Models")
    print("=" * 80)
    print()
    
    # Find all archived models
    if args.model:
        model_dirs = [archived_dir / args.model]
    else:
        model_dirs = [d for d in archived_dir.iterdir() 
                     if d.is_dir() and d.name != ".git" and not d.name.startswith(".")]
    
    model_dirs = sorted(model_dirs)
    
    print(f"Found {len(model_dirs)} archived model(s) to evaluate")
    print()
    
    evaluated_count = 0
    skipped_count = 0
    
    for model_dir in model_dirs:
        print(colored(f"Evaluating: {model_dir.name}", "cyan", attrs=["bold"]))
        
        # Check if already evaluated
        existing_results = model_dir / "results" / "evaluations.npz"
        if args.skip_existing and existing_results.exists():
            print(f"   ⏭️  Skipping (results already exist)")
            skipped_count += 1
            print()
            continue
        
        # Find best_model.zip
        best_model_path = model_dir / "best_model.zip"
        if not best_model_path.exists():
            print(f"   ⚠️  best_model.zip not found, skipping")
            print()
            continue
        
        # Evaluate
        try:
            results = evaluate_model(best_model_path, args.n_episodes, args.n_test)
            if results:
                save_evaluation_results(model_dir, results)
                evaluated_count += 1
        except Exception as e:
            print(f"   ✗ Error evaluating model: {e}")
        
        print()
    
    print("=" * 80)
    print(f"✓ Evaluated {evaluated_count} model(s)")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} model(s) (already have results)")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

