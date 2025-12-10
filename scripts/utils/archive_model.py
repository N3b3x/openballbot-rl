#!/usr/bin/env python3
"""
Archive a trained model to the legacy folder.

This script copies the best model and essential metadata from a training run
to the legacy folder with proper naming and organization.

Usage:
    python scripts/utils/archive_model.py \
        --experiment outputs/experiments/runs/20241209_143022_ppo_perlin_directional_seed10 \
        --name "ppo-perlin-directional-seed10"
    
    # Or with automatic date
    python scripts/utils/archive_model.py \
        --experiment outputs/experiments/runs/20241209_143022_ppo_perlin_directional_seed10
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import yaml
import json


def archive_model(experiment_path: Path, model_name: str = None, date: str = None, experiment_info: Dict = None):
    """
    Archive a model to the archived_models folder.
    
    Args:
        experiment_path: Path to the experiment directory
        model_name: Name for the model (optional, will be inferred if not provided)
        date: Date string in YYYY-MM-DD format (optional, uses today if not provided)
        experiment_info: Optional dict with experiment info (for checkpoint selection)
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
    
    if not experiment_path.is_dir():
        raise ValueError(f"Path is not a directory: {experiment_path}")
    
    # Get date
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Get model name
    if model_name is None:
        # Try to infer from experiment name
        exp_name = experiment_path.name
        # Remove timestamp prefix if present (format: YYYYMMDD_HHMMSS_...)
        if exp_name.count('_') >= 2:
            parts = exp_name.split('_', 2)
            if len(parts[0]) == 8 and len(parts[1]) == 6:  # Date and time format
                model_name = parts[2].replace('_', '-')
            else:
                model_name = exp_name.replace('_', '-')
        else:
            model_name = exp_name.replace('_', '-')
    
    # Create archive folder name
    archive_folder_name = f"{date}_{model_name}"
    archive_path = Path("outputs/experiments/archived_models") / archive_folder_name
    
    # Check if already exists
    if archive_path.exists():
        response = input(f"Archive folder {archive_path} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(archive_path)
    
    # Create archive folder
    archive_path.mkdir(parents=True, exist_ok=True)
    print(f"Created archive folder: {archive_path}")
    
    # Copy best model
    best_model_src = experiment_path / "best_model" / "best_model.zip"
    if best_model_src.exists():
        best_model_dst = archive_path / "best_model.zip"
        shutil.copy2(best_model_src, best_model_dst)
        print(f"✓ Copied best_model.zip")
    else:
        print(f"⚠️  Warning: best_model.zip not found at {best_model_src}")
    
    # Copy config.yaml
    config_src = experiment_path / "config.yaml"
    if config_src.exists():
        config_dst = archive_path / "config.yaml"
        shutil.copy2(config_src, config_dst)
        print(f"✓ Copied config.yaml")
    else:
        print(f"⚠️  Warning: config.yaml not found at {config_src}")
    
    # Copy info.txt
    info_src = experiment_path / "info.txt"
    if info_src.exists():
        info_dst = archive_path / "info.txt"
        shutil.copy2(info_src, info_dst)
        print(f"✓ Copied info.txt")
    else:
        print(f"⚠️  Warning: info.txt not found at {info_src}")
    
    # Optionally copy progress.csv (small file)
    progress_src = experiment_path / "progress.csv"
    if progress_src.exists():
        progress_dst = archive_path / "progress.csv"
        shutil.copy2(progress_src, progress_dst)
        print(f"✓ Copied progress.csv")
    
    # Copy evaluation results if available
    results_src = experiment_path / "results" / "evaluations.npz"
    if results_src.exists():
        results_dir = archive_path / "results"
        results_dir.mkdir(exist_ok=True)
        shutil.copy2(results_src, results_dir / "evaluations.npz")
        print(f"✓ Copied evaluations.npz")
    
    # Copy a few key checkpoints (not all, just important ones)
    checkpoints_src = experiment_path / "checkpoints"
    if checkpoints_src.exists():
        checkpoints = sorted(checkpoints_src.glob("*.zip"))
        if checkpoints:
            # Copy last checkpoint, middle checkpoint, and first checkpoint if available
            checkpoints_dir = archive_path / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            
            # Always copy the last checkpoint
            if len(checkpoints) > 0:
                shutil.copy2(checkpoints[-1], checkpoints_dir / checkpoints[-1].name)
                print(f"✓ Copied checkpoint: {checkpoints[-1].name}")
            
            # Copy middle checkpoint if there are multiple
            if len(checkpoints) > 2:
                mid_idx = len(checkpoints) // 2
                shutil.copy2(checkpoints[mid_idx], checkpoints_dir / checkpoints[mid_idx].name)
                print(f"✓ Copied checkpoint: {checkpoints[mid_idx].name}")
            
            # Copy first checkpoint if training was substantial
            if len(checkpoints) > 1 and experiment_info.get("max_steps", 0) > 100000:
                shutil.copy2(checkpoints[0], checkpoints_dir / checkpoints[0].name)
                print(f"✓ Copied checkpoint: {checkpoints[0].name}")
    
    # Create a basic README if it doesn't exist
    readme_path = archive_path / "README.md"
    if not readme_path.exists():
        # Try to extract info from config and info.txt
        description = f"# {model_name.replace('-', ' ').title()}\n\n"
        description += f"**Date:** {date}\n\n"
        
        if config_src.exists():
            try:
                with open(config_src) as f:
                    config = yaml.safe_load(f)
                    algo = config.get("algo", {}).get("name", "unknown")
                    seed = config.get("seed", "unknown")
                    terrain = config.get("problem", {}).get("terrain", {}).get("type", "unknown")
                    reward = config.get("problem", {}).get("reward", {}).get("type", "unknown")
                    
                    description += f"**Algorithm:** {algo}\n"
                    description += f"**Seed:** {seed}\n"
                    description += f"**Terrain:** {terrain}\n"
                    description += f"**Reward:** {reward}\n\n"
            except:
                pass
        
        description += "## Performance\n\n"
        description += "_(Add performance metrics here)_\n\n"
        description += "## Training Details\n\n"
        description += "_(Add training details here)_\n\n"
        description += "## Notes\n\n"
        description += "_(Add any relevant notes here)_\n"
        
        readme_path.write_text(description)
        print(f"✓ Created README.md template")
    
    print(f"\n✓ Model archived successfully to: {archive_path}")
    print(f"\nNext steps:")
    print(f"  1. Review and update {readme_path}")
    print(f"  2. Add performance metrics and notes")
    print(f"  3. Commit to git: git add {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Archive a trained model to the archived_models folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Archive with automatic name inference
  python scripts/utils/archive_model.py \\
      --experiment outputs/experiments/runs/20241209_143022_ppo_perlin_directional_seed10
  
  # Archive with custom name
  python scripts/utils/archive_model.py \\
      --experiment outputs/experiments/runs/20241209_143022_ppo_perlin_directional_seed10 \\
      --name "ppo-perlin-directional-seed10"
  
  # Archive with specific date
  python scripts/utils/archive_model.py \\
      --experiment outputs/experiments/runs/20241209_143022_ppo_perlin_directional_seed10 \\
      --name "ppo-perlin-directional-seed10" \\
      --date "2024-12-09"
        """
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to the experiment directory to archive"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for the archived model (inferred from experiment name if not provided)"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format (uses today if not provided)"
    )
    
    args = parser.parse_args()
    
    experiment_path = Path(args.experiment).resolve()
    
    # Get experiment info for checkpoint selection
    from scan_and_archive_runs import get_experiment_info
    experiment_info = get_experiment_info(experiment_path)
    
    try:
        archive_model(experiment_path, args.name, args.date, experiment_info)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

