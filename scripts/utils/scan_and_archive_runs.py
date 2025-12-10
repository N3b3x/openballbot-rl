#!/usr/bin/env python3
"""
Scan training runs and archive good models.

This script:
1. Scans all training runs in outputs/experiments/runs/
2. Identifies runs with good models (large step counts, best models)
3. Archives them to outputs/experiments/archived_models/ with proper organization
4. Includes the legacy folder as part of the archive system

Usage:
    # Scan and show what would be archived (dry run)
    python scripts/utils/scan_and_archive_runs.py --dry-run
    
    # Archive all good models automatically
    python scripts/utils/scan_and_archive_runs.py --min-steps 100000
    
    # Archive specific runs
    python scripts/utils/scan_and_archive_runs.py --runs run1 run2 run3
"""

import argparse
import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_experiment_info(experiment_path: Path) -> Optional[Dict]:
    """Extract information about an experiment."""
    info = {
        "path": experiment_path,
        "name": experiment_path.name,
        "has_best_model": False,
        "has_checkpoints": False,
        "max_steps": 0,
        "config": None,
        "metadata": None,
    }
    
    # Check for best model
    best_model_path = experiment_path / "best_model" / "best_model.zip"
    if best_model_path.exists():
        info["has_best_model"] = True
        info["best_model_size"] = best_model_path.stat().st_size / (1024 * 1024)  # MB
    
    # Check for checkpoints
    checkpoints_dir = experiment_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.zip"))
        if checkpoints:
            info["has_checkpoints"] = True
            info["num_checkpoints"] = len(checkpoints)
            # Extract max steps from checkpoint names
            for cp in checkpoints:
                try:
                    # Format: ppo_agent_200000_steps.zip
                    parts = cp.stem.split("_")
                    for i, part in enumerate(parts):
                        if part == "steps":
                            steps = int(parts[i - 1])
                            info["max_steps"] = max(info["max_steps"], steps)
                except:
                    pass
    
    # Try to get max steps from progress.csv
    progress_path = experiment_path / "progress.csv"
    if progress_path.exists() and HAS_PANDAS:
        try:
            df = pd.read_csv(progress_path)
            if "time/total_timesteps" in df.columns:
                info["max_steps"] = max(info["max_steps"], int(df["time/total_timesteps"].max()))
        except:
            pass
    elif progress_path.exists():
        # Fallback: try to read last line manually
        try:
            with open(progress_path) as f:
                lines = f.readlines()
                if len(lines) > 1:  # Skip header
                    # Try to parse last line
                    last_line = lines[-1].strip().split(',')
                    if len(last_line) > 0:
                        # Look for timesteps column (usually first or second)
                        for i, val in enumerate(last_line[:5]):  # Check first 5 columns
                            try:
                                steps = int(float(val))
                                if steps > 1000:  # Reasonable threshold
                                    info["max_steps"] = max(info["max_steps"], steps)
                                    break
                            except:
                                pass
        except:
            pass
    
    # Load config
    config_path = experiment_path / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                info["config"] = yaml.safe_load(f)
        except:
            pass
    
    # Load metadata
    info_path = experiment_path / "info.txt"
    if info_path.exists():
        try:
            with open(info_path) as f:
                info["metadata"] = json.load(f)
        except:
            pass
    
    return info


def scan_runs(runs_dir: Path, min_steps: int = 0) -> List[Dict]:
    """Scan all runs and return information about them."""
    runs = []
    
    if not runs_dir.exists():
        return runs
    
    # Get all experiment directories (exclude legacy, .gitkeep)
    for exp_dir in runs_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name in ["legacy", ".git"]:
            continue
        
        info = get_experiment_info(exp_dir)
        if info:
            # Filter by minimum steps
            if info["max_steps"] >= min_steps:
                runs.append(info)
    
    # Sort by max_steps (descending)
    runs.sort(key=lambda x: x["max_steps"], reverse=True)
    
    return runs


def archive_model_to_archived(experiment_info: Dict, archived_dir: Path, date: str = None):
    """Archive a model to the archived_models directory."""
    experiment_path = experiment_info["path"]
    
    if date is None:
        # Try to extract date from experiment name or use today
        exp_name = experiment_info["name"]
        try:
            # Format: YYYYMMDD_HHMMSS_...
            if exp_name.count('_') >= 2:
                date_str = exp_name.split('_')[0]
                if len(date_str) == 8:
                    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        except:
            pass
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
    
    # Create model name from experiment
    exp_name = experiment_info["name"]
    if exp_name.count('_') >= 2:
        parts = exp_name.split('_', 2)
        if len(parts[0]) == 8 and len(parts[1]) == 6:
            model_name = parts[2].replace('_', '-')
        else:
            model_name = exp_name.replace('_', '-')
    else:
        model_name = exp_name.replace('_', '-')
    
    # Create archive folder name
    archive_folder_name = f"{date}_{model_name}"
    archive_path = archived_dir / archive_folder_name
    
    # Check if already exists
    if archive_path.exists():
        print(f"⚠️  {archive_folder_name} already exists, skipping...")
        return False
    
    # Create archive folder
    archive_path.mkdir(parents=True, exist_ok=True)
    print(f"📦 Archiving: {archive_folder_name}")
    
    # Copy best model
    best_model_src = experiment_path / "best_model" / "best_model.zip"
    if best_model_src.exists():
        best_model_dst = archive_path / "best_model.zip"
        shutil.copy2(best_model_src, best_model_dst)
        print(f"   ✓ best_model.zip ({experiment_info.get('best_model_size', 0):.1f} MB)")
    
    # Copy config.yaml
    config_src = experiment_path / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, archive_path / "config.yaml")
        print(f"   ✓ config.yaml")
    
    # Copy info.txt
    info_src = experiment_path / "info.txt"
    if info_src.exists():
        shutil.copy2(info_src, archive_path / "info.txt")
        print(f"   ✓ info.txt")
    
    # Copy progress.csv (small file)
    progress_src = experiment_path / "progress.csv"
    if progress_src.exists():
        shutil.copy2(progress_src, archive_path / "progress.csv")
        print(f"   ✓ progress.csv")
    
    # Copy evaluation results if available
    results_src = experiment_path / "results" / "evaluations.npz"
    if results_src.exists():
        results_dir = archive_path / "results"
        results_dir.mkdir(exist_ok=True)
        shutil.copy2(results_src, results_dir / "evaluations.npz")
        print(f"   ✓ evaluations.npz")
    
    # Copy key checkpoints (last, middle, first if substantial training)
    checkpoints_src = experiment_path / "checkpoints"
    if checkpoints_src.exists():
        checkpoints = sorted(checkpoints_src.glob("*.zip"))
        if checkpoints:
            checkpoints_dir = archive_path / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            
            # Always copy last checkpoint
            if len(checkpoints) > 0:
                shutil.copy2(checkpoints[-1], checkpoints_dir / checkpoints[-1].name)
                print(f"   ✓ checkpoint: {checkpoints[-1].name}")
            
            # Copy middle checkpoint if multiple
            if len(checkpoints) > 2:
                mid_idx = len(checkpoints) // 2
                shutil.copy2(checkpoints[mid_idx], checkpoints_dir / checkpoints[mid_idx].name)
                print(f"   ✓ checkpoint: {checkpoints[mid_idx].name}")
            
            # Copy first checkpoint if substantial training
            if len(checkpoints) > 1 and experiment_info.get("max_steps", 0) > 100000:
                shutil.copy2(checkpoints[0], checkpoints_dir / checkpoints[0].name)
                print(f"   ✓ checkpoint: {checkpoints[0].name}")
    
    # Create README
    readme_path = archive_path / "README.md"
    readme_content = generate_readme(experiment_info, date)
    readme_path.write_text(readme_content)
    print(f"   ✓ README.md")
    
    return True


def generate_readme(experiment_info: Dict, date: str) -> str:
    """Generate a README.md for the archived model."""
    exp_name = experiment_info["name"]
    config = experiment_info.get("config", {})
    metadata = experiment_info.get("metadata", {})
    
    # Extract info
    algo = config.get("algo", {}).get("name", "unknown") if config else "unknown"
    seed = config.get("seed", "unknown") if config else metadata.get("seed", "unknown")
    terrain = config.get("problem", {}).get("terrain", {}).get("type", "unknown") if config else "unknown"
    reward = config.get("problem", {}).get("reward", {}).get("type", "unknown") if config else "unknown"
    max_steps = experiment_info.get("max_steps", 0)
    
    # Create model name for title
    model_name = exp_name.replace('_', ' ').title()
    if exp_name.count('_') >= 2:
        parts = exp_name.split('_', 2)
        if len(parts[0]) == 8:
            model_name = parts[2].replace('_', ' ').title()
    
    readme = f"# {model_name}\n\n"
    readme += f"**Date:** {date}\n"
    readme += f"**Algorithm:** {algo}\n"
    readme += f"**Seed:** {seed}\n"
    readme += f"**Terrain:** {terrain}\n"
    readme += f"**Reward:** {reward}\n"
    readme += f"**Training Steps:** {max_steps:,}\n\n"
    
    readme += "## Performance\n\n"
    readme += "_(Add performance metrics here)_\n\n"
    
    readme += "## Training Details\n\n"
    if config:
        total_timesteps = config.get("total_timesteps", "unknown")
        num_envs = config.get("num_envs", "unknown")
        readme += f"- **Total Timesteps:** {total_timesteps}\n"
        readme += f"- **Parallel Environments:** {num_envs}\n"
    readme += "\n"
    
    readme += "## Notes\n\n"
    readme += "_(Add any relevant notes here)_\n"
    
    return readme


def move_legacy_to_archived(legacy_dir: Path, archived_dir: Path):
    """Move legacy folder contents to archived_models, preserving structure."""
    if not legacy_dir.exists():
        return
    
    print("📦 Moving legacy models to archived_models...")
    
    # Check if legacy has subdirectories or files directly
    legacy_items = list(legacy_dir.iterdir())
    
    # If legacy has a model directly (old structure)
    if (legacy_dir / "best_model.zip").exists() or (legacy_dir / "config.yaml").exists():
        # This is Salehi's model - create a special folder for it
        salehi_path = archived_dir / "legacy_salehi-2025-original"
        if not salehi_path.exists():
            salehi_path.mkdir(parents=True, exist_ok=True)
            # Copy all files
            for item in legacy_items:
                if item.is_file() and item.name != "README.md":
                    shutil.copy2(item, salehi_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, salehi_path / item.name, dirs_exist_ok=True)
            
            # Create README for Salehi's model
            readme = salehi_path / "README.md"
            if not readme.exists():
                readme.write_text("""# Salehi 2025 - Original Model

**Source:** Salehi, Achkan. "Reinforcement Learning for Ballbot Navigation in Uneven Terrain." arXiv preprint arXiv:2505.18417 (2025)

This is the original model provided by Salehi in the research paper.

## Model Details

- **Paper:** [arXiv:2505.18417](https://arxiv.org/abs/2505.18417)
- **Repository:** Original implementation from the paper

## Usage

This model can be used for:
- Baseline comparisons
- Understanding the original approach
- Reproducing paper results

## Notes

This model was provided as part of the original research implementation.
""")
            print(f"   ✓ Moved to: legacy_salehi-2025-original")
    else:
        # Legacy has subdirectories - move each one
        for item in legacy_items:
            if item.is_dir() and item.name != ".git":
                # Move the subdirectory
                dest = archived_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
                    print(f"   ✓ Moved: {item.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan training runs and archive good models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without actually archiving"
    )
    
    parser.add_argument(
        "--min-steps",
        type=int,
        default=100000,
        help="Minimum training steps to consider for archiving (default: 100000)"
    )
    
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Specific run names to archive (overrides min-steps filter)"
    )
    
    parser.add_argument(
        "--archive-dir",
        type=str,
        default="outputs/experiments/archived_models",
        help="Directory to archive models to (default: outputs/experiments/archived_models)"
    )
    
    args = parser.parse_args()
    
    runs_dir = Path("outputs/experiments/runs")
    archived_dir = Path(args.archive_dir)
    legacy_dir = runs_dir / "legacy"
    
    print("=" * 80)
    print("Scanning Training Runs for Archiving")
    print("=" * 80)
    print()
    
    # Scan runs
    print(f"Scanning runs in: {runs_dir}")
    all_runs = scan_runs(runs_dir, min_steps=0)  # Get all first
    
    if args.runs:
        # Filter to specific runs
        runs_to_archive = [r for r in all_runs if r["name"] in args.runs]
    else:
        # Filter by min_steps
        runs_to_archive = [r for r in all_runs if r["max_steps"] >= args.min_steps]
    
    print(f"\nFound {len(all_runs)} total runs")
    print(f"Found {len(runs_to_archive)} runs meeting criteria (min_steps >= {args.min_steps})")
    print()
    
    # Show runs
    print("Runs to archive:")
    print("-" * 80)
    for run in runs_to_archive:
        print(f"  {run['name']}")
        print(f"    Steps: {run['max_steps']:,}")
        print(f"    Best Model: {'✓' if run['has_best_model'] else '✗'}")
        print(f"    Checkpoints: {run.get('num_checkpoints', 0)}")
        print()
    
    if args.dry_run:
        print("DRY RUN - No files were archived")
        return 0
    
    # Create archived directory
    archived_dir.mkdir(parents=True, exist_ok=True)
    
    # Move legacy first
    if legacy_dir.exists():
        move_legacy_to_archived(legacy_dir, archived_dir)
        print()
    
    # Archive runs
    if not runs_to_archive:
        print("No runs to archive.")
        return 0
    
    print(f"Archiving {len(runs_to_archive)} runs to: {archived_dir}")
    print()
    
    archived_count = 0
    for run in runs_to_archive:
        if archive_model_to_archived(run, archived_dir):
            archived_count += 1
        print()
    
    print("=" * 80)
    print(f"✓ Archived {archived_count} models successfully!")
    print(f"  Location: {archived_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

