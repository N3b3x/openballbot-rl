#!/usr/bin/env python3
"""
Enhance existing archived models with evaluations and checkpoints.

This script adds missing components to already-archived models:
- Evaluation results (evaluations.npz)
- Key checkpoints (last, middle, first)
- Updates README with evaluation metrics if available

Usage:
    # Enhance all archived models
    python scripts/utils/enhance_archived_models.py
    
    # Enhance specific model
    python scripts/utils/enhance_archived_models.py --model 2025-12-04_ppo-flat-directional-seed10
"""

import argparse
import shutil
import numpy as np
from pathlib import Path
import yaml
import json


def find_original_run(archived_model_path: Path, runs_dir: Path) -> Path:
    """Find the original training run for an archived model."""
    # Try to match by name patterns
    archived_name = archived_model_path.name
    
    # Extract date and model name
    if archived_name.startswith("legacy_"):
        # Legacy model - check legacy folder
        legacy_path = runs_dir / "legacy"
        if legacy_path.exists():
            return legacy_path
        return None
    
    # Try to find matching run
    # Format: YYYY-MM-DD_model-name -> YYYYMMDD_HHMMSS_model-name
    parts = archived_name.split("_", 1)
    if len(parts) == 2:
        date_part = parts[0].replace("-", "")
        name_part = parts[1].replace("-", "_")
        
        # Search for runs matching this pattern
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name.startswith(date_part) and name_part.replace("_", "") in run_dir.name.replace("_", ""):
                return run_dir
    
    return None


def add_evaluations(archived_path: Path, original_path: Path):
    """Add evaluation results to archived model."""
    results_src = original_path / "results" / "evaluations.npz"
    if not results_src.exists():
        return False
    
    results_dir = archived_path / "results"
    results_dir.mkdir(exist_ok=True)
    results_dst = results_dir / "evaluations.npz"
    
    if results_dst.exists():
        return False  # Already exists
    
    shutil.copy2(results_src, results_dst)
    return True


def add_checkpoints(archived_path: Path, original_path: Path, max_steps: int = 0):
    """Add key checkpoints to archived model."""
    checkpoints_src = original_path / "checkpoints"
    if not checkpoints_src.exists():
        return False
    
    checkpoints = sorted(checkpoints_src.glob("*.zip"))
    if not checkpoints:
        return False
    
    checkpoints_dir = archived_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    added = False
    
    # Always copy last checkpoint
    if len(checkpoints) > 0:
        dst = checkpoints_dir / checkpoints[-1].name
        if not dst.exists():
            shutil.copy2(checkpoints[-1], dst)
            added = True
    
    # Copy middle checkpoint if multiple
    if len(checkpoints) > 2:
        mid_idx = len(checkpoints) // 2
        dst = checkpoints_dir / checkpoints[mid_idx].name
        if not dst.exists():
            shutil.copy2(checkpoints[mid_idx], dst)
            added = True
    
    # Copy first checkpoint if substantial training
    if len(checkpoints) > 1 and max_steps > 100000:
        dst = checkpoints_dir / checkpoints[0].name
        if not dst.exists():
            shutil.copy2(checkpoints[0], dst)
            added = True
    
    return added


def extract_eval_metrics(evaluations_path: Path) -> dict:
    """Extract evaluation metrics from evaluations.npz."""
    try:
        data = np.load(evaluations_path)
        results = data.get("results", [])
        timesteps = data.get("timesteps", [])
        
        if len(results) > 0:
            # Get final evaluation
            final_results = results[-1] if isinstance(results[-1], np.ndarray) else results
            mean_reward = float(np.mean(final_results))
            std_reward = float(np.std(final_results))
            final_timestep = int(timesteps[-1]) if len(timesteps) > 0 else 0
            
            return {
                "final_mean_reward": mean_reward,
                "final_std_reward": std_reward,
                "final_timestep": final_timestep,
                "num_evaluations": len(results)
            }
    except:
        pass
    
    return {}


def update_readme_with_metrics(archived_path: Path):
    """Update README with evaluation metrics if available."""
    readme_path = archived_path / "README.md"
    evaluations_path = archived_path / "results" / "evaluations.npz"
    
    if not readme_path.exists() or not evaluations_path.exists():
        return False
    
    # Extract metrics
    metrics = extract_eval_metrics(evaluations_path)
    if not metrics:
        return False
    
    # Read current README
    content = readme_path.read_text()
    
    # Update performance section
    if "## Performance" in content:
        # Replace placeholder or add metrics
        perf_section = f"""## Performance

- **Final Eval Reward:** {metrics['final_mean_reward']:.2f} ± {metrics['final_std_reward']:.2f}
- **Evaluated At:** {metrics['final_timestep']:,} steps
- **Number of Evaluations:** {metrics['num_evaluations']}
"""
        
        # Replace existing performance section
        import re
        pattern = r"## Performance\n\n.*?\n\n"
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, perf_section + "\n", content, flags=re.DOTALL)
        else:
            # Insert after Performance header
            content = content.replace("## Performance\n\n", perf_section)
        
        readme_path.write_text(content)
        return True
    
    return False


def enhance_model(archived_path: Path, runs_dir: Path):
    """Enhance a single archived model."""
    print(f"\n📦 Enhancing: {archived_path.name}")
    
    # Find original run
    original_path = find_original_run(archived_path, runs_dir)
    if not original_path or not original_path.exists():
        print(f"   ⚠️  Could not find original run, skipping")
        return False
    
    # Get max steps from config or info
    max_steps = 0
    config_path = archived_path / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                max_steps = config.get("total_timesteps", 0)
                if isinstance(max_steps, str) and "e" in max_steps.lower():
                    max_steps = float(max_steps.replace("e", "E"))
        except:
            pass
    
    enhanced = False
    
    # Add evaluations
    if add_evaluations(archived_path, original_path):
        print(f"   ✓ Added evaluations.npz")
        enhanced = True
    
    # Add checkpoints
    if add_checkpoints(archived_path, original_path, max_steps):
        print(f"   ✓ Added checkpoints")
        enhanced = True
    
    # Update README with metrics
    if update_readme_with_metrics(archived_path):
        print(f"   ✓ Updated README with evaluation metrics")
        enhanced = True
    
    if not enhanced:
        print(f"   ℹ️  Already complete or no enhancements needed")
    
    return enhanced


def main():
    parser = argparse.ArgumentParser(
        description="Enhance archived models with evaluations and checkpoints"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to enhance (folder name in archived_models)"
    )
    
    parser.add_argument(
        "--archived-dir",
        type=str,
        default="outputs/experiments/archived_models",
        help="Directory containing archived models"
    )
    
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="outputs/experiments/runs",
        help="Directory containing training runs"
    )
    
    args = parser.parse_args()
    
    archived_dir = Path(args.archived_dir)
    runs_dir = Path(args.runs_dir)
    
    if not archived_dir.exists():
        print(f"Error: Archived models directory not found: {archived_dir}")
        return 1
    
    print("=" * 80)
    print("Enhancing Archived Models")
    print("=" * 80)
    
    if args.model:
        # Enhance specific model
        model_path = archived_dir / args.model
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            return 1
        enhance_model(model_path, runs_dir)
    else:
        # Enhance all models
        models = [d for d in archived_dir.iterdir() if d.is_dir() and d.name != ".git"]
        print(f"Found {len(models)} archived models\n")
        
        enhanced_count = 0
        for model_path in models:
            if enhance_model(model_path, runs_dir):
                enhanced_count += 1
        
        print("\n" + "=" * 80)
        print(f"✓ Enhanced {enhanced_count} models")
        print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())

