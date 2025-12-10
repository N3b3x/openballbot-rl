#!/usr/bin/env python3
"""
Visualize all archived models and show their training progress reports.

This script:
1. Lists all archived models (excluding legacy)
2. Shows training progress summary for each
3. Optionally plots training curves
4. Optionally visualizes models in MuJoCo

Usage:
    # Show progress reports for all models
    python scripts/utils/visualize_all_archived_models.py
    
    # Show progress and plot training curves (interactive display)
    python scripts/utils/visualize_all_archived_models.py --plot
    
    # Save plots without displaying them (reward over training steps only)
    python scripts/utils/visualize_all_archived_models.py --save-plots
    
    # Visualize models in MuJoCo (interactive)
    python scripts/utils/visualize_all_archived_models.py --visualize
    
    # Do everything: progress, plots, and visualization
    python scripts/utils/visualize_all_archived_models.py --plot --visualize
    
    # Save plots and visualize (plots saved, then visualize)
    python scripts/utils/visualize_all_archived_models.py --save-plots --visualize
    
    # Specific model only
    python scripts/utils/visualize_all_archived_models.py --model 2025-12-04_ppo-flat-directional-seed10 --save-plots
"""

import argparse
import csv
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

from termcolor import colored

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print(colored("⚠️  matplotlib not available, plotting disabled", "yellow"))


def load_progress_csv(csv_path: Path) -> Dict:
    """Load and parse progress.csv file."""
    if not csv_path.exists():
        return None
    
    data = {
        "eval_rewards": [],
        "eval_timesteps": [],
        "train_rewards": [],
        "train_timesteps": [],
        "max_eval_reward": None,
        "final_eval_reward": None,
        "max_train_reward": None,
        "final_train_reward": None,
        "total_timesteps": 0,
        "num_eval_points": 0,
        "num_train_points": 0
    }
    
    try:
        with open(csv_path, 'r') as f:
            first_line = f.readline()
            headers = first_line.lstrip('#').strip().split(',')
            reader = csv.DictReader(f, fieldnames=headers)
            
            row_counter = 0
            for row in reader:
                if not row_counter:
                    row_counter += 1
                    continue  # Skip first row (often incomplete)
                
                timesteps = row.get("time/total_timesteps", "")
                if timesteps:
                    timesteps = float(timesteps)
                    data["total_timesteps"] = max(data["total_timesteps"], timesteps)
                
                # Evaluation data
                eval_reward = row.get("eval/mean_reward", "")
                if eval_reward:
                    data["eval_rewards"].append(float(eval_reward))
                    data["eval_timesteps"].append(timesteps if timesteps else 0)
                    data["num_eval_points"] += 1
                
                # Training data
                train_reward = row.get("rollout/ep_rew_mean", "")
                if train_reward:
                    data["train_rewards"].append(float(train_reward))
                    data["train_timesteps"].append(timesteps if timesteps else 0)
                    data["num_train_points"] += 1
                
                row_counter += 1
        
        # Calculate max and final values
        if data["eval_rewards"]:
            data["max_eval_reward"] = max(data["eval_rewards"])
            data["final_eval_reward"] = data["eval_rewards"][-1]
        
        if data["train_rewards"]:
            data["max_train_reward"] = max(data["train_rewards"])
            data["final_train_reward"] = data["train_rewards"][-1]
        
        return data
    except Exception as e:
        print(colored(f"⚠️  Error reading {csv_path}: {e}", "yellow"))
        return None


def print_progress_summary(model_name: str, progress_data: Dict, config: Dict = None):
    """Print a summary of training progress."""
    print(colored(f"\n{'='*80}", "cyan", attrs=["bold"]))
    print(colored(f"📊 {model_name}", "cyan", attrs=["bold"]))
    print(colored(f"{'='*80}", "cyan"))
    
    if config:
        terrain_type = config.get("problem", {}).get("terrain", {}).get("type", "unknown")
        reward_type = config.get("problem", {}).get("reward", {}).get("type", "unknown")
        total_timesteps = config.get("total_timesteps", "unknown")
        print(colored(f"   Terrain: {terrain_type}", "white"))
        print(colored(f"   Reward: {reward_type}", "white"))
        print(colored(f"   Target Steps: {total_timesteps}", "white"))
    
    if progress_data:
        print(colored(f"\n   Training Progress:", "yellow", attrs=["bold"]))
        print(colored(f"   Total Timesteps: {progress_data['total_timesteps']:,}", "white"))
        print(colored(f"   Evaluation Points: {progress_data['num_eval_points']}", "white"))
        print(colored(f"   Training Points: {progress_data['num_train_points']}", "white"))
        
        if progress_data["max_eval_reward"] is not None:
            print(colored(f"\n   Evaluation Rewards:", "green", attrs=["bold"]))
            print(colored(f"   Max: {progress_data['max_eval_reward']:.2f}", "green"))
            print(colored(f"   Final: {progress_data['final_eval_reward']:.2f}", "green"))
        
        if progress_data["max_train_reward"] is not None:
            print(colored(f"\n   Training Rewards:", "blue", attrs=["bold"]))
            print(colored(f"   Max: {progress_data['max_train_reward']:.2f}", "blue"))
            print(colored(f"   Final: {progress_data['final_train_reward']:.2f}", "blue"))
    else:
        print(colored("   ⚠️  No progress data available", "yellow"))


def plot_training_progress(model_name: str, progress_data: Dict, output_dir: Path = None, save_only: bool = False):
    """Plot training progress curves.
    
    Args:
        model_name: Name of the model
        progress_data: Dictionary with training progress data
        output_dir: Directory to save plots (if None, plots are only displayed)
        save_only: If True, save plot without displaying it
    """
    if not HAS_MATPLOTLIB:
        print(colored("   ⚠️  matplotlib not available, skipping plot", "yellow"))
        return
    
    if not progress_data or not progress_data["eval_rewards"]:
        print(colored("   ⚠️  No evaluation data to plot", "yellow"))
        return
    
    # Create single plot for reward over training steps (no histogram)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot rewards over training steps
    if progress_data["train_rewards"]:
        ax.plot(progress_data["train_timesteps"], progress_data["train_rewards"], 
                label="Training", color="blue", linewidth=2, alpha=0.7)
    ax.plot(progress_data["eval_timesteps"], progress_data["eval_rewards"], 
            label="Evaluation", color="darkorange", linewidth=2)
    ax.set_xlabel("Total Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title(f"{model_name}\nReward Progress", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_path = output_dir / f"{model_name}_progress.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(colored(f"   ✓ Saved plot to: {output_path}", "green"))
    
    # Display plot only if not save_only
    if not save_only:
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to allow plot to render
    else:
        plt.close(fig)  # Close figure to free memory when saving only


def visualize_model_in_mujoco(model_path: Path, background: bool = False, keep_open: bool = False):
    """Visualize a model in MuJoCo.
    
    Args:
        model_path: Path to the model file
        background: If True, launch in background process (non-blocking)
        keep_open: If True, use --keep_open flag to keep viewer open
    """
    is_macos = platform.system() == 'Darwin'
    python_cmd = "mjpython" if is_macos else "python"
    
    print(colored(f"\n   🎮 Launching MuJoCo visualization...", "cyan"))
    print(colored(f"   Command: {python_cmd} -m ballbot_rl.visualization.visualize_model --model_path {model_path}", "white"))
    
    import subprocess
    try:
        cmd = [
            python_cmd, "-m", "ballbot_rl.visualization.visualize_model",
            "--model_path", str(model_path),
            "--n_episodes", "1"
        ]
        
        if background or keep_open:
            cmd.append("--keep_open")
        
        if background:
            # Launch in background (non-blocking) - keeps viewer open
            # Redirect stdout/stderr to avoid cluttering terminal with interleaved output
            # The MuJoCo GUI will still show, just not the console output
            import os
            devnull = open(os.devnull, 'w')
            try:
                # Create new process group on Unix systems for better signal handling
                preexec_fn = os.setsid if hasattr(os, 'setsid') else None
                process = subprocess.Popen(
                    cmd,
                    stdout=devnull,
                    stderr=devnull,
                    preexec_fn=preexec_fn
                )
                # Note: devnull will be closed when process exits
                # Don't close devnull here - let it be closed when process exits
                # Closing it immediately can cause issues on some systems
                print(colored(f"   ✓ Launched in background (PID: {process.pid})", "green"))
                return process
            except Exception as e:
                devnull.close()
                raise
        else:
            # Blocking call (waits for viewer to close)
            subprocess.run(cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(colored(f"   ⚠️  Visualization failed: {e}", "yellow"))
        return None
    except FileNotFoundError:
        if is_macos:
            print(colored(f"   ⚠️  mjpython not found. Please install MuJoCo and ensure mjpython is in PATH", "yellow"))
        else:
            print(colored(f"   ⚠️  python not found", "yellow"))
        return None


def process_archived_models(
    archived_dir: Path,
    model_filter: Optional[str] = None,
    plot: bool = False,
    visualize: bool = False,
    exclude_legacy: bool = True,
    parallel: bool = False,
    args=None
):
    """Process all archived models."""
    models = []
    
    for model_dir in archived_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Skip legacy if requested
        if exclude_legacy and "legacy" in model_name.lower():
            continue
        
        # Filter by name if specified
        if model_filter and model_filter not in model_name:
            continue
        
        model_path = model_dir / "best_model.zip"
        config_path = model_dir / "config.yaml"
        progress_path = model_dir / "progress.csv"
        
        if not model_path.exists():
            continue
        
        models.append({
            "name": model_name,
            "path": model_dir,
            "model_path": model_path,
            "config_path": config_path,
            "progress_path": progress_path
        })
    
    # Sort by name
    models.sort(key=lambda x: x["name"])
    
    print(colored(f"\n{'='*80}", "cyan", attrs=["bold"]))
    print(colored(f"Found {len(models)} archived model(s) to process", "cyan", attrs=["bold"]))
    print(colored(f"{'='*80}", "cyan"))
    
    # Create output directory for plots
    output_dir = None
    save_only = hasattr(args, 'save_plots_only') and args.save_plots_only
    if (plot or save_only) and HAS_MATPLOTLIB:
        output_dir = archived_dir / "progress_plots"
        output_dir.mkdir(exist_ok=True)
        if save_only:
            print(colored(f"\n📁 Saving plots to: {output_dir}", "cyan", attrs=["bold"]))
    
    for i, model_info in enumerate(models, 1):
        print(colored(f"\n[{i}/{len(models)}] Processing: {model_info['name']}", "yellow", attrs=["bold"]))
        
        # Load config
        config = None
        if model_info["config_path"].exists():
            try:
                with open(model_info["config_path"]) as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                print(colored(f"   ⚠️  Error loading config: {e}", "yellow"))
        
        # Load progress data
        progress_data = load_progress_csv(model_info["progress_path"])
        
        # Print summary
        print_progress_summary(model_info["name"], progress_data, config)
        
        # Plot if requested (or always show if not visualizing, or show before visualizing)
        save_only = hasattr(args, 'save_plots_only') and args.save_plots_only
        if plot or (not visualize) or save_only:
            if save_only:
                print(colored(f"\n   💾 Saving training curves plot...", "cyan"))
            else:
                print(colored(f"\n   📈 Plotting training curves...", "cyan"))
            plot_training_progress(model_info["name"], progress_data, output_dir, save_only=save_only)
        elif visualize and progress_data:
            # If visualizing without --plot, at least show a quick summary
            print(colored(f"\n   💡 Tip: Use --plot flag to see detailed training curves", "cyan"))
        
        # Visualize if requested
        if visualize and parallel and i == 1:
            # Launch all in parallel (background processes) - do this once at the start
            print(colored(f"\n   🚀 Launching all models in parallel mode (all viewers will open)...", "cyan", attrs=["bold"]))
            processes = []
            for model_info in models:
                print(colored(f"   Launching: {model_info['name']}", "yellow"))
                proc = visualize_model_in_mujoco(model_info["model_path"], background=True)
                if proc:
                    processes.append((model_info["name"], proc))
                import time
                time.sleep(0.5)  # Small delay between launches
            
            print(colored(f"\n✓ Launched {len(processes)} viewer(s) in parallel", "green", attrs=["bold"]))
            print(colored("   All MuJoCo viewers should now be open. Close them manually when done.", "cyan"))
            print(colored("   Press Ctrl+C here to exit (viewers will stay open).", "yellow"))
            
            # Wait for user interrupt
            try:
                import time
                import signal
                
                interrupted = False
                def signal_handler(sig, frame):
                    nonlocal interrupted
                    interrupted = True
                    print(colored("\n\n⚠️  Interrupt received, terminating processes...", "yellow", attrs=["bold"]))
                
                signal.signal(signal.SIGINT, signal_handler)
                
                while True:
                    if interrupted:
                        break
                    time.sleep(0.5)
                    # Check if any processes are still running
                    # poll() returns None if process is still running, otherwise returns exit code
                    if all(p.poll() is not None for _, p in processes):
                        break
            except KeyboardInterrupt:
                interrupted = True
                print(colored("\n   Exiting script. Viewers will remain open.", "yellow"))
            
            # Terminate all processes if interrupted
            if interrupted:
                print(colored("\n   Terminating all visualization processes...", "yellow", attrs=["bold"]))
                import os
                import signal
                import time
                
                terminated_count = 0
                for name, proc in processes:
                    try:
                        if proc.poll() is None:  # Process is still running
                            # Try graceful termination first
                            try:
                                if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                                    # Kill entire process group (includes children)
                                    pgid = os.getpgid(proc.pid)
                                    os.killpg(pgid, signal.SIGTERM)
                                else:
                                    proc.terminate()
                            except (OSError, ProcessLookupError, AttributeError):
                                # Fallback to direct termination
                                try:
                                    proc.terminate()
                                except:
                                    pass
                            
                                            # Wait a bit for graceful shutdown
                            try:
                                import subprocess
                                proc.wait(timeout=1)
                                terminated_count += 1
                                print(colored(f"   ✓ Terminated: {name}", "green"))
                            except subprocess.TimeoutExpired:
                                # Force kill if still running
                                try:
                                    if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                                        pgid = os.getpgid(proc.pid)
                                        os.killpg(pgid, signal.SIGKILL)
                                    else:
                                        proc.kill()
                                    proc.wait(timeout=0.5)
                                    terminated_count += 1
                                    print(colored(f"   ✓ Force killed: {name}", "yellow"))
                                except (ProcessLookupError, OSError):
                                    # Already dead
                                    terminated_count += 1
                                    print(colored(f"   ✓ Already terminated: {name}", "green"))
                    except (ProcessLookupError, OSError):
                        # Process already terminated
                        terminated_count += 1
                        print(colored(f"   ✓ Already terminated: {name}", "green"))
                    except Exception as e:
                        print(colored(f"   ⚠️  Could not terminate {name}: {e}", "yellow"))
                        try:
                            proc.kill()
                        except:
                            pass
                
                if terminated_count > 0:
                    print(colored(f"\n   ✓ Terminated {terminated_count}/{len(processes)} process(es)", "green", attrs=["bold"]))
                else:
                    print(colored(f"\n   ⚠️  No processes were running", "yellow"))
            
            break  # Exit the loop since we processed all models
        elif visualize and not parallel:
            # Sequential visualization (one at a time)
            # Show plots first if available (unless save_only mode)
            if progress_data and not plot and not save_only:
                print(colored(f"\n   📈 Showing training curves before visualization...", "cyan"))
                plot_training_progress(model_info["name"], progress_data, output_dir, save_only=False)
                input(colored("\n   Press Enter to continue to visualization...", "cyan"))
            
            response = input(colored(f"\n   Visualize this model in MuJoCo? [y/N]: ", "cyan"))
            if response.lower() == 'y':
                visualize_model_in_mujoco(model_info["model_path"], background=False, keep_open=True)
        
        # Pause between models (except last)
        if i < len(models) and not (visualize and parallel):
            if not visualize:  # If visualizing, user already interacted
                input(colored("\n   Press Enter to continue to next model...", "cyan"))
    
    print(colored(f"\n{'='*80}", "cyan", attrs=["bold"]))
    print(colored(f"✓ Processed {len(models)} model(s)", "green", attrs=["bold"]))
    if plot and output_dir:
        print(colored(f"✓ Plots saved to: {output_dir}", "green"))
    print(colored(f"{'='*80}", "cyan"))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all archived models and show training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show progress summaries for all models
  python scripts/utils/visualize_all_archived_models.py
  
  # Show progress and plot training curves (interactive display)
  python scripts/utils/visualize_all_archived_models.py --plot
  
  # Save plots without displaying them (reward over training steps only)
  python scripts/utils/visualize_all_archived_models.py --save-plots
  
  # Visualize models in MuJoCo (interactive)
  python scripts/utils/visualize_all_archived_models.py --visualize
  
  # Do everything: progress, plots, and visualization
  python scripts/utils/visualize_all_archived_models.py --plot --visualize
  
  # Save plots and visualize (plots saved, then visualize)
  python scripts/utils/visualize_all_archived_models.py --save-plots --visualize
  
  # Specific model only
  python scripts/utils/visualize_all_archived_models.py --model 2025-12-04_ppo-flat-directional-seed10 --save-plots
        """
    )
    
    parser.add_argument(
        "--archived-dir",
        type=str,
        default="outputs/experiments/archived_models",
        help="Directory containing archived models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter to specific model (partial name match)"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot training progress curves (display interactively)"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        dest="save_plots_only",
        help="Save training progress plots without displaying them (reward over training steps only)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize models in MuJoCo (interactive)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Launch all visualizations in parallel (all viewers open simultaneously)"
    )
    
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy Salehi model"
    )
    
    args = parser.parse_args()
    
    archived_dir = Path(args.archived_dir)
    if not archived_dir.exists():
        print(colored(f"❌ Error: Archived models directory not found: {archived_dir}", "red", attrs=["bold"]))
        return 1
    
    process_archived_models(
        archived_dir=archived_dir,
        model_filter=args.model,
        plot=args.plot,
        visualize=args.visualize,
        exclude_legacy=not args.include_legacy,
        parallel=args.parallel,
        args=args
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
