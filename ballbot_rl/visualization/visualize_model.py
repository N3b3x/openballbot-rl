#!/usr/bin/env python3
"""
Visualize a trained ballbot model in MuJoCo.

Usage:
    # As CLI command (after installation):
    ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip
    
    # As Python module:
    python -m ballbot_rl.visualization.visualize_model --model_path outputs/experiments/runs/.../best_model/best_model.zip
    
    # With more episodes:
    ballbot-visualize-model --model_path .../best_model.zip --n_episodes 5
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from termcolor import colored

from stable_baselines3 import PPO, SAC
from stable_baselines3.common import policies as sb3_policies
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, ActorCriticPolicy
from ballbot_rl.training.utils import make_ballbot_env
from ballbot_gym.core.config import load_config, get_component_config


def visualize_model(model_path, n_episodes=3, gui=True, seed=42, keep_open=False):
    """
    Load a trained model and visualize it in MuJoCo.
    
    Args:
        model_path: Path to the trained model (.zip file)
        n_episodes: Number of episodes to visualize
        gui: Whether to show MuJoCo GUI
        seed: Random seed for reproducibility
        keep_open: If True, keep viewer open after episodes (for parallel viewing)
    """
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        print(colored(f"❌ Error: Model file not found: {model_path}", "red", attrs=["bold"]))
        return
    
    print(colored(f"📦 Loading model from: {model_path}", "cyan", attrs=["bold"]))
    
    # Check if this is a legacy model (models saved with older module structure)
    is_legacy = "legacy" in str(model_path).lower() or "salehi" in str(model_path).lower()
    
    # For legacy models, create a temporary 'policies' module alias if needed
    # This handles cases where the model was saved with a relative import
    if is_legacy and "policies" not in sys.modules:
        # Create an alias so 'policies' can be imported during deserialization
        sys.modules["policies"] = sb3_policies
        print(colored("🔧 Created 'policies' module alias for legacy model...", "yellow"))
    
    # Load model
    try:
        # Try to detect algorithm from file or default to PPO
        if is_legacy:
            # For legacy models, provide policy class explicitly via custom_objects
            print(colored("🔧 Detected legacy model, using custom_objects for loading...", "yellow"))
            custom_objects = {
                "policy_class": MultiInputActorCriticPolicy,
            }
            model = PPO.load(str(model_path), custom_objects=custom_objects)
        else:
            model = PPO.load(str(model_path))
        print(colored("✓ Model loaded successfully (PPO)", "green"))
    except Exception as e:
        error_msg = str(e).lower()
        is_policies_error = "no module named 'policies'" in error_msg or "cannot import name" in error_msg
        
        if is_policies_error and "policies" not in sys.modules:
            # Create the alias if we haven't already
            sys.modules["policies"] = sb3_policies
            print(colored("🔧 Created 'policies' module alias (detected from error)...", "yellow"))
        
        print(colored(f"⚠️  Failed to load as PPO, trying SAC: {e}", "yellow"))
        try:
            if is_legacy:
                # For legacy SAC models, try with ActorCriticPolicy
                custom_objects = {
                    "policy_class": ActorCriticPolicy,
                }
                model = SAC.load(str(model_path), custom_objects=custom_objects)
            else:
                model = SAC.load(str(model_path))
            print(colored("✓ Model loaded successfully (SAC)", "green"))
        except Exception as e2:
            # If still failing, try with explicit policy class even if not detected as legacy
            print(colored(f"⚠️  Retrying with explicit policy class...", "yellow"))
            try:
                custom_objects = {
                    "policy_class": MultiInputActorCriticPolicy,
                }
                model = PPO.load(str(model_path), custom_objects=custom_objects)
                print(colored("✓ Model loaded successfully (PPO with custom_objects)", "green"))
            except Exception as e3:
                print(colored(f"❌ Failed to load model: {e3}", "red", attrs=["bold"]))
                print(colored(f"   Original PPO error: {e}", "red"))
                print(colored(f"   Original SAC error: {e2}", "red"))
                return
    
    # Try to load training config if available
    # Check multiple possible locations:
    # 1. Same directory as model (for archived models: archived_models/MODEL_NAME/config.yaml)
    # 2. Parent directory (for training runs: runs/RUN_NAME/config.yaml)
    config_path = None
    possible_paths = [
        model_path.parent / "config.yaml",  # Same dir as model (archived models)
        model_path.parent.parent / "config.yaml",  # Parent dir (training runs)
    ]
    
    for path in possible_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path:
        config = load_config(str(config_path))
        terrain_config = get_component_config(config, "terrain")
        reward_config = get_component_config(config, "reward")
        env_config = {
            "camera": config.get("camera", {}),
            "env": config.get("env", {}),
            "logging": config.get("logging", {})
        }
        print(colored(f"✓ Loaded training config from: {config_path}", "green"))
        print(colored(f"   Terrain type: {terrain_config.get('type', 'unknown')}", "cyan"))
    else:
        # Use defaults
        terrain_config = {"type": "perlin", "config": {}}
        reward_config = {"type": "directional", "config": {"target_direction": [0.0, 1.0]}}
        env_config = None
        print(colored("⚠️  No config.yaml found, using defaults (perlin terrain)", "yellow"))
    
    # Extract model name for window title
    # Try to get a descriptive name from the path
    model_name = model_path.parent.name if model_path.parent.name != "best_model" else model_path.parent.parent.name
    terrain_type = terrain_config.get('type', 'unknown') if config_path else 'perlin'
    
    # Create a descriptive window title
    viewer_title = f"Ballbot RL - {model_name} ({terrain_type})"
    
    # Create environment with GUI
    print(colored(f"\n🎮 Creating environment with GUI={gui}...", "cyan", attrs=["bold"]))
    env_factory = make_ballbot_env(
        terrain_config=terrain_config,
        reward_config=reward_config,
        env_config=env_config,
        gui=gui,
        log_options={"cams": False, "reward_terms": False},
        seed=seed,
        eval_env=True,
        viewer_title=viewer_title if gui else None
    )
    env = env_factory()
    
    if gui and hasattr(env, 'passive_viewer') and env.passive_viewer is None:
        print(colored(
            "⚠️  Warning: MuJoCo viewer not available.\n"
            "   On macOS, you may need to use 'mjpython' instead of 'python':\n"
            "   mjpython visualize_model.py --model_path ...",
            "yellow", attrs=["bold"]
        ))
    
    print(colored(f"\n🎬 Running {n_episodes} episode(s) with deterministic policy...\n", "cyan", attrs=["bold"]))
    
    # Run episodes
    total_reward = 0.0
    episode_rewards = []
    
    # Set up signal handler for immediate Ctrl+C response
    # Use a list to allow modification from nested functions
    interrupt_flag = [False]
    
    def signal_handler(sig, frame):
        interrupt_flag[0] = True
        print(colored("\n\n⚠️  Interrupt received (Ctrl+C), exiting gracefully...", "yellow", attrs=["bold"]))
    
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        for episode in range(n_episodes):
            if interrupt_flag[0]:
                print(colored("   Exiting early due to interrupt...", "yellow"))
                break
            obs, info = env.reset(seed=seed + episode)
            episode_reward = 0.0
            step_count = 0
            done = False
            prev_step_counter = 0
            
            print(colored(f"Episode {episode + 1}/{n_episodes} starting...", "yellow"))
            
            while not done and step_count < 4000 and not interrupt_flag[0]:  # Max episode length
                if interrupt_flag[0]:
                    break
                
                # Use deterministic policy (no exploration)
                action, _ = model.predict(obs, deterministic=True)
                
                # Track step_counter before step to detect GUI resets
                prev_step_counter = getattr(env, 'step_counter', step_count)
                
                # Step environment (this may trigger GUI reset detection internally)
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except KeyboardInterrupt:
                    interrupt_flag[0] = True
                    break
                
                # Check if GUI reset happened (step_counter reset to 0)
                current_step_counter = getattr(env, 'step_counter', step_count + 1)
                if current_step_counter == 0 and prev_step_counter > 0 and step_count > 0:
                    # GUI reset detected - environment already reset, observation is fresh
                    print(colored("🔄 GUI Reset detected - episode restarted...", "cyan", attrs=["bold"]))
                    # Reset episode tracking
                    episode_reward = 0.0
                    step_count = 0
                    done = False
                    # Observation is already fresh from step() after reset
                    continue  # Skip reward accumulation for this step
                
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"  Step {step_count}: Reward={episode_reward:.2f}, Current reward={reward:.4f}")
            
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            print(colored(
                f"✓ Episode {episode + 1} completed: {step_count} steps, Total reward: {episode_reward:.2f}",
                "green", attrs=["bold"]
            ))
            
            if episode < n_episodes - 1 and not interrupt_flag[0]:
                try:
                    input(colored("\nPress Enter to continue to next episode (or Ctrl+C to exit)...", "cyan"))
                except (KeyboardInterrupt, EOFError):
                    interrupt_flag[0] = True
                    break
        
        avg_reward = total_reward / n_episodes if n_episodes > 0 else 0.0
        print(colored(
            f"\n📊 Summary: {n_episodes} episode(s), Average reward: {avg_reward:.2f}",
            "cyan", attrs=["bold"]
        ))
        
        if keep_open and not interrupt_flag[0]:
            print(colored("\n✓ Episodes complete. Viewer will stay open.", "green", attrs=["bold"]))
            print(colored("   💡 Tip: Click 'Reset' in MuJoCo GUI to run another episode!", "cyan", attrs=["bold"]))
            print(colored("   Press Ctrl+C or close the MuJoCo window to exit.", "yellow"))
            
            # Continue running policy if user resets via GUI
            # The environment's step() method already handles GUI reset detection,
            # so we just need to keep running the policy loop
            episode_count = n_episodes
            current_episode_reward = 0.0
            current_step_count = 0
            done = False
            
            try:
                import time
                
                # Start continuous mode - reset for first additional episode
                obs, info = env.reset(seed=seed + episode_count)
                episode_count += 1
                print(colored(f"\n🔄 Episode {episode_count} (continuous mode - click Reset in viewer to restart)...", "cyan", attrs=["bold"]))
                
                while not interrupt_flag[0]:
                    # Check if viewer is still alive
                    if hasattr(env, 'passive_viewer') and env.passive_viewer is None:
                        print(colored("   Viewer closed, exiting...", "yellow"))
                        break
                    
                    try:
                        # Run policy step
                        action, _ = model.predict(obs, deterministic=True)
                        
                        # Track step_counter before step to detect GUI resets
                        prev_step_counter = getattr(env, 'step_counter', current_step_count)
                        
                        # Step environment (this may trigger GUI reset detection internally)
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        # Check for interrupt after step (step might take time)
                        if interrupt_flag[0]:
                            print(colored("   Exiting...", "yellow"))
                            break
                        
                        # Check if GUI reset happened (step_counter reset to 0)
                        current_step_counter = getattr(env, 'step_counter', current_step_count + 1)
                        if current_step_counter == 0 and prev_step_counter > 0:
                            # GUI reset detected - start new episode
                            print(colored(f"\n🔄 GUI Reset detected - Episode {episode_count} starting...", "cyan", attrs=["bold"]))
                            if current_step_count > 0:
                                episode_rewards.append(current_episode_reward)
                                print(colored(
                                    f"   Previous episode: {current_step_count} steps, Reward: {current_episode_reward:.2f}",
                                    "green"
                                ))
                            episode_count += 1
                            current_episode_reward = 0.0
                            current_step_count = 0
                            done = False
                            # Observation is already fresh from step() after reset
                            continue  # Skip reward accumulation for this step
                        
                        current_episode_reward += reward
                        current_step_count += 1
                        done = terminated or truncated
                        
                        # Check termination
                        if done or current_step_count >= 4000:
                            episode_rewards.append(current_episode_reward)
                            print(colored(
                                f"✓ Episode {episode_count} completed: {current_step_count} steps, Reward: {current_episode_reward:.2f}",
                                "green", attrs=["bold"]
                            ))
                            print(colored("   Click 'Reset' in MuJoCo GUI to run another episode...", "cyan"))
                            # Reset for next episode (will wait for GUI reset or auto-reset)
                            current_episode_reward = 0.0
                            current_step_count = 0
                            done = False
                            # Auto-reset for next episode (user can also use GUI reset)
                            obs, info = env.reset(seed=seed + episode_count)
                            episode_count += 1
                        
                        # Small sleep to avoid busy-waiting (but keep it responsive)
                        time.sleep(0.01)
                    except KeyboardInterrupt:
                        interrupt_flag[0] = True
                        print(colored("\n   Interrupt received, exiting...", "yellow"))
                        break
                    
            except KeyboardInterrupt:
                interrupt_flag[0] = True
                print(colored("\n   Closing viewer...", "yellow"))
        else:
            # Default behavior: pause before closing so user can see the final state
            print(colored("\n✓ Episodes complete. Viewer will close in 5 seconds...", "green"))
            print(colored("   (Use --keep_open flag to keep viewer open indefinitely)", "yellow"))
            import time
            time.sleep(5)  # Give user time to see the final state
            print(colored("   Closing environment...", "yellow"))
    
    except KeyboardInterrupt:
        interrupt_flag[0] = True
        print(colored("\n\n⚠️  Interrupted by user (Ctrl+C)", "yellow", attrs=["bold"]))
        print(colored("   Closing environment...", "yellow"))
    except Exception as e:
        print(colored(f"\n❌ Error during visualization: {e}", "red", attrs=["bold"]))
        import traceback
        traceback.print_exc()
    finally:
        # Always close environment properly
        try:
            if 'env' in locals():
                if interrupt_flag[0]:
                    print(colored("   Cleaning up after interrupt...", "yellow"))
                env.close()
                if interrupt_flag[0]:
                    print(colored("   ✓ Environment closed", "green"))
        except Exception as e:
            print(colored(f"⚠️  Warning during cleanup: {e}", "yellow"))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained ballbot model in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize best model from a training run
  ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip
  
  # Or as Python module
  python -m ballbot_rl.visualization.visualize_model --model_path outputs/experiments/runs/.../best_model/best_model.zip
  
  # Visualize with 5 episodes
  ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip --n_episodes 5
  
  # Visualize without GUI (headless)
  ballbot-visualize-model --model_path outputs/experiments/runs/.../best_model/best_model.zip --no_gui
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=3,
        help="Number of episodes to visualize (default: 3)"
    )
    parser.add_argument(
        "--no_gui",
        action="store_true",
        help="Disable MuJoCo GUI (headless mode)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--keep_open",
        action="store_true",
        help="Keep MuJoCo viewer open after episodes complete (for parallel viewing)"
    )
    
    args = parser.parse_args()
    
    visualize_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        gui=not args.no_gui,
        seed=args.seed,
        keep_open=args.keep_open
    )


if __name__ == "__main__":
    main()

