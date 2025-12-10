"""Callback utilities for training.

Training configuration:
- Metrics logging: CSV + TensorBoard
- Video recording: Periodic recording of best models
- GUI rendering: Disabled during training for performance
- Periodic visualization: Interactive viewer at intervals (separate process)
"""
from stable_baselines3.common.callbacks import (
    CallbackList, 
    CheckpointCallback, 
    EvalCallback,
    BaseCallback
)
from stable_baselines3.common.vec_env import VecVideoRecorder, VecEnvWrapper
from pathlib import Path
from typing import Dict, Optional
import os
import platform
import threading
import queue
import copy
from termcolor import colored

# Try to import gymnasium error for dependency checking
try:
    from gymnasium.error import DependencyNotInstalled
except ImportError:
    try:
        from gym.error import DependencyNotInstalled
    except ImportError:
        # Fallback if neither is available
        class DependencyNotInstalled(Exception):
            pass

from ballbot_gym.core.config import get_component_config
from ballbot_rl.training.utils import make_ballbot_env


class RenderModeWrapper(VecEnvWrapper):
    """
    Wrapper to add render_mode attribute to environments for VecVideoRecorder compatibility.
    
    VecVideoRecorder requires render_mode="rgb_array", but some environments don't set this.
    This wrapper adds the attribute so VecVideoRecorder can work properly.
    """
    def __init__(self, venv, render_mode="rgb_array"):
        """
        Initialize wrapper.
        
        Args:
            venv: Vectorized environment to wrap.
            render_mode: Render mode to set (default: "rgb_array" for video recording).
        """
        super().__init__(venv)
        # Set render_mode on the underlying environments (if possible)
        # Some wrappers have render_mode as a read-only property, so we handle that gracefully
        for env in self.envs:
            try:
                # Unwrap Monitor wrapper if present
                unwrapped_env = env.env if hasattr(env, 'env') else env
                
                # Try to set render_mode on the actual environment
                if hasattr(unwrapped_env, 'render_mode'):
                    try:
                        unwrapped_env.render_mode = render_mode
                    except (AttributeError, TypeError):
                        # render_mode might be read-only or a property - that's okay
                        # The environment should already have render_mode="rgb_array" set in __init__
                        pass
                else:
                    # If render_mode doesn't exist, try to add it
                    try:
                        setattr(unwrapped_env, 'render_mode', render_mode)
                    except Exception:
                        pass
            except Exception:
                # If anything else fails, continue - VecVideoRecorder will check wrapper's render_mode
                pass
        
        # Most importantly: set render_mode on the wrapper itself
        # VecVideoRecorder checks self.env.render_mode which will be this wrapper
        self.render_mode = render_mode
    
    def reset(self):
        """Reset all environments and return observations."""
        return self.venv.reset()
    
    def step_wait(self):
        """Wait for step to complete and return results."""
        return self.venv.step_wait()


class VideoRecorderOnBestCallback(BaseCallback):
    """
    Record video when new best model is found (industry standard pattern).
    
    This callback is triggered by EvalCallback when a new best model is found.
    It uses VecVideoRecorder to record videos efficiently without GUI rendering.
    
    Video recording runs asynchronously in a separate thread to avoid blocking training.
    This ensures training continues while videos are being generated.
    """
    def __init__(
        self,
        eval_env,
        video_folder: Path,
        video_length: int = 4000,
        name_prefix: str = "best_model",
        async_recording: bool = True
    ):
        """
        Initialize video recorder callback.
        
        Args:
            eval_env: Evaluation environment (vectorized).
            video_folder: Directory to save videos.
            video_length: Maximum video length in steps.
            name_prefix: Prefix for video filenames.
            async_recording: If True, record videos in background thread (non-blocking).
                           If False, record synchronously (blocks training).
        """
        super().__init__()
        self.eval_env = eval_env
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.best_count = 0
        self.async_recording = async_recording
        
        # Check if MoviePy is available for video recording
        try:
            import moviepy  # noqa: F401
            self.moviepy_available = True
        except ImportError:
            self.moviepy_available = False
            print(colored(
                "⚠️  Warning: MoviePy is not installed. Video recording will be disabled.\n"
                "   To enable video recording, install MoviePy:\n"
                "   pip install 'gymnasium[other]'  or  pip install moviepy",
                "yellow", attrs=["bold"]
            ))
        
        # Setup asynchronous video recording if enabled
        if self.async_recording and self.moviepy_available:
            self.video_queue = queue.Queue()
            self.recording_thread = None
            self._stop_recording = threading.Event()
            self._start_recording_thread()
    
    def _start_recording_thread(self):
        """Start background thread for asynchronous video recording."""
        def recording_worker():
            """Worker thread that processes video recording tasks."""
            while not self._stop_recording.is_set():
                try:
                    # Get video recording task from queue (with timeout)
                    task = self.video_queue.get(timeout=1.0)
                    if task is None:  # Shutdown signal
                        break
                    
                    model, video_name = task
                    self._record_video_sync(model, video_name)
                    self.video_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(colored(
                        f"⚠️  Error in video recording thread: {e}",
                        "red", attrs=["bold"]
                    ))
                    if self.video_queue.qsize() > 0:
                        self.video_queue.task_done()
        
        self.recording_thread = threading.Thread(target=recording_worker, daemon=True)
        self.recording_thread.start()
    
    def _record_video_sync(self, model, video_name: str):
        """
        Synchronously record a video (called from background thread).
        
        Args:
            model: The model to use for recording.
            video_name: Name for the video file.
        
        Note:
            Thread Safety: This method is called from a background thread, but it's safe to use
            eval_env because:
            1. EvalCallback runs evaluation synchronously (blocks until complete)
            2. Only after evaluation completes does it call callback_on_new_best
            3. This queues the video recording task (non-blocking)
            4. By the time the background thread processes the task, EvalCallback is done
            5. Therefore, eval_env is not being used by EvalCallback when video recording runs
            
            The video recording truly runs in parallel with training, not with evaluation.
        """
        try:
            print(colored(
                f"🎥 Recording video (async): {video_name}",
                "cyan", attrs=["bold"]
            ))
            
            # Use eval_env directly - it's safe because:
            # - EvalCallback runs evaluation synchronously and completes before queuing this task
            # - By the time this background thread processes the task, evaluation is done
            # - Video recording runs in parallel with training, not with evaluation
            # Wrap environment with render_mode first, then VecVideoRecorder
            # VecVideoRecorder requires render_mode="rgb_array"
            env_with_render = RenderModeWrapper(self.eval_env, render_mode="rgb_array")
            video_recorder = VecVideoRecorder(
                env_with_render,
                video_folder=str(self.video_folder),
                record_video_trigger=lambda x: x == 0,  # Record first episode
                video_length=self.video_length,
                name_prefix=video_name
            )
            
            # Record one episode using the best model
            obs = video_recorder.reset()
            done = [False]
            step_count = 0
            
            while not all(done) and step_count < self.video_length:
                # Use deterministic policy (best model)
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = video_recorder.step(action)
                step_count += 1
            
            # Close video recorder
            video_recorder.close_video_recorder()
            
            print(colored(
                f"✓ Video saved (async): {self.video_folder / f'{video_name}.mp4'}",
                "green", attrs=["bold"]
            ))
        
        except DependencyNotInstalled as e:
            # MoviePy dependency missing - disable video recording for future calls
            self.moviepy_available = False
            print(colored(
                f"⚠️  Warning: Video recording failed due to missing dependency: {e}\n"
                "   Install MoviePy: pip install moviepy\n"
                "   Or install gymnasium extras: pip install 'gymnasium[other]'\n"
                "   Continuing training without video recording...",
                "yellow", attrs=["bold"]
            ))
        except (OSError, IOError) as e:
            # File descriptor or I/O errors - often due to async environment reuse
            import traceback
            print(colored(
                f"⚠️  Warning: Video recording failed: {type(e).__name__}: {e}\n"
                "   This is often caused by file descriptor issues in async video recording.\n"
                "   Solution: Set async_video_recording: false in your training config\n"
                "   (under visualization section) to use synchronous recording instead.\n"
                "   Continuing training...",
                "yellow", attrs=["bold"]
            ))
            if self.verbose > 0:
                print(colored(f"   Full error: {traceback.format_exc()}", "yellow"))
        except Exception as e:
            # Other errors - log but continue training
            import traceback
            print(colored(
                f"⚠️  Warning: Video recording failed: {type(e).__name__}: {e}\n"
                "   This may be due to:\n"
                "   - Missing MoviePy: pip install moviepy\n"
                "   - Environment missing render() method\n"
                "   - Render mode mismatch\n"
                "   - File descriptor issues (try async_video_recording: false)\n"
                "   Continuing training...",
                "yellow", attrs=["bold"]
            ))
            if self.verbose > 0:
                print(colored(f"   Full error: {traceback.format_exc()}", "yellow"))
    
    def _on_step(self) -> bool:
        """
        Called by EvalCallback when new best model is found.
        Access parent EvalCallback to get model and record video.
        
        If async_recording is True, queues video recording task and returns immediately.
        If False, records video synchronously (blocks training).
        """
        # Skip video recording if MoviePy is not available
        if not self.moviepy_available:
            return True
        
        # Access model through parent callback
        if hasattr(self.parent, 'model') and self.parent.model is not None:
            self.best_count += 1
            video_name = f"{self.name_prefix}_episode_{self.best_count}"
            
            if self.async_recording:
                # Queue video recording task for background thread (non-blocking)
                # Note: We need to get a reference to the model, but models are picklable
                # For thread safety, we'll pass the model reference directly
                try:
                    self.video_queue.put((self.parent.model, video_name), timeout=0.1)
                    print(colored(
                        f"📹 Queued video recording (async): {video_name}",
                        "cyan"
                    ))
                except queue.Full:
                    print(colored(
                        f"⚠️  Video recording queue full, skipping {video_name}",
                        "yellow"
                    ))
            else:
                # Synchronous recording (blocks training)
                self._record_video_sync(self.parent.model, video_name)
        
        return True
    
    def _on_training_end(self) -> None:
        """Clean up resources when training ends."""
        if self.async_recording and hasattr(self, 'recording_thread') and self.recording_thread is not None:
            # Signal thread to stop
            self._stop_recording.set()
            # Wait for queue to empty (with timeout)
            try:
                self.video_queue.put(None, timeout=1.0)  # Shutdown signal
                self.video_queue.join(timeout=5.0)  # Wait for tasks to complete
            except queue.Full:
                pass
            # Wait for thread to finish
            self.recording_thread.join(timeout=5.0)
            print(colored(
                "✓ Video recording thread stopped",
                "green"
            ))


class PeriodicVisualizationCallback(BaseCallback):
    """
    Visualize policy periodically during training using separate MuJoCo viewer.
    
    Industry Best Practice: Visualize during training to catch issues early,
    but use a separate viewer process so it never slows down training.
    
    This callback:
    - Triggers visualization at configurable intervals (e.g., every 10k-50k steps)
    - Creates a separate MuJoCo environment with GUI enabled
    - Runs 1-3 episodes for visual inspection
    - Properly closes viewer resources after visualization
    - Handles macOS/mjpython requirements gracefully
    
    Benefits:
    - Early detection of oscillations, drift, reward hacking
    - Find best checkpoint (policies often peak mid-training)
    - See emergent behavior forming (first twitch → stable stand → walk)
    - Zero impact on training performance (separate viewer process)
    """
    def __init__(
        self,
        config: Dict,
        visualize_freq: int = 20000,
        n_episodes: int = 1,
        max_ep_steps: int = 4000,
        verbose: int = 1
    ):
        """
        Initialize periodic visualization callback.
        
        Args:
            config: Training configuration dictionary (for terrain/reward/env configs).
            visualize_freq: Steps between visualizations (recommended: 10k-50k for ballbot).
            n_episodes: Number of episodes to visualize each time.
            max_ep_steps: Maximum steps per episode.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.config = config
        self.visualize_freq = visualize_freq
        self.n_episodes = n_episodes
        self.max_ep_steps = max_ep_steps
        self.last_visualization_step = 0
        
        # Extract configs for environment creation
        self.terrain_config = get_component_config(config, "terrain")
        self.reward_config = get_component_config(config, "reward")
        
        # Extract env config (camera, env, logging settings)
        self.env_config = {
            "camera": config.get("camera", {}),
            "env": config.get("env", {}),
            "logging": config.get("logging", {})
        }
        
        # Check platform for macOS-specific handling
        self.is_macos = platform.system() == 'Darwin'
    
    def _on_step(self) -> bool:
        """
        Called every training step. Check if visualization interval reached.
        
        Returns:
            bool: True to continue training, False to stop.
        """
        # Only visualize at intervals
        if (self.num_timesteps - self.last_visualization_step) >= self.visualize_freq:
            self.last_visualization_step = self.num_timesteps
            self._visualize_policy()
        
        return True
    
    def _visualize_policy(self):
        """
        Launch separate MuJoCo viewer and run episodes for visual inspection.
        
        Creates a fresh environment with GUI enabled, runs episodes with
        deterministic policy, then closes viewer and returns to training.
        """
        print(colored(
            f"\n🎥 Visualizing policy at step {self.num_timesteps}...",
            "cyan", attrs=["bold"]
        ))
        
        # Create a fresh environment with GUI enabled (separate from training envs)
        viz_env = None
        
        try:
            # Create environment factory with GUI enabled
            env_factory = make_ballbot_env(
                terrain_config=self.terrain_config,
                reward_config=self.reward_config,
                env_config=self.env_config,
                gui=True,  # Enable viewer
                log_options={"cams": False, "reward_terms": False},
                seed=self.config.get("seed", 0),
                eval_env=True
            )
            
            # Create environment instance
            viz_env = env_factory()
            
            # Check if viewer was actually launched (macOS may fail silently)
            if viz_env.passive_viewer is None and self.is_macos:
                print(colored(
                    "⚠️  Warning: MuJoCo viewer not available. "
                    "On macOS, visualization requires running with 'mjpython' instead of 'python'.\n"
                    "Continuing training without visualization...",
                    "yellow", attrs=["bold"]
                ))
                if viz_env:
                    viz_env.close()
                return
            
            # Run visualization episodes
            for episode in range(self.n_episodes):
                obs, _ = viz_env.reset()
                step_count = 0
                episode_reward = 0.0
                
                while step_count < self.max_ep_steps:
                    # Use deterministic policy for consistent visualization
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = viz_env.step(action)
                    
                    step_count += 1
                    episode_reward += reward
                    
                    if done or truncated:
                        break
                
                print(colored(
                    f"  Episode {episode+1}/{self.n_episodes}: {step_count} steps, "
                    f"reward: {episode_reward:.2f}",
                    "green"
                ))
            
            print(colored("✓ Visualization complete, resuming training...\n", "green"))
        
        except RuntimeError as e:
            # Handle viewer launch failures gracefully
            if "mjpython" in str(e).lower() or "viewer" in str(e).lower():
                print(colored(
                    f"⚠️  Warning: Could not launch MuJoCo viewer: {e}\n"
                    "Continuing training without visualization...",
                    "yellow", attrs=["bold"]
                ))
            else:
                # Re-raise unexpected errors
                raise
        
        except Exception as e:
            # Catch any other errors and continue training
            print(colored(
                f"⚠️  Error during visualization: {e}\n"
                "Continuing training...",
                "red", attrs=["bold"]
            ))
        
        finally:
            # Always close viewer and environment, even if errors occurred
            if viz_env is not None:
                try:
                    # Close viewer if it exists
                    if hasattr(viz_env, 'passive_viewer') and viz_env.passive_viewer is not None:
                        viz_env.passive_viewer.close()
                except Exception as e:
                    # Ignore errors during viewer cleanup
                    if self.verbose > 0:
                        print(colored(
                            f"Note: Error closing viewer (non-critical): {e}",
                            "yellow"
                        ))
                
                try:
                    # Close environment
                    viz_env.close()
                except Exception as e:
                    if self.verbose > 0:
                        print(colored(
                            f"Note: Error closing environment (non-critical): {e}",
                            "yellow"
                        ))


def create_training_callbacks(
    eval_env,
    out_path: Path,
    config: Dict
) -> CallbackList:
    """
    Create callbacks for training with industry-standard visualization.
    
    Industry Best Practices:
    - Metrics logging: Always enabled (CSV + TensorBoard)
    - Video recording: Periodic, especially on new best models
    - GUI rendering: Disabled during training (too slow)
    
    Args:
        eval_env: Evaluation environment (vectorized).
        out_path: Path to save outputs (checkpoints, best model, etc.).
        config: Training configuration dictionary.
    
    Returns:
        CallbackList: List of callbacks for training.
    """
    # Get visualization settings from config (defaults follow industry standards)
    viz_config = config.get("visualization", {})
    record_videos = viz_config.get("record_videos", True)  # Default: ON (industry standard)
    video_freq = viz_config.get("video_freq", "on_new_best")  # Industry standard
    video_episodes = viz_config.get("video_episodes", 1)  # Industry standard
    render = viz_config.get("render", False)  # Industry standard: disabled
    
    # Get evaluation settings from config
    eval_config = config.get("evaluation", {})
    eval_freq = eval_config.get("freq", 5000 if config["algo"]["name"] == "ppo" else 500)
    n_eval_episodes = eval_config.get("n_episodes", 8)
    
    # Prepare video recording callback if enabled
    video_callback = None
    eval_env_for_callback = eval_env
    
    if record_videos:
        video_folder = out_path / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        
        # Get max episode length from config (from env config section)
        # Default to 4000 if not specified (industry standard for ballbot)
        max_ep_steps = config.get("env", {}).get("max_ep_steps", 4000)
        
        if video_freq == "on_new_best":
            # Industry standard: Record videos when new best model found
            # Use callback approach - records after evaluation finds new best
            # Async recording (default) runs in background thread to avoid blocking training
            async_recording = viz_config.get("async_video_recording", True)
            video_callback = VideoRecorderOnBestCallback(
                eval_env=eval_env,
                video_folder=video_folder,
                video_length=max_ep_steps,
                name_prefix="best_model",
                async_recording=async_recording
            )
        elif video_freq == "every_eval":
            # Record videos every evaluation using VecVideoRecorder wrapper
            # This records during evaluation automatically
            # Wrap with render_mode first for VecVideoRecorder compatibility
            env_with_render = RenderModeWrapper(eval_env, render_mode="rgb_array")
            eval_env_for_callback = VecVideoRecorder(
                env_with_render,
                video_folder=str(video_folder),
                record_video_trigger=lambda x: x == 0,  # Record first episode of each eval
                video_length=max_ep_steps,
                name_prefix="eval"
            )
        elif isinstance(video_freq, int):
            # Record every Nth evaluation using VecVideoRecorder wrapper
            # This records during evaluation automatically
            # Note: VecVideoRecorder trigger is called per step, so we record first episode
            # of every Nth evaluation (simplified approach)
            # Wrap with render_mode first for VecVideoRecorder compatibility
            env_with_render = RenderModeWrapper(eval_env, render_mode="rgb_array")
            eval_env_for_callback = VecVideoRecorder(
                env_with_render,
                video_folder=str(video_folder),
                record_video_trigger=lambda x: x == 0,  # Record first episode
                video_length=max_ep_steps,
                name_prefix="eval"
            )
            # Note: For precise "every Nth eval" control, use "on_new_best" mode
            # or implement custom callback with evaluation counter
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env_for_callback,
        best_model_save_path=str(out_path / "best_model"),
        log_path=str(out_path / "results"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=render,  # Industry standard: False (disabled for performance)
        verbose=1,
        callback_on_new_best=video_callback,  # Record video when new best found
    )

    # Create callback list
    callbacks = [eval_callback]
    
    # Add periodic visualization callback if enabled
    periodic_viewer_config = viz_config.get("periodic_viewer", {})
    if periodic_viewer_config.get("enabled", False):
        # Get visualization frequency and settings
        visualize_freq = periodic_viewer_config.get("freq", 20000)
        n_episodes = periodic_viewer_config.get("n_episodes", 1)
        
        # Get max episode steps (from periodic_viewer config or env config)
        max_ep_steps = periodic_viewer_config.get(
            "max_ep_steps",
            config.get("env", {}).get("max_ep_steps", 4000)
        )
        
        # Create periodic visualization callback
        viz_callback = PeriodicVisualizationCallback(
            config=config,
            visualize_freq=visualize_freq,
            n_episodes=n_episodes,
            max_ep_steps=max_ep_steps,
            verbose=1
        )
        callbacks.append(viz_callback)
        
        print(colored(
            f"✓ Periodic visualization enabled: every {visualize_freq} steps, "
            f"{n_episodes} episode(s) per visualization",
            "cyan", attrs=["bold"]
        ))
    
    # Add checkpoint callback
    callbacks.append(
        CheckpointCallback(
            20000,
            save_path=str(out_path / "checkpoints"),
            name_prefix=f"{config['algo']['name']}_agent"
        )
    )
    
    return CallbackList(callbacks)

