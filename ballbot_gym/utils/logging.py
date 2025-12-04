"""Logging utilities for ballbot environment."""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def save_episode_logs(
    log_dir: str,
    rgbd_hist_0: List,
    rgbd_hist_1: List,
    reward_term_1_hist: List,
    reward_term_2_hist: List,
    terrain_seed: Optional[int],
    terrain_type: str,
    depth_only: bool,
    log_options: Dict,
    num_episodes: int,
    eval_env: bool
):
    """
    Save episode logs to disk (cameras, rewards, terrain seeds).
    
    This function is called at the end of each episode (when terminated=True).
    It saves:
    - Camera frames (RGB and/or depth) if logging is enabled
    - Reward component histories for analysis
    - Terrain seeds for reproducibility
    
    Args:
        log_dir (str): Directory path where logs should be saved.
        rgbd_hist_0 (List): History of RGB-D images from camera 0.
        rgbd_hist_1 (List): History of RGB-D images from camera 1.
        reward_term_1_hist (List): History of directional reward component.
        reward_term_2_hist (List): History of action regularization reward component.
        terrain_seed (Optional[int]): Random seed used for terrain generation.
        terrain_type (str): Type of terrain ("perlin" or "flat").
        depth_only (bool): If True, only depth images were captured (no RGB).
        log_options (Dict): Logging configuration with keys "cams" and "reward_terms".
        num_episodes (int): Current episode number.
        eval_env (bool): If True, skip logging (evaluation mode).
    
    Note:
        Logs are only saved during training (not evaluation).
        Files are saved to log_dir, which should be created before calling this function.
    """
    # Skip logging in evaluation mode
    if eval_env:
        return

    # Save camera frames if enabled
    if log_options.get("cams", False) and len(rgbd_hist_0) > 0:
        # Create episode-specific directories
        dir_name = f"{log_dir}/rgbd_log_episode_{num_episodes}"
        dir_name_rgb = dir_name + "/rgb/"
        dir_name_d = dir_name + "/depth/"
        os.mkdir(dir_name)
        os.mkdir(dir_name_d)
        os.mkdir(dir_name_rgb)

        # Save each frame in the history
        for ii in range(len(rgbd_hist_0)):
            if not depth_only:
                # Save RGB images (convert from BGR to RGB for OpenCV)
                # OpenCV uses BGR format, so we reverse the channels
                rgb_0 = cv2.merge(
                    cv2.split(rgbd_hist_0[ii][:, :, :3])[::-1]
                ) * 255
                rgb_1 = cv2.merge(
                    cv2.split(rgbd_hist_1[ii][:, :, :3])[::-1]
                ) * 255
                cv2.imwrite(
                    f"{dir_name_rgb}/rbgd_a_{ii}.png",
                    rgb_0.astype("uint8")
                )
                cv2.imwrite(
                    f"{dir_name_rgb}/rbgd_b_{ii}.png",
                    rgb_1.astype("uint8")
                )
                
                # Save depth images (4th channel)
                cv2.imwrite(
                    f"{dir_name_d}/depth_a_{ii}.png",
                    (rgbd_hist_0[ii][:, :, 3] * 255).astype("uint8")
                )
                cv2.imwrite(
                    f"{dir_name_d}/depth_b_{ii}.png",
                    (rgbd_hist_1[ii][:, :, 3] * 255).astype("uint8")
                )
            else:
                # Depth-only mode: save depth images directly
                cv2.imwrite(
                    f"{dir_name_d}/depth_a_{ii}.png",
                    (rgbd_hist_0[ii] * 255).astype("uint8")
                )
                cv2.imwrite(
                    f"{dir_name_d}/depth_b_{ii}.png",
                    (rgbd_hist_1[ii] * 255).astype("uint8")
                )

    # Save reward component histories for analysis
    if log_options.get("reward_terms", False):
        if len(reward_term_1_hist):
            np.save(
                log_dir + "/term_1",
                np.array(reward_term_1_hist)
            )
            np.save(
                log_dir + "/term_2",
                np.array(reward_term_2_hist)
            )

    # Append terrain seed to history file (for reproducibility)
    if terrain_type == "perlin" and terrain_seed is not None:
        with open(f"{log_dir}/terrain_seed_history", "a") as fl:
            fl.write(f"{terrain_seed}\n")

