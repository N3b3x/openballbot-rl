"""Data loading utilities for ballbot RL."""
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List


def collect_depth_image_paths(root_dir):
    """
    Collects paths to all depth images under the given root directory.

    Returns a dict:
    {
        'log_<id>': {
            episode_idx: [list of depth image paths]
        },
        ...
    }
    """
    root = Path(root_dir)
    data = {}

    for log_dir in root.glob('log_*'):
        log_id = log_dir.name
        data[log_id] = {}

        for ep_dir in log_dir.glob('rgbd_log_episode_*'):
            try:
                episode_idx = int(ep_dir.name.split('_')[-1])
            except ValueError:
                continue  # Skip malformed dirs

            depth_dir = ep_dir / 'depth'
            if not depth_dir.is_dir():
                continue

            image_paths = sorted(
                depth_dir.glob('*.*'))  # Adjust extension filter if needed
            data[log_id][episode_idx] = list(image_paths)

    return data


def load_depth_images(data_dict):
    """
    Converts image paths to actual image arrays (numpy).

    Returns a structure with same shape as `data_dict`, but image paths replaced by arrays.
    """
    return {
        log_id: {
            ep_idx: [np.array(Image.open(p)) for p in image_paths]
            for ep_idx, image_paths in episodes.items()
        }
        for log_id, episodes in data_dict.items()
    }

