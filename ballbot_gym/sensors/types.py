"""Type definitions for sensor data."""
from dataclasses import dataclass
import numpy as np


@dataclass
class StampedImPair:
    """
    Holds a pair of images (e.g. consecutive frames) alongside a time stamp.
    
    Attributes:
        im_0 (np.ndarray): First image, typically at time t.
        im_1 (np.ndarray): Second image, typically at time t+dt.
        ts (float): Time stamp (seconds or simulation time).
    """
    im_0: np.ndarray
    im_1: np.ndarray
    ts: float

