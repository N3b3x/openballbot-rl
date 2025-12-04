"""Dataset classes for ballbot RL data."""
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class DepthImageDataset(Dataset):

    def __init__(self, image_data_dict):
        """
        image_data_dict expects the structure documented in collect_depth_image_paths and load_depth_images
        """
        self.samples = []
        for log_id, episodes in image_data_dict.items():
            for ep_id, images in episodes.items():
                for img in images:
                    self.samples.append((log_id, ep_id, img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, _, img = self.samples[idx]
        img = torch.from_numpy(img).float().reshape(1, img.shape[0],
                                                    img.shape[1])
        return img / 255.0  # we normalize here to avoid a large pickle file

    def plot_im(self, idx):
        _, _, im = self.samples[idx]
        plt.imshow(im)
        plt.show()

    def merge(self, other_ds):
        """
        merges two datasets
        """
        self.samples = self.samples + other_ds.samples
        return self

