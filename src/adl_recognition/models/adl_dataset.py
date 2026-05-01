import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class ADLDataset(Dataset):
    """
    Dataset class for Activity of Daily Living (ADL) skeleton data.
    Supports loading from JSON/NPY files containing keypoints.
    """
    def __init__(self, data_root, split='train', window_size=30, step=1, transform=None):
        self.data_root = data_root
        self.split = split
        self.window_size = window_size
        self.step = step
        self.transform = transform
        
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        # Placeholder for loading logic
        # Expecting a directory structure like:
        # data_root/train/labels.json
        # data_root/train/skeletons/*.npy
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        skeleton = self.data[idx] # (C, T, V, M) format common in ST-GCN
        label = self.labels[idx]
        
        if self.transform:
            skeleton = self.transform(skeleton)
            
        return torch.from_numpy(skeleton).float(), torch.tensor(label).long()

class SkeletonAugmentation:
    """Utilities for augmenting skeleton data during research/training."""
    @staticmethod
    def random_move(skeleton, max_dist=0.1):
        # (C, T, V, M)
        C, T, V, M = skeleton.shape
        move = np.random.uniform(-max_dist, max_dist, (C, 1, 1, M))
        return skeleton + move

    @staticmethod
    def random_rotate(skeleton, max_angle=10):
        # Simple 2D rotation for research
        pass
