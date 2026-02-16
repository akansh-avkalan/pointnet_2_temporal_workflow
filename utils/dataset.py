# utils/dataset.py

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PointCloudDataset(Dataset):
    def __init__(self, file_paths):
        self.files = file_paths
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        points = data['points']
        return torch.from_numpy(points).float()


def create_dataloaders(processed_data_path, batch_size=8, num_workers=4, train_split=0.8):
    """Create train and validation dataloaders."""
    
    all_files = glob.glob(os.path.join(processed_data_path, '*.npz'))
    all_files.sort()
    
    if len(all_files) == 0:
        raise ValueError(f"No .npz files found in {processed_data_path}")
    
    random.seed(42)
    random.shuffle(all_files)
    
    split = int(train_split * len(all_files))
    train_paths = all_files[:split]
    val_paths = all_files[split:]
    
    train_dataset = PointCloudDataset(train_paths)
    val_dataset = PointCloudDataset(val_paths)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, len(train_paths), len(val_paths)