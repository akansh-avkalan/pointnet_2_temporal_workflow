# configs/train_config.py

from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 200
    learning_rate: float = 1e-3
    num_workers: int = 4
    
    latent_dim: int = 128
    num_points: int = 2048
    
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_interval: int = 10
    snapshot_interval: int = 50
    val_interval: int = 10
    
    train_split: float = 0.8