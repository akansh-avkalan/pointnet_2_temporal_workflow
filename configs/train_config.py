from dataclasses import dataclass

@dataclass
class TrainConfig:
    processed_data_path: str = "outputs/processed"
    checkpoint_dir: str = "outputs/checkpoints"
    log_path: str = "outputs/logs/train_log.txt"

    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4

    resume: bool = True
    device: str = "cuda"
