# configs/dataset_config.py

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    raw_data_path: str = "test_field_maize_data"
    processed_data_path: str = "outputs/processed"
    npoints: int = 2048
    normalize: bool = True