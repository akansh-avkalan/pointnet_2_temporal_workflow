from dataclasses import dataclass

"""
Dataset configuration for PointNet++ Temporal pipeline.

This config controls:
- Raw dataset location
- Processed dataset location
- Preprocessing behavior
"""


@dataclass
class DatasetConfig:
    raw_data_path: str = "test_field_maize_data" #"FielGrwon_ZeaMays_RawPCD_10k"
    processed_data_path: str = "outputs/processed"
    npoints: int = 2048
    normalize: bool = True

