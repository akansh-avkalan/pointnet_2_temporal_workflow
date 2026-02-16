# configs/__init__.py

"""
Configuration module for the PointNet++ Temporal pipeline.

Contains configuration dataclasses for:
- Dataset paths and preprocessing parameters
- Training hyperparameters (to be added)
- Evaluation settings (to be added)
"""

from configs.dataset_config import DatasetConfig

__all__ = ["DatasetConfig"]