# shared.py

from dataclasses import dataclass
from typing import Optional

from configs.dataset_config import DatasetConfig


TASK_QUEUE = "pointnet_pipeline_queue"


@dataclass
class MLPipelineInput:
    dataset_config: DatasetConfig