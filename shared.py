from dataclasses import dataclass
from typing import Optional

from configs.dataset_config import DatasetConfig
from configs.train_config import TrainConfig
from configs.eval_config import EvalConfig


TASK_QUEUE = "pointnet_pipeline_queue"


@dataclass
class MLPipelineInput:
    dataset_config: DatasetConfig
    train_config: TrainConfig
    eval_config: Optional[EvalConfig] = None
