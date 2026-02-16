# workflow_run.py

import asyncio
from temporalio.client import Client

from shared import TASK_QUEUE, MLPipelineInput
from configs.dataset_config import DatasetConfig
from configs.train_config import TrainConfig
from configs.eval_config import EvalConfig


async def main():
    client = await Client.connect("localhost:7233")
    
    dataset_config = DatasetConfig()
    train_config = TrainConfig()
    eval_config = EvalConfig()  # Set to None to skip evaluation
    
    pipeline_input = MLPipelineInput(
        dataset_config=dataset_config,
        train_config=train_config,
        eval_config=eval_config,
    )
    
    handle = await client.start_workflow(
        "MLPipelineWorkflow",
        pipeline_input,
        id="ml-pipeline-run-001",
        task_queue=TASK_QUEUE,
    )
    
    print(f"Workflow started: {handle.id}")
    
    result = await handle.result()
    
    if eval_config:
        print(f"Workflow completed. Evaluation results saved to: {result}")
    else:
        print(f"Workflow completed. Model saved to: {result}")


if __name__ == "__main__":
    asyncio.run(main())