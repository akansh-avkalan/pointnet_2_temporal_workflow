# Start the workflow/Initialize the workflow with values. 

# run_workflow.py

import asyncio
from temporalio.client import Client

from shared import TASK_QUEUE
from shared import MLPipelineInput
from configs.dataset_config import DatasetConfig
from configs.train_config import TrainConfig
from configs.eval_config import EvalConfig


async def main():
    """
    Starts ML pipeline workflow.
    """

    client = await Client.connect("localhost:7233")

    # --------------------------------------------------
    # Dataset Config (defaults used if not overridden)
    # --------------------------------------------------

    dataset_config = DatasetConfig()

    # --------------------------------------------------
    # Train Config (placeholder for now)
    # --------------------------------------------------

    # train_config = TrainConfig()

    # --------------------------------------------------
    # Eval Config (optional)
    # --------------------------------------------------

    # eval_config = None

    pipeline_input = MLPipelineInput(
        dataset_config=dataset_config,
       # train_config=train_config,
       # evaluate_config=eval_config,
    )

    handle = await client.start_workflow(
        "MLPipelineWorkflow",
        pipeline_input,
        id="ml-pipeline-run-001",   # Change for new run
        task_queue=TASK_QUEUE,
    )

    print("Workflow started.")
    print("Workflow ID:", handle.id)
    print("Run ID:", handle.result_run_id)

    result = await handle.result()

    print("Workflow completed.")
    print("Processed dataset path:", result)


if __name__ == "__main__":
    asyncio.run(main())
