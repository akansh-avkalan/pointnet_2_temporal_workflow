# Worker : Define the task queue it will pick task and workflow name.


# worker.py

import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from shared import TASK_QUEUE
from workflow.run_ml_workflow import MLPipelineWorkflow
from activities.preprocess_data import preprocess_data


async def main():
    """
    Starts Temporal worker.
    """

    # Connect to Temporal server
    client = await Client.connect("localhost:7233")

    # Create worker
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MLPipelineWorkflow],
        activities=[preprocess_data],
    )

    print("Worker started. Listening on task queue:", TASK_QUEUE)

    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
