# worker.py

import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from shared import TASK_QUEUE
from workflow.run_ml_workflow import MLPipelineWorkflow
from activities.preprocess_data import preprocess_activity
from activities.train_pointnet_2 import train_activity
from activities.evaluate_pointnet_2 import evaluate_activity


async def main():
    client = await Client.connect("localhost:7233")
    
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MLPipelineWorkflow],
        activities=[preprocess_activity, train_activity, evaluate_activity],
    )
    
    print(f"Worker started. Listening on task queue: {TASK_QUEUE}")
    
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())