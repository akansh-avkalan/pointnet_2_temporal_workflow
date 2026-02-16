# workflow/run_ml_workflow.py
# Import - Shared (config), activities, Interact with files inside activity. 
# Exection flow -> Preprocess dataset -> Train the model -> Evaluate the model

# workflows/run_ml_workflow.py

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

from shared import MLPipelineInput
from activities.preprocess_data import preprocess_data


@workflow.defn
class MLPipelineWorkflow:

    @workflow.run
    async def run(self, pipeline_input: MLPipelineInput) -> str:
        """
        Orchestrates the ML pipeline.

        Current step:
        1. Preprocess dataset

        Returns:
            str: Path to processed dataset
        """

        processed_path = await workflow.execute_activity(
            preprocess_data,
            pipeline_input.dataset_config,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(
                maximum_attempts=3
            ),
        )

        return processed_path
