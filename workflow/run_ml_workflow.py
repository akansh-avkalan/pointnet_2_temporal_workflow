# workflow/ml_pipeline_workflow.py

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from shared import MLPipelineInput
    from activities.preprocess_data import preprocess_activity
    from activities.train_pointnet_2 import train_activity
    from activities.evaluate_pointnet_2 import evaluate_activity


@workflow.defn
class MLPipelineWorkflow:
    
    @workflow.run
    async def run(self, input: MLPipelineInput) -> str:
        
        # Step 1: Preprocessing
        preprocessing_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=5),
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
            backoff_coefficient=2.0,
        )
        
        processed_data_path = await workflow.execute_activity(
            preprocess_activity,
            input.dataset_config,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=preprocessing_retry_policy,
            heartbeat_timeout=timedelta(minutes=5),
        )
        
        # Step 2: Training
        training_retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=10),
            maximum_interval=timedelta(minutes=5),
            maximum_attempts=5,
            backoff_coefficient=2.0,
        )
        
        model_path = await workflow.execute_activity(
            train_activity,
            args=[input.dataset_config, input.train_config],
            start_to_close_timeout=timedelta(hours=48),
            retry_policy=training_retry_policy,
            heartbeat_timeout=timedelta(minutes=30),
        )
        
        # Step 3: Evaluation (optional)
        if input.eval_config is not None:
            eval_retry_policy = RetryPolicy(
                initial_interval=timedelta(seconds=5),
                maximum_interval=timedelta(minutes=2),
                maximum_attempts=3,
                backoff_coefficient=2.0,
            )
            
            eval_results_path = await workflow.execute_activity(
                evaluate_activity,
                args=[input.dataset_config, input.train_config, input.eval_config, model_path],
                start_to_close_timeout=timedelta(hours=3),
                retry_policy=eval_retry_policy,
                heartbeat_timeout=timedelta(minutes=5),
            )
            
            return eval_results_path
        
        return model_path