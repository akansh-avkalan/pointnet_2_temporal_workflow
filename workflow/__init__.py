# workflow/__init__.py

"""
Workflow module for Temporal orchestration.

Contains the main ML pipeline workflow that orchestrates:
- Preprocessing
- Training
- Evaluation
"""

from workflow.run_ml_workflow import MLPipelineWorkflow

__all__ = ["MLPipelineWorkflow"]