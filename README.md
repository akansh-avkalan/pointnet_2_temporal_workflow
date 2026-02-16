# Project 
Temporal-Based Training Pipeline for PointNet++ Autoencoder

## Project Overview: 
Project Name:
PointNet++ Autoencoder Training Pipeline with Temporal Orchestration

Objective:
To build a fault-tolerant, resumable ML training pipeline using Temporal workflows for training a PointNet++ Autoencoder on 3D point cloud data.



```
Execution Flow:

run_workflow.py
      ↓
MLPipelineWorkflow
      ↓
1. preprocess_activity
      ↓
2. train_activity (25+ hours, resumable)
      ↓
3. evaluate_activity
      ↓
Return trained model path
```



```
Folder Structure: 

pointnet_temporal/
│
├── worker.py                         # Starts Temporal worker (registers workflows + activities)
├── run_workflow.py                   # Client entry point to start ML pipeline workflow
├── shared.py                         # Shared dataclasses (DatasetConfig, TrainConfig, EvalConfig)
├── configs/
│   ├── dataset_config.py
│   ├── train_config.py
│   └── eval_config.py
│
│
├── requirements.txt
│
├── data/
│   └── FieldGrow_ZeaMays_RawPCD_10k/  # Raw dataset (1045 .ply files)
│       ├── 0001.ply
│       ├── 0002.ply
│       └── ...
│
├── workflow/
│   ├── __init__.py
│   └── ml_pipeline_workflow.py       # Orchestrates:
│                                      #   1. Preprocess
│                                      #   2. Train
│                                      #   3. Evaluate
│                                      #   + Failure handling & retries
│
├── activities/
│   ├── __init__.py
│   ├── preprocess_activity.py        # Converts .ply → processed .npz
│   ├── train_activity.py             # Model training logic
│   └── evaluate_activity.py          # Model evaluation/inference
│
├── models/
│   ├── __init__.py
│   └── pointnet2.py                  # PointNet++ architecture implementation
│
├── utils/
│   ├── preprocessing.py              # FPS, normalization, 2048 downsampling
│   ├── dataset.py                    # Dataset class + train/val split + DataLoader
│   ├── trainer.py                    # Training loop + checkpoint logic
│   ├── metrics.py                    # Chamfer Loss / evaluation metrics
│   ├── visualization.py              # Plots & comparison charts
│   └── checkpoint.py                 # Resume utilities (optional but recommended)
│
├── outputs/
│   ├── processed/                    # Preprocessed .npz dataset
│   ├── checkpoints/                  # Model checkpoints
│   ├── logs/                         # Training logs
│   └── evaluation/                   # Inference outputs
│
└── README.md

```