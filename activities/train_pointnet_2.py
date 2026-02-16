# activities/train_pointnet_2.py

# Input : Path of preprocessed point cloud
# Output : Path of trained model
# Processing: 
#       1. Dataset Creation -> Split if wanted -> Dataset loader
#       2. Model Architecture -> model/pointnet_2.py
#       3. Optimizer, loss, epoch, lr 
#       4. Checkpoint, resume, logging. 

from temporalio import activity
from configs.train_config import TrainConfig
from configs.dataset_config import DatasetConfig


@activity.defn
async def train_activity(dataset_config: DatasetConfig, train_config: TrainConfig) -> str:
    """
    Train PointNet++ autoencoder model.
    """
    
    import torch
    from model.pointnet_2 import PointNetPPAutoEncoder
    from utils.dataset import create_dataloaders
    from utils.trainer import Trainer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, n_train, n_val = create_dataloaders(
        dataset_config.processed_data_path,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        train_split=train_config.train_split
    )
    
    model = PointNetPPAutoEncoder(
        latent_dim=train_config.latent_dim,
        num_points=train_config.num_points
    )
    
    trainer = Trainer(model, train_loader, val_loader, train_config, device)
    
    resumed = trainer.load_checkpoint()
    if resumed:
        activity.heartbeat("Resumed from checkpoint")
    
    for epoch in range(trainer.start_epoch, train_config.num_epochs):
        train_loss = trainer.train_epoch(epoch)
        
        if (epoch + 1) % train_config.val_interval == 0:
            val_loss = trainer.validate()
            trainer.log(f"Epoch [{epoch+1}/{train_config.num_epochs}] | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        else:
            trainer.log(f"Epoch [{epoch+1}/{train_config.num_epochs}] | Train: {train_loss:.6f}")
        
        if (epoch + 1) % train_config.checkpoint_interval == 0:
            trainer.save_checkpoint(epoch)
            activity.heartbeat(f"Epoch {epoch+1}/{train_config.num_epochs}")
    
    final_path = trainer.train()
    
    return final_path