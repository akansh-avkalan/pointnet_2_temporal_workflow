# utils/trainer.py

import os
import torch
from datetime import datetime
from utils.metrics import chamfer_distance


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.start_epoch = 0
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.checkpoint_dir, "train_log.txt")
        self.log_f = open(self.log_file, "a")
    
    def load_checkpoint(self):
        """Load checkpoint if exists."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epoch = checkpoint["epoch"] + 1
            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            
            return True
        return False
    
    def save_checkpoint(self, epoch):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, checkpoint_path)
        
        if (epoch + 1) % self.config.snapshot_interval == 0:
            snapshot_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(checkpoint, snapshot_path)
    
    def log(self, msg):
        """Log message to file and console."""
        print(msg)
        self.log_f.write(msg + "\n")
        self.log_f.flush()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, points in enumerate(self.train_loader):
            points = points.to(self.device)
            
            self.optimizer.zero_grad()
            
            recon, latent = self.model(points)
            loss = chamfer_distance(recon, points)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(self.train_loader)
        return epoch_loss
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for points in self.val_loader:
                points = points.to(self.device)
                recon, latent = self.model(points)
                loss = chamfer_distance(recon, points)
                val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        return val_loss
    
    def train(self):
        """Main training loop."""
        self.log(f"Training started: {datetime.now()}")
        self.log(f"Starting from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss = self.train_epoch(epoch)
            
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate()
                self.log(f"Epoch [{epoch+1}/{self.config.num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            else:
                self.log(f"Epoch [{epoch+1}/{self.config.num_epochs}] | Train Loss: {train_loss:.6f}")
            
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
        
        final_path = os.path.join(self.checkpoint_dir, "pointnetpp_ae_final.pt")
        torch.save(self.model.state_dict(), final_path)
        
        self.log(f"Training complete: {datetime.now()}")
        self.log_f.close()
        
        return final_path