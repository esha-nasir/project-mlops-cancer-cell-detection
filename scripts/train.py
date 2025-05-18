#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Add project root to Python path

import hydra
from omegaconf import DictConfig,OmegaConf
import torch

from monai.data import DataLoader
from monai.transforms import Compose, Activations, AsDiscrete

# Updated imports to match new structure
from brain_tumor_segmentation.data.dataset import get_datasets
from brain_tumor_segmentation.data.transforms import (
    get_train_transforms, 
    get_val_transforms
)
from brain_tumor_segmentation.training.metrics import get_postprocessing_transforms
from brain_tumor_segmentation.models.segresnet import get_model
from brain_tumor_segmentation.training.trainer import Trainer
from brain_tumor_segmentation.utils.visualization import log_data_samples
from brain_tumor_segmentation.utils.logger import WandBLogger



    
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Print full config for debugging
    print(OmegaConf.to_yaml(cfg))
    
    # Access config values properly
    print("Training epochs:", cfg.max_train_epochs)
    print("Batch size:", cfg.batch_size)
    
    # Initialize W&B with proper config access
    logger = WandBLogger(cfg)
    
    # Set random seed
    torch.manual_seed(cfg.seed)
    
    # Create directories
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # Get transforms - Updated to use new functions
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)
    post_trans = get_postprocessing_transforms()
    
    # Get datasets
    train_dataset, val_dataset = get_datasets(cfg, train_transform, val_transform)
    
    # Log sample data
    log_data_samples(train_dataset, val_dataset, cfg)
    print(">>>>>>>>>>>>>>>>>>",cfg.batch_size)
    # Training loop
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    print("Data Load-------------",train_loader)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get model
    model = get_model(cfg).to(device)
    
    # Create trainer
    trainer = Trainer(model, device, cfg)
    print(">>>>>>>>>>>>>>>>>>",cfg.max_train_epochs)
    # Training loop
    for epoch in range(cfg.max_train_epochs):
        # Train for one epoch
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Log metrics
        logger.log_metrics({
            "mean_train_loss": train_loss,
            "learning_rate": trainer.lr_scheduler.get_last_lr()[0]  # Fixed reference
        }, step_type="epoch")

        # Validate
        if (epoch + 1) % cfg.validation_intervals == 0:
            val_metric = trainer.validate(val_loader, post_trans, epoch)
            
            # Save checkpoint
            checkpoint_path = os.path.join(cfg.checkpoint_dir, "model.pth")
            trainer.save_checkpoint(checkpoint_path)
            logger.log_model_checkpoint(model, epoch)
    
    # Finish W&B run
    logger.finish()

if __name__ == "__main__":
    main()