#!/usr/bin/env python
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger as PLWandbLogger

from brain_tumor_segmentation.data.dataset import download_data, get_datasets
from brain_tumor_segmentation.data.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from brain_tumor_segmentation.training.trainer import Trainer
from brain_tumor_segmentation.utils.logger import WandBLogger
from brain_tumor_segmentation.utils.visualization import log_data_samples

# Use absolute path for config to avoid Windows path issues
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    # Debug: Print full configuration
    print("Full configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Warn if 'training' key is missing
    if "training" not in cfg:
        print(
            "Warning: 'training' key missing in configuration. Using default wandb settings."
        )
        wandb_config = {}
    else:
        wandb_config = cfg.training.get("wandb", {})

    # Set random seed
    torch.manual_seed(cfg.seed)

    # Create directories
    os.makedirs(cfg.dataset_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Download data using DVC
    download_data(cfg)

    # Get transforms
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # Get datasets
    train_dataset, val_dataset = get_datasets(cfg, train_transform, val_transform)

    # Initialize logger with default values
    project = wandb_config.get("project", "mipt-brain-tumor-segmentation")
    entity = wandb_config.get("entity", None)
    wandb_logger = PLWandbLogger(project=project, entity=entity, config=dict(cfg))
    logger = WandBLogger(cfg)

    # Log sample data
    log_data_samples(train_dataset, val_dataset, cfg)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # Initialize model
    model = Trainer(cfg)

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename="model-{epoch:02d}-{validation/mean_dice:.4f}",
        monitor="validation/mean_dice",
        mode="max",
        save_top_k=1,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_train_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=cfg.validation_intervals,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Log final model as MLflow artifact
    logger.log_model_artifact(
        str(Path(cfg.checkpoint_dir) / "model.pth"), "final_model"
    )

    # Finish W&B run
    wandb_logger.finalize("success")
    logger.finish()


if __name__ == "__main__":
    main()
