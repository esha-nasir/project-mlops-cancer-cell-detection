#!/usr/bin/env python
import os

import hydra
import numpy as np
import torch
from monai.data import DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from omegaconf import DictConfig

from brain_tumor_segmentation.data.dataset import download_data, get_val_dataset
from brain_tumor_segmentation.data.transforms import get_val_transforms
from brain_tumor_segmentation.models.segresnet import get_model
from brain_tumor_segmentation.training.metrics import get_postprocessing_transforms
from brain_tumor_segmentation.utils.logger import WandBLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Download data using DVC
    download_data(cfg)

    # Initialize W&B
    logger = WandBLogger(cfg)
    torch.manual_seed(cfg.seed)

    # Get transform
    val_transform = get_val_transforms(cfg)

    # Get validation dataset
    val_dataset = get_val_dataset(cfg, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get model
    model = get_model(cfg).to(device)
    print("-------------Check after model--------", model)
    # Load checkpoint
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "model.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print("-------------After Checkpoints--------", model)

    # Post-processing transforms
    post_trans = get_postprocessing_transforms()
    print("Post Processing Transformation", post_trans)
    # Initialize metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    print("Dice Metric", dice_metric)
    print("HD Metric", hd_metric)
    # Perform inference with metrics
    with torch.no_grad():
        for val_data in val_loader:
            inputs = val_data["image"].to(device)
            labels = val_data["label"].to(device)

            # Inference
            outputs = model(inputs)
            preds = [post_trans(i) for i in decollate_batch(outputs)]

            labs = decollate_batch(labels)

            # Calculate metrics
            dice_metric(y_pred=preds, y=labs)
            hd_metric(y_pred=preds, y=labs)

        # Get aggregated results
        dice_scores = dice_metric.aggregate().cpu().numpy()
        hd_scores = hd_metric.aggregate().cpu().numpy()

        # Debug print to check dice_scores length and values
        print(f"\nDice scores array (length {len(dice_scores)}): {dice_scores}")

        # Print metrics safely
        print("\n=== Validation Metrics ===")
        print(f"Whole Tumor Dice: {dice_scores[0]: .4f}")
        print(f"Tumor Core Dice: {dice_scores[1]: .4f}")
        if len(dice_scores) > 2:
            print(f"Enhancing Tumor Dice: {dice_scores[2]: .4f}")
        else:
            print("Enhancing Tumor Dice: Not available")
        print(f"HD95 Scores: {hd_scores}")

        # Prepare metrics dict safely
        metrics = {
            "wt_dice": dice_scores[0],
            "tc_dice": dice_scores[1],
            "mean_hd95": np.mean(hd_scores),
        }
        if len(dice_scores) > 2:
            metrics["et_dice"] = dice_scores[2]

        # Log metrics to W&B and MLflow
        logger.log_metrics(metrics, step_type="validation")

    # Finish W&B run
    logger.finish()


if __name__ == "__main__":
    main()
