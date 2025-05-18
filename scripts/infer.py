# #!/usr/bin/env python
# import os
# import hydra
# from omegaconf import DictConfig
# import torch
# import wandb
# from brain_tumor_segmentation.training.metrics import get_postprocessing_transforms
# from monai.data import DataLoader
# from monai.transforms import Compose, Activations, AsDiscrete
# from brain_tumor_segmentation.utils.logger import WandBLogger
# from brain_tumor_segmentation.data.dataset import get_val_dataset
# from brain_tumor_segmentation.data.transforms import get_val_transforms
# from brain_tumor_segmentation.models.segresnet import get_model
# from brain_tumor_segmentation.utils.visualization import log_predictions
# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def main(cfg: DictConfig):
#     # Initialize W&B
#     logger = WandBLogger(cfg)  # Add this line
    
#     # Set random seed
#     torch.manual_seed(cfg.seed)
    
#     # Get transform
#     val_transform = get_val_transforms(cfg)
    
#     # Get validation dataset
#     val_dataset = get_val_dataset(cfg, val_transform)
    
#     # Get device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     # Get model
#     model = get_model(cfg).to(device)
    
#     # Load checkpoint
#     checkpoint_path = os.path.join(cfg.checkpoint_dir, "model.pth")
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()
    
#     # Post-processing transforms
#     post_trans = get_postprocessing_transforms()
    
#     # Perform inference and log predictions - Add this block
#     log_predictions(model, val_dataset, post_trans, cfg, device)
    
#     # Finish W&B run - Add this line
#     logger.finish()




#!/usr/bin/env python
import os
import hydra
import torch
import wandb
import numpy as np
from omegaconf import DictConfig
from monai.data import DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from brain_tumor_segmentation.training.metrics import get_postprocessing_transforms
from brain_tumor_segmentation.utils.logger import WandBLogger
from brain_tumor_segmentation.data.dataset import get_val_dataset
from brain_tumor_segmentation.data.transforms import get_val_transforms
from brain_tumor_segmentation.models.segresnet import get_model
from brain_tumor_segmentation.utils.visualization import log_predictions

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
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
    
    # Load checkpoint
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "model.pth")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # Post-processing transforms
    post_trans = get_postprocessing_transforms()
    
    # Initialize metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    
    # Perform inference with metrics
    with torch.no_grad():
        for val_data in val_loader:
            inputs = val_data["image"].to(device)
            labels = val_data["label"].to(device)
            
            # Inference
            outputs = model(inputs)
            preds = [post_trans(i) for i in decollate_batch(outputs)]
            
            # Calculate metrics
            dice_metric(y_pred=preds, y=labels)
            hd_metric(y_pred=preds, y=labels)
        
        # Get aggregated results
        dice_scores = dice_metric.aggregate().cpu().numpy()
        hd_scores = hd_metric.aggregate().cpu().numpy()
        
        # Print metrics
        print("\n=== Validation Metrics ===")
        print(f"Whole Tumor Dice: {dice_scores[0]:.4f}")
        print(f"Tumor Core Dice: {dice_scores[1]:.4f}")
        print(f"Enhancing Tumor Dice: {dice_scores[2]:.4f}")
        print(f"HD95 Scores: {hd_scores}")
        
        # Log metrics to W&B
        wandb.log({
            "metrics/wt_dice": dice_scores[0],
            "metrics/tc_dice": dice_scores[1],
            "metrics/et_dice": dice_scores[2],
            "metrics/mean_hd95": np.mean(hd_scores)
        })
    
    # Keep your existing visualization
    log_predictions(model, val_dataset, post_trans, cfg, device)
    
    # Finish W&B run
    logger.finish()

if __name__ == "__main__":
    main()















# #!/usr/bin/env python
# import os
# import csv
# import hydra
# import numpy as np
# from omegaconf import DictConfig
# import torch
# import wandb
# from pathlib import Path
# from monai.data import DataLoader
# from monai.transforms import Compose
# from brain_tumor_segmentation.training.metrics import get_postprocessing_transforms
# from brain_tumor_segmentation.utils.logger import WandBLogger
# from brain_tumor_segmentation.data.dataset import get_val_dataset
# from brain_tumor_segmentation.data.transforms import get_val_transforms
# from brain_tumor_segmentation.models.segresnet import get_model
# from brain_tumor_segmentation.utils.visualization import log_predictions

# def save_predictions_to_csv(predictions: np.ndarray, output_path: str):
#     """Save predictions to CSV file.
    
#     Args:
#         predictions: NumPy array of shape (num_samples, num_classes, ...)
#         output_path: Path to save CSV
#     """
#     # Flatten predictions to 2D: (num_samples, num_features)
#     flat_preds = predictions.reshape(predictions.shape[0], -1)
    
#     with open(output_path, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['sample_id'] + [f'feature_{i}' for i in range(flat_preds.shape[1])])
#         for idx, pred in enumerate(flat_preds):
#             writer.writerow([idx] + list(pred))

# @hydra.main(version_base=None, config_path="../configs", config_name="config")
# def main(cfg: DictConfig):
#     # Initialize W&B
#     logger = WandBLogger(cfg)
#     torch.manual_seed(cfg.seed)
    
#     # Setup paths
#     output_dir = Path(cfg.get("output_dir", "outputs"))
#     output_dir.mkdir(exist_ok=True)
#     csv_path = output_dir / "predictions.csv"
    
#     # Get data and model
#     val_dataset = get_val_dataset(cfg, get_val_transforms(cfg))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_model(cfg).to(device)
    
#     # Load model
#     checkpoint_path = Path(cfg.checkpoint_dir) / "model.pth"
#     model.load_state_dict(torch.load(checkpoint_path))
#     model.eval()
    
#     # Inference
#     post_trans = get_postprocessing_transforms()
#     all_preds = []
    
#     with torch.no_grad():
#         for sample in val_dataset:
#             inputs = sample["image"].unsqueeze(0).to(device)
#             outputs = model(inputs)
#             preds = post_trans(outputs[0]).cpu().numpy()
#             all_preds.append(preds)
    
#     # Convert to numpy array (num_samples, num_classes, ...)
#     all_preds = np.stack(all_preds)
    
#     # Save to CSV
#     save_predictions_to_csv(all_preds, csv_path)
#     print(f"Saved predictions to {csv_path}")
    
#     # Optional: Log visualizations
#     log_predictions(model, val_dataset, post_trans, cfg, device)
#     logger.finish()

# if __name__ == "__main__":
#     main()