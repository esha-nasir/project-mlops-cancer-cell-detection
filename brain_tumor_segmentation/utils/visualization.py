import numpy as np
from tqdm.auto import tqdm
import wandb
from monai.data import decollate_batch
import torch
from itertools import islice

def log_data_samples(train_dataset, val_dataset, config):
    """Log sample data from training and validation sets to W&B."""
    table = wandb.Table(
        columns=[
            "Split",
            "Data Index",
            "Slice Index",
            "Image-Channel-0",
            "Image-Channel-1",
            "Image-Channel-2",
            "Image-Channel-3",
        ]
    )

    # Log training samples
    max_samples = (
        min(config.max_train_images_visualized, len(train_dataset))
        if config.max_train_images_visualized > 0
        else len(train_dataset)
    )
    progress_bar = tqdm(
        enumerate(islice(train_dataset, max_samples)),
        total=max_samples,
        desc="Generating Train Dataset Visualizations:",
    )
    for data_idx, sample in progress_bar:
        sample_image = sample["image"].detach().cpu().numpy()
        sample_label = sample["label"].detach().cpu().numpy()
        table = _add_data_sample_to_table(
            sample_image,
            sample_label,
            split="train",
            data_idx=data_idx,
            table=table,
        )

    # Log validation samples
    max_samples = (
        min(config.max_val_images_visualized, len(val_dataset))
        if config.max_val_images_visualized > 0
        else len(val_dataset)
    )
    progress_bar = tqdm(
        enumerate(islice(val_dataset, max_samples)),
        total=max_samples,
        desc="Generating Validation Dataset Visualizations:",
    )
    for data_idx, sample in progress_bar:
        sample_image = sample["image"].detach().cpu().numpy()
        sample_label = sample["label"].detach().cpu().numpy()
        
        table = _add_data_sample_to_table(
            sample_image,
            sample_label,
            split="val",
            data_idx=data_idx,
            table=table,
        )
        print("End")

    wandb.log({"Tumor-Segmentation-Data": table})


def _add_data_sample_to_table(
    sample_image: np.array,
    sample_label: np.array,
    split: str,
    data_idx: int,
    table: wandb.Table,
):
    """Helper function to add a single sample to W&B table."""
    # Check if we have a batch dimension (5D tensor)
    if len(sample_image.shape) == 5:
        # Remove batch dimension by selecting first item
        sample_image = sample_image[0]  # shape becomes (4, 224, 224, 144)
        sample_label = sample_label[0]  # shape becomes (3, 224, 224, 144)
    
    # Now unpack the expected 4D shape
    num_channels, height, width, num_slices = sample_image.shape

    with tqdm(total=num_slices, leave=False) as progress_bar:
        for slice_idx in range(num_slices):
            ground_truth_images = []
            for channel_idx in range(num_channels):
                masks = {
                    "ground-truth/Tumor-Core": {
                        "mask_data": sample_label[0, :, :, slice_idx],
                        "class_labels": {0: "background", 1: "Tumor Core"},
                    },
                    "ground-truth/Whole-Tumor": {
                        "mask_data": sample_label[1, :, :, slice_idx] * 2,
                        "class_labels": {0: "background", 2: "Whole Tumor"},
                    },
                    "ground-truth/Enhancing-Tumor": {
                        "mask_data": sample_label[2, :, :, slice_idx] * 3,
                        "class_labels": {0: "background", 3: "Enhancing Tumor"},
                    },
                }
                ground_truth_images.append(
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks=masks,
                    )
                )
                
            table.add_data(
                split,
                data_idx,
                slice_idx,
                *ground_truth_images
            )
            progress_bar.update(1)
    return table



# def _add_data_sample_to_table(
#     sample_image: np.array,
#     sample_label: np.array,
#     split: str,
#     data_idx: int,
#     table: wandb.Table,
# ):
    
#     """Helper function to add a single sample to W&B table."""
   
#     print(f"Sample image shape: {sample_image.shape}")  # Debug print
#     print(f"Sample label shape: {sample_label.shape}")  # Debug print
    
#     try:
#         num_channels, _, _, num_slices = sample_image.shape
#     except ValueError as e:
#         raise ValueError(f"Expected 4D tensor but got shape {sample_image.shape}. Original error: {str(e)}")

#     # Add a debug line before anything else
#     print(f"[DEBUG] sample_image.shape = {sample_image.shape}")

#     # Make sure you unpack correctly
#     if sample_image.ndim == 4:
#         num_channels, _, _, num_slices = sample_image.shape
#     elif sample_image.ndim == 3:
#         num_channels = 1
#         _, _, num_slices = sample_image.shape
#         sample_image = np.expand_dims(sample_image, 0)
#     else:
#         raise ValueError(f"Unexpected sample_image shape: {sample_image.shape}")
    
#     with tqdm(total=num_slices, leave=False) as progress_bar:
#         for slice_idx in range(num_slices):
#             ground_truth_images = []
#             for channel_idx in range(num_channels):
#                 masks = {
#                     "ground-truth/Tumor-Core": {
#                         "mask_data": sample_label[0, :, :, slice_idx],
#                         "class_labels": {0: "background", 1: "Tumor Core"},
#                     },
#                     "ground-truth/Whole-Tumor": {
#                         "mask_data": sample_label[1, :, :, slice_idx] * 2,
#                         "class_labels": {0: "background", 2: "Whole Tumor"},
#                     },
#                     "ground-truth/Enhancing-Tumor": {
#                         "mask_data": sample_label[2, :, :, slice_idx] * 3,
#                         "class_labels": {0: "background", 3: "Enhancing Tumor"},
#                     },
#                 }
#                 ground_truth_images.append(
#                     wandb.Image(
#                         sample_image[channel_idx, :, :, slice_idx],
#                         masks=masks,
#                     )
#                 )
                
#             table.add_data(
#                 split,
#                 data_idx,
#                 slice_idx,
#                 *ground_truth_images
#             )
#             progress_bar.update(1)
#     return table

def log_predictions(model, val_dataset, post_trans, config, device):
    """Log model predictions to W&B."""
    prediction_table = wandb.Table(
        columns=[
            "Split",
            "Data Index",
            "Slice Index",
            "Image-Channel-0/Tumor-Core",
            "Image-Channel-1/Tumor-Core",
            "Image-Channel-2/Tumor-Core",
            "Image-Channel-3/Tumor-Core",
            "Image-Channel-0/Whole-Tumor",
            "Image-Channel-1/Whole-Tumor",
            "Image-Channel-2/Whole-Tumor",
            "Image-Channel-3/Whole-Tumor",
            "Image-Channel-0/Enhancing-Tumor",
            "Image-Channel-1/Enhancing-Tumor",
            "Image-Channel-2/Enhancing-Tumor",
            "Image-Channel-3/Enhancing-Tumor",
        ]
    )

    max_samples = (
        min(config.max_prediction_images_visualized, len(val_dataset))
        if config.max_prediction_images_visualized > 0
        else len(val_dataset)
    )
  
    progress_bar = tqdm(
        enumerate(islice(val_dataset, max_samples)),
        total=max_samples,
        desc="Generating Predictions:",
    )
    
    with torch.no_grad():
        for data_idx, sample in progress_bar:
            val_input = sample["image"].unsqueeze(0).to(device)
            val_output = model(val_input)
            val_output = post_trans(val_output[0])
            
            prediction_table = _add_prediction_to_table(
                sample_image=sample["image"].cpu().numpy(),
                sample_label=sample["label"].cpu().numpy(),
                predicted_label=val_output.cpu().numpy(),
                data_idx=data_idx,
                table=prediction_table,
            )
    
    wandb.log({"Predictions/Tumor-Segmentation-Data": prediction_table})

def _add_prediction_to_table(
    sample_image: np.array,
    sample_label: np.array,
    predicted_label: np.array,
    data_idx: int,
    table: wandb.Table,
):
    """Helper function to add predictions to W&B table."""
    num_channels, _, _, num_slices = sample_image.shape
    
    with tqdm(total=num_slices, leave=False) as progress_bar:
        for slice_idx in range(num_slices):
            wandb_images = []
            for channel_idx in range(num_channels):
                # Tumor Core visualization
                wandb_images.append(
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Tumor-Core": {
                                "mask_data": sample_label[0, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Tumor Core"},
                            },
                            "prediction/Tumor-Core": {
                                "mask_data": predicted_label[0, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Tumor Core"},
                            },
                        },
                    )
                )
                
                # Whole Tumor visualization
                wandb_images.append(
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Whole-Tumor": {
                                "mask_data": sample_label[1, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Whole Tumor"},
                            },
                            "prediction/Whole-Tumor": {
                                "mask_data": predicted_label[1, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Whole Tumor"},
                            },
                        },
                    )
                )
                
                # Enhancing Tumor visualization
                wandb_images.append(
                    wandb.Image(
                        sample_image[channel_idx, :, :, slice_idx],
                        masks={
                            "ground-truth/Enhancing-Tumor": {
                                "mask_data": sample_label[2, :, :, slice_idx],
                                "class_labels": {0: "background", 1: "Enhancing Tumor"},
                            },
                            "prediction/Enhancing-Tumor": {
                                "mask_data": predicted_label[2, :, :, slice_idx] * 2,
                                "class_labels": {0: "background", 2: "Enhancing Tumor"},
                            },
                        },
                    )
                )
            
            table.add_data(
                "validation",
                data_idx,
                slice_idx,
                *wandb_images
            )
            progress_bar.update(1)
    
    return table
