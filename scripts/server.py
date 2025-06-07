from pathlib import Path

import hydra
import numpy as np
import tritonclient.http as tritonhttp
from monai.transforms import (
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    NormalizeIntensity,
    Orientation,
    Resize,
    Spacing,
)
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Set up Triton client
    triton_client = tritonhttp.InferenceServerClient(url="localhost:8000")
    model_name = "brain_tumor_model"

    # Load and preprocess input image
    images_ts_dir = Path(cfg.dataset_dir) / "Task01_BrainTumour/imagesTs"
    image_path = next(images_ts_dir.glob("*.nii.gz"))
    transform = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        EnsureType(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=cfg.pixdim, mode="bilinear"),
        Resize(spatial_size=cfg.roi_size, mode="trilinear"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
    image = transform[0](image_path)
    for t in transform[1:]:
        image = t(image)
    image = image[np.newaxis, ...].astype(np.float32)

    # Prepare input for Triton
    inputs = [tritonhttp.InferInput("input", image.shape, "FP32")]
    inputs[0].set_data_from_numpy(image)
    outputs = [tritonhttp.InferRequestedOutput("output")]

    # Perform inference
    response = triton_client.infer(model_name, inputs, outputs=outputs)
    output = response.as_numpy("output")

    # Save output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / f"prediction_{image_path.stem}.npy", output)
    print(f"Inference completed. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
