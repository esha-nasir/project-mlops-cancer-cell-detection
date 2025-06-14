## Brain Tumor Cancer Cell Segmentation

###  Overview

This project implements a 3D brain tumor segmentation model using the **MONAI** framework and **SegResNet** architecture. It segments tumors into:

* **Whole Tumor (WT)**
* **Tumor Core (TC)**
* **Enhancing Tumor (ET)**

It uses the BraTS dataset and includes:

* **Training**: PyTorch Lightning with Hydra configs
* **Data**: DVC with Google Drive
* **Logging**: MLflow + Weights & Biases (W\&B)
* **Deployment**: Triton Inference Server (ONNX/TensorRT)
* **Cloud**: Railway-app for inference(Clean UI, supports Dockerfile builds)

---

### Expected Performance

| Metric               | Target  | Latest    |
| -------------------- | ------- | --------- |
| Whole Tumor Dice     | > 0.85  | 0.8652    |
| Tumor Core Dice      | > 0.70  | 0.1582    |
| Enhancing Tumor Dice | > 0.65  | N/A       |
| HD95 (Hausdorff 95)  | < 50 mm | 43.431 mm |

>  *Low Tumor Core Dice may indicate data imbalance or poor augmentation (check ****`configs/training.yaml`****).*

---

###  Reproducibility

* Seed fixed in `configs/config.yaml`: `seed: 42`
* Applied via `torch.manual_seed(cfg.seed)`
* Lightning's `deterministic=True` for reproducibility

---

### ⚙️ Setup

1. **Clone Repository**

   ```bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. **Environment Setup**

   ```bash
   curl -L https://install.python-poetry.org | python3 -
   poetry env use python3.8
   poetry shell
   poetry install
   ```

3. **Code Quality (Optional)**

   ```bash
   pre-commit install
   pre-commit run -a
   ```
   Note: Run the pre-commit run -a twice, because black, isort, modify the files in place during the first run.
![git-commit](https://github.com/user-attachments/assets/9571dc1c-3dc4-41d2-aabf-0f5340762222)


4. **DVC + Google Drive**

   ```bash
   pip install dvc[gdrive]
   dvc pull
   ```

5. **(Optional) Launch MLflow**

   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

---

### Training

```bash
poetry run python scripts/train.py
```

## Training: `train.py`

The `train.py` script handles end-to-end training of the brain tumor segmentation model using PyTorch Lightning and MONAI.

---

### Workflow Summary

1. **Configuration & Setup**
   - Loads settings via Hydra from `config.yaml`
   - Creates necessary directories for dataset and checkpoints
   - Sets random seed for reproducibility

2. **Data Loading**
   - Downloads the dataset using DVC
   - Applies training and validation transforms
   - Loads training and validation datasets via MONAI

3. **Logging**
   - Initializes [Weights & Biases (W&B)](https://wandb.ai/) logger
   - Optionally logs visual samples from the datasets

4. **Model Preparation**
   - Initializes the segmentation model using a custom `Trainer` class
   - Wraps the model with PyTorch Lightning for structured training

5. **Training Setup**
   - Configures model checkpointing (best by validation Dice score)
   - Sets training parameters (epochs, device, validation intervals, etc.)

6. **Training Loop**
   - Starts training using `trainer.fit(...)`
   - Periodically evaluates on the validation set
   - Saves best model checkpoint automatically

7. **Post-training**
   - Logs the best model as an MLflow artifact
   - Finalizes W&B run

---

### Output

- Trained model checkpoint saved in `cfg.checkpoint_dir`
- Metrics and visual logs tracked in W&B dashboard
- Final model saved as `model.pth` and optionally logged to MLflow

---

>  **Note:** Before training, ensure DVC is initialized and datasets are accessible. Adjust training parameters in `config.yaml` as needed.




**Includes**:

* DVC-pulled data
* Resize: `128x128x128`
* Logs to: MLflow, W\&B
* Checkpoints: `checkpoints/model.pth`

---

### Production Preparation

1. **Export**

   ```bash
   poetry run python scripts/export.py
   ```

   Outputs:

   * `model.onnx`
   * `model.trt`
   * Placed in: `triton_models/brain_tumor_model/1/model.onnx`

2. **Triton Repository**

   ```
   triton_models/
     └── brain_tumor_model/
           ├── 1/
           │   └── model.onnx
           └── config.pbtxt
   ```

---

### Inference

#### Option 1: Triton Server

```bash
docker run --rm --name triton_server -p8000:8000 -p8001:8001 -p8002:8002 \
-v $(pwd)/triton_models:/models \
nvcr.io/nvidia/tritonserver:24.03-py3 \
tritonserver --model-repository=/models
```

```bash
poetry run python scripts/server.py
```

#### Option 2: Local Inference

```bash
poetry run python scripts/infer.py
```

## Inference: `infer.py`

The `infer.py` script is used to evaluate a trained brain tumor segmentation model on the validation dataset.

###  Workflow Summary

1. **Configuration & Setup**
   - Loads settings via Hydra (`config.yaml`)
   - Sets the random seed for reproducibility

2. **Data Loading**
   - Downloads validation data using DVC
   - Applies validation-time transforms
   - Initializes the validation DataLoader

3. **Model Initialization**
   - Loads the specified model architecture (e.g., SegResNet)
   - Loads pretrained weights from `model.pth`
   - Transfers the model to the appropriate device (GPU or CPU)

4. **Inference & Post-processing**
   - Performs forward pass on each validation sample
   - Applies post-processing (e.g., argmax, thresholding)
   - Computes evaluation metrics: Dice Score & Hausdorff Distance (HD95)

5. **Metrics Logging**
   - Aggregates Dice & HD95 metrics across the dataset
   - Prints per-class metrics (Whole Tumor, Tumor Core, Enhancing Tumor)
   - Logs metrics to Weights & Biases (W&B) and MLflow

### Output
- Validation metrics printed to console
- Metrics logged to experiment tracking tools (W&B, MLflow)

> Ensure the checkpoint file `model.pth` exists in `cfg.checkpoint_dir` before running inference.


---

###  Data Management

* Dataset: `dataset/Task01_BrainTumour/`
* DVC: with Google Drive
* Pulled automatically by `data/dataset.py`

---

### Example Input/Output

* **Input**: `imagesTs/BRATS_531.nii.gz`
* **Output**: `output/prediction_BRATS_531.nii.npy`

  * Shape: `[3, 128, 128, 128]`
  * Channels: WT, TC, ET

---

### Notes

* Uses MONAI's 3D transforms and SegResNet.
* Evaluation metrics: Dice, HD95.
* Pre-trained models available in `checkpoints/`.

---

Further following is the prediction
![prediction](https://github.com/user-attachments/assets/9d7e396d-1630-4b07-a514-0daa1614a5ed)

## Plots - W&B 
## Batch/Train Loss 
![W B Chart 07_06_2025, 17_47_01](https://github.com/user-attachments/assets/6058470d-043c-44ef-8ced-3bcc479d52b9)
## At Validation Log Mean_dice at different Stages
![W B Chart 07_06_2025, 17_57_22](https://github.com/user-attachments/assets/58beaca4-3e47-46cb-a3ef-e9ce9dc674fe)
## Note: 
I implemented both MLflow and WandB.ai, but as the project progressed, I found WandB more user-friendly. The plots I have shared are from the WandB.ai dashboard.
## Deployment: 
For the deployment in Railway.app there are two major steps create file DockerFile and check the folder structure in my case triton_models should contain model.trt file the exported file from tensorRT should be correctly specified and also noted that Triton's official image is hosted on nvcr.io or ghcr.io,
these registries (especially ghcr.io) sometimes block access in certain regions so we have to rehost the image on Docker Hub, in this project i did following steps
docker pulling docker pull nvcr.io/nvidia/tritonserver:24.03-py3
tagged with docker hub
docker tag nvcr.io/nvidia/tritonserver:24.03-py3 eshanasir/tritonserver:24.03-py3
and pushed it back to the docker hub
docker push {eshanasir}/tritonserver:24.03-py3, here eshanasir is my username you have to write yours
### There might be a possible reason why the app fails in production. It could be due to the free plan, which only provides 4GB of memory. Additionally, I couldn't access the required service because of region restrictions. Apart from this, everything is working fine. I have already explained the step-by-step approach for the deployment above.


![W B Chart 07_06_2025, 17_59_44](https://github.com/user-attachments/assets/703c4638-bf93-40df-85ce-fc66569d1ddc)

![W B Chart 07_06_2025, 18_00_18](https://github.com/user-attachments/assets/4c13e7c4-2463-406d-a7d6-61c40c985ba2)


![W B Chart 07_06_2025, 18_02_46](https://github.com/user-attachments/assets/70d1f406-1e60-4626-b1f7-b875cea663b2)
