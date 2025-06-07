## Brain Tumor Segmentation

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
* **Cloud**: Fly.io for inference

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

![W B Chart 07_06_2025, 17_59_44](https://github.com/user-attachments/assets/703c4638-bf93-40df-85ce-fc66569d1ddc)

![W B Chart 07_06_2025, 18_00_18](https://github.com/user-attachments/assets/4c13e7c4-2463-406d-a7d6-61c40c985ba2)


![W B Chart 07_06_2025, 18_02_46](https://github.com/user-attachments/assets/70d1f406-1e60-4626-b1f7-b875cea663b2)








