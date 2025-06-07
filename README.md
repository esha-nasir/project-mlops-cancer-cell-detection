Brain Tumor Segmentation

Project Description

This project implements a 3D segmentation model for brain tumors using the MONAI framework. The model, based on the SegResNet architecture, segments brain tumors into three classes—Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)—using the BraTS dataset. It leverages PyTorch Lightning for training, DVC for data management (stored in Google Drive), Hydra for configuration management, MLflow and Weights & Biases (W&B) for logging, and Triton Inference Server for production inference. The model is exported to ONNX and TensorRT formats for optimized deployment.

Expected Model Performance

The SegResNet model is expected to achieve the following performance on the BraTS validation set:





Whole Tumor Dice: >0.85



Tumor Core Dice: >0.70



Enhancing Tumor Dice: >0.65



Hausdorff Distance (95th percentile): <50 mm

Current metrics (as of latest run, logged to MLflow and W&B):





Whole Tumor Dice: 0.8652



Tumor Core Dice: 0.1582



Enhancing Tumor Dice: Not consistently available



Mean HD95: 43.431 mm

Note: The low Tumor Core Dice score indicates potential issues with model training or data preprocessing, which may require further investigation (e.g., adjusting hyperparameters in configs/training.yaml or verifying dataset quality).

Reproducibility

To ensure reproducible training and validation results, a fixed random seed (seed: 42) is defined in configs/config.yaml. This seed is applied using torch.manual_seed(cfg.seed) in scripts/train.py and scripts/infer.py, ensuring consistent data shuffling, augmentations, and model initialization across runs. Additionally, PyTorch Lightning’s deterministic mode is enabled to minimize randomness in GPU operations.

Setup

To set up the project for development, training, or inference:





Clone the repository:

git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation



Set up the Python environment:





Ensure Python 3.8.10 is installed.



Install Poetry:

curl -sSL https://install.python-poetry.org | python3 -



Create and activate a virtual environment:

poetry env use python3.8
poetry shell



Install dependencies:

poetry install



Install pre-commit hooks:





Configure hooks for black, isort, flake8, and prettier:

pre-commit install



Verify code quality:

pre-commit run -a



Set up DVC for data management:





Install DVC with Google Drive support:

pip install dvc[gdrive]



Configure Google Drive credentials by creating a client_id and client_secret (see DVC Google Drive documentation). Update .dvc/config with these credentials.



Pull the BraTS dataset:

dvc pull





This downloads the dataset to dataset/Task01_BrainTumour.



Set up MLflow server (optional for logging):





Start the MLflow server locally:

mlflow server --host 127.0.0.1 --port 8080



Access the MLflow UI at http://127.0.0.1:8080 to view logged metrics and artifacts (e.g., model checkpoints in mlruns/).

Train

To train the SegResNet model:





Ensure the dataset is available (dvc pull).



Run the training script:

poetry run python scripts/train.py





This script:





Pulls the BraTS dataset via DVC if not present.



Applies preprocessing (e.g., resizing to 128x128x128, normalization) defined in configs/data.yaml.



Trains the model using PyTorch Lightning with hyperparameters from configs/model.yaml and configs/training.yaml.



Logs metrics (WT Dice, TC Dice, ET Dice, Mean HD95) to MLflow (http://127.0.0.1:8080) and W&B.



Saves model checkpoints to checkpoints/model.pth and logs them as artifacts.



Saves metric plots to plots/metrics_*.txt.

Production Preparation

To prepare the model for production:





Export the model to ONNX and TensorRT:

poetry run python scripts/export.py





Generates:





checkpoints/model.onnx: ONNX model for inference.



checkpoints/model.trt: TensorRT engine for optimized inference.



The ONNX model is copied to triton_models/brain_tumor_model/1/model.onnx for Triton Inference Server.



Set up Triton model repository:





Ensure triton_models/brain_tumor_model/ contains:





1/model.onnx: Exported ONNX model.



config.pbtxt: Configuration specifying input/output dimensions (e.g., [1, 4, 128, 128, 128] for input, [1, 3, 128, 128, 128] for output) and batch size.



These files are used by the Triton server for inference.

Inference

To perform inference on new data:





Prepare input data:





Place .nii.gz files (e.g., BRATS_531.nii.gz) in dataset/Task01_BrainTumour/imagesTs.



Files must follow the BraTS format with four MRI modalities (T1, T1ce, T2, FLAIR) in NIfTI format.



Pull the dataset:

dvc pull



Option 1: Inference with Triton Inference Server (Local):





Start the Triton server:

docker run --rm --name triton_server -p8000:8000 -p8001:8001 -p8002:8002 -v \\wsl$\Ubuntu\home\lick\projects\brain_tumor_segmentation\brain_tumor_segmentation\triton_models:/models nvcr.io/nvidia/tritonserver:24.03-py3 tritonserver --model-repository=/models --strict-model-config=false





Mounts triton_models and starts the server on ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics).



Run the inference script:

poetry run python scripts/server.py





Loads BRATS_531.nii.gz, applies preprocessing (resizing, normalization), sends data to the Triton server, and saves the output as output/prediction_BRATS_531.nii.npy (shape [3, 128, 128, 128] for WT, TC, ET masks).



Verify server status:

curl -v http://localhost:8000/v2/health/ready





Expected response: HTTP/1.1 200 OK.



Option 2: Local Inference (without Triton):





Run the local inference script:

poetry run python scripts/infer.py





Loads the dataset, applies validation transforms, performs inference with checkpoints/model.pth, logs metrics to MLflow/W&B, and saves visualizations to W&B.

Data Management

The BraTS dataset is managed with DVC and stored in Google Drive:





Install DVC: pip install dvc[gdrive].



Configure Google Drive credentials in .dvc/config.



Run dvc pull to fetch dataset/Task01_BrainTumour.



The dataset is automatically pulled during training (scripts/train.py) or inference (scripts/infer.py, scripts/server.py) via data/dataset.py.

Example Input Data

An example input is dataset/Task01_BrainTumour/imagesTs/BRATS_531.nii.gz, containing four MRI modalities (T1, T1ce, T2, FLAIR) in NIfTI format. It is resized to 128x128x128 with 1.0x1.0x1.0 mm voxel spacing (per configs/data.yaml). The output is a .npy file (e.g., output/prediction_BRATS_531.nii.npy) with shape [3, 128, 128, 128], representing segmentation masks for WT, TC, and ET.

Cloud Deployment (Fly.io Free Tier)

To deploy the Triton Inference Server on Fly.io’s free tier (3 VMs, 256MB RAM each, 3GB storage):





Install Fly.io CLI:

curl -L https://fly.io/install.sh | sh
flyctl auth signup



Initialize Fly.io app:





In the project directory:

flyctl launch





Choose an app name (e.g., brain-tumor-segmentation).



Select a region (e.g., iad for Ashburn, Virginia).



Decline database setup.



This generates fly.toml.



Create a Dockerfile:





Ensure a Dockerfile exists in the project root:

FROM nvcr.io/nvidia/tritonserver:24.03-py3
WORKDIR /models
COPY triton_models/brain_tumor_model /models/brain_tumor_model
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]



Deploy to Fly.io:

flyctl deploy





Builds and deploys the Triton server Docker image.



Check status:

flyctl status



View logs:

flyctl logs



Test the deployment:





Verify the server is ready:

curl -v https://brain-tumor-segmentation.fly.dev:8000/v2/health/ready





Expected response: HTTP/1.1 200 OK.



Update scripts/server.py to use the Fly.io endpoint:

triton_client = tritonhttp.InferenceServerClient(url="https://brain-tumor-segmentation.fly.dev:8000")



Run inference locally:

poetry run python scripts/server.py





Requires dataset/Task01_BrainTumour/imagesTs/BRATS_531.nii.gz (via dvc pull).



Monitor resources:





Check usage:

flyctl dashboard



The free tier supports limited traffic; upgrade to a paid plan for production use.

Note: Ensure triton_models/brain_tumor_model contains 1/model.onnx and config.pbtxt. For cloud-based data access, consider hosting the dataset on a public Google Drive link or S3 and updating data/dataset.py.

Troubleshooting





Triton server fails to start: Check flyctl logs or docker logs triton_server for errors (e.g., missing model.onnx).



MLflow artifacts empty: Ensure cleanup_checkpoints is False in utils/logger.py.



Low metrics: Investigate low Tumor Core Dice (0.1582) by reviewing configs/training.yaml or dataset preprocessing.



Fly.io limits: The free tier may pause if bandwidth (160GB/month) or CPU limits are exceeded.

Further following is the prediction
![prediction](https://github.com/user-attachments/assets/9d7e396d-1630-4b07-a514-0daa1614a5ed)

