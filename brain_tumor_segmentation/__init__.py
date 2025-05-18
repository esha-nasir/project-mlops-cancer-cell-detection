"""Brain Tumor Segmentation package using MONAI."""

__version__ = "0.1.0"

# Import key components for easier access
from .data.dataset import get_datasets, get_val_dataset
from .models.segresnet import get_model, load_model_from_checkpoint
from .training.trainer import Trainer
from .utils.logger import WandBLogger
from .utils.visualization import log_data_samples, log_predictions

__all__ = [
    'get_datasets',
    'get_val_dataset',
    'get_model',
    'load_model_from_checkpoint',
    'Trainer',
    'WandBLogger',
    'log_data_samples', 
    'log_predictions'
]
