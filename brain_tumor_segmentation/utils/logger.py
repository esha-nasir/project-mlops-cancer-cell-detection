import wandb
from datetime import datetime
import torch
import os
from pathlib import Path

class WandBLogger:
    def __init__(self, config):
        """Initialize W&B logger with project configuration."""
        self.config = config
        self._init_wandb()
        
    def _init_wandb(self):
        """Initialize W&B run with configuration."""
        wandb_config = {
        'project': self.config.get('training', {}).get('wandb', {}).get('project', 'default-project'),
        'entity': self.config.get('training', {}).get('wandb', {}).get('entity'),
        'config': dict(self.config)
        }
        wandb.init(**wandb_config)
        
        # Define metric steps
        wandb.define_metric("epoch/epoch_step")
        wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
        wandb.define_metric("batch/batch_step")
        wandb.define_metric("batch/*", step_metric="batch/batch_step")
        wandb.define_metric("validation/validation_step")
        wandb.define_metric("validation/*", step_metric="validation/validation_step")
    
    def log_metrics(self, metrics: dict, step_type: str = "batch"):
        """Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step_type: Type of step ('batch', 'epoch', or 'validation')
        """
        if step_type not in ["batch", "epoch", "validation"]:
            raise ValueError("step_type must be 'batch', 'epoch', or 'validation'")
        
        # Add step type prefix to metrics
        prefixed_metrics = {
            f"{step_type}/{k}": v for k, v in metrics.items()
        }
        
        # Add step counter
        if step_type == "batch":
            prefixed_metrics["batch/batch_step"] = metrics.get("step", 0)
        elif step_type == "epoch":
            prefixed_metrics["epoch/epoch_step"] = metrics.get("step", 0)
        elif step_type == "validation":
            prefixed_metrics["validation/validation_step"] = metrics.get("step", 0)
        
        wandb.log(prefixed_metrics)
    
    def log_model_checkpoint(self, model, epoch: int):
        """Log model checkpoint as W&B artifact."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}_{timestamp}.pth"
        
        torch.save(model.state_dict(), checkpoint_path)
        
        artifact = wandb.Artifact(
            name=f"model-checkpoint-epoch-{epoch}",
            type="model",
            metadata=dict(epoch=epoch, timestamp=timestamp),
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        
        # Optionally remove local checkpoint after uploading
        if self.config.get("cleanup_checkpoints", True):
            os.remove(checkpoint_path)
    
    def log_config(self, config):
        """Log configuration to W&B."""
        wandb.config.update(config)
    
    def finish(self):
        """Finish W&B run."""
        wandb.finish()

class LocalLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize local file logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{self.timestamp}.txt"
        
    def log(self, message: str, print_message: bool = True):
        """Log message to file and optionally print to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        
        if print_message:
            print(log_entry.strip())
    
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        log_entry = f"[{timestamp}] Step {step}: {metrics_str}\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry)
    
    def close(self):
        """Close logger (no-op for file logger)."""
        pass
