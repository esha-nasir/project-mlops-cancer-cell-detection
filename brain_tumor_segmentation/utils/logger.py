import os
from datetime import datetime
from pathlib import Path

import git
import mlflow
import torch

import wandb


class WandBLogger:
    def __init__(self, config):
        """Initialize W&B and MLflow logger with project configuration."""
        self.config = config
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        self._init_wandb()
        self._init_mlflow()

        # Log git commit ID
        try:
            repo = git.Repo(search_parent_directories=True)
            commit_id = repo.head.object.hexsha
            wandb.log({"git_commit_id": commit_id})
            mlflow.log_param("git_commit_id", commit_id)
        except Exception as e:
            print(f"Warning: Could not log git commit ID: {str(e)}")

    def _init_wandb(self):
        """Initialize W&B run with configuration."""
        wandb_config = {
            "project": self.config.get("training", {})
            .get("wandb", {})
            .get("project", "default-project"),
            "entity": self.config.get("training", {}).get("wandb", {}).get("entity"),
            "config": dict(self.config),
        }
        wandb.init(**wandb_config)

        # Define metric steps
        wandb.define_metric("epoch/epoch_step")
        wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
        wandb.define_metric("batch/batch_step")
        wandb.define_metric("batch/*", step_metric="batch/batch_step")
        wandb.define_metric("validation/validation_step")
        wandb.define_metric("validation/*", step_metric="validation/validation_step")

    def _init_mlflow(self):
        """Initialize MLflow run."""
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment(
            self.config.get("training", {})
            .get("wandb", {})
            .get("project", "default-project")
        )
        mlflow.start_run()
        mlflow.log_params(dict(self.config))

    def log_metrics(self, metrics: dict, step_type: str = "batch"):
        """Log metrics to W&B, MLflow, and save to plots directory."""
        if step_type not in ["batch", "epoch", "validation"]:
            raise ValueError("step_type must be 'batch', 'epoch', or 'validation'")

        # Add step type prefix to metrics
        prefixed_metrics = {f"{step_type}/{k}": v for k, v in metrics.items()}

        # Add step counter
        if step_type == "batch":
            prefixed_metrics["batch/batch_step"] = metrics.get("step", 0)
        elif step_type == "epoch":
            prefixed_metrics["epoch/epoch_step"] = metrics.get("step", 0)
        elif step_type == "validation":
            prefixed_metrics["validation/validation_step"] = metrics.get("step", 0)

        # Log to W&B
        wandb.log(prefixed_metrics)

        # Log to MLflow
        for k, v in prefixed_metrics.items():
            mlflow.log_metric(
                k, v, step=prefixed_metrics.get(f"{step_type}/{step_type}_step", 0)
            )

        # Save metrics to plots directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.plots_dir / f"metrics_{timestamp}.txt", "a") as f:
            f.write(f"[{timestamp}] {prefixed_metrics}\n")

    def log_model_checkpoint(self, model, epoch: int):
        """Log model checkpoint as W&B and MLflow artifact."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}_{timestamp}.pth"

        torch.save(model.state_dict(), checkpoint_path)

        # W&B artifact
        artifact = wandb.Artifact(
            name=f"model-checkpoint-epoch-{epoch}",
            type="model",
            metadata=dict(epoch=epoch, timestamp=timestamp),
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)

        # MLflow artifact
        mlflow.log_artifact(str(checkpoint_path))

        # Optionally remove local checkpoint
        if self.config.get("cleanup_checkpoints", True):
            os.remove(checkpoint_path)

    def log_model_artifact(self, model_path: str, artifact_name: str):
        """Log model file as MLflow artifact."""
        mlflow.log_artifact(model_path, artifact_path="models")

    def finish(self):
        """Finish W&B and MLflow runs."""
        wandb.finish()
        mlflow.end_run()
