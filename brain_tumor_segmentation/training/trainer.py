
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from tqdm.auto import tqdm
import wandb
from monai.data import decollate_batch


class Trainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            config.initial_learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_train_epochs
        )
        
        self.loss_function = DiceLoss(
            smooth_nr=config.dice_loss.smoothen_numerator,
            smooth_dr=config.dice_loss.smoothen_denominator,
            squared_pred=config.dice_loss.squared_prediction,
            to_onehot_y=config.dice_loss.target_onehot,
            sigmoid=config.dice_loss.apply_sigmoid,
        )
        
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize W&B metrics
        self._init_wandb_metrics()
    
    def _init_wandb_metrics(self):
        wandb.define_metric("epoch/epoch_step")
        wandb.define_metric("epoch/*", step_metric="epoch/epoch_step")
        wandb.define_metric("batch/batch_step")
        wandb.define_metric("batch/*", step_metric="batch/batch_step")
        wandb.define_metric("validation/validation_step")
        wandb.define_metric("validation/*", step_metric="validation/validation_step")


    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0
        batch_step = 0
    
        total_batch_steps = len(train_loader)
        batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
    
        for batch_data in batch_progress_bar:
            inputs, labels = (
                batch_data["image"].to(self.device),
                batch_data["label"].to(self.device),
            )
        
            self.optimizer.zero_grad()
        
        # Modified gradient scaling
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
            
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
            epoch_loss += loss.item()
            batch_progress_bar.set_description(f"train_loss: {loss.item():.4f}")
            wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
            batch_step += 1
    
        self.lr_scheduler.step()
        epoch_loss /= total_batch_steps
    
        wandb.log({
            "epoch/epoch_step": epoch,
            "epoch/mean_train_loss": epoch_loss,
            "epoch/learning_rate": self.lr_scheduler.get_last_lr()[0],
        })
    
        return epoch_loss    
    
    # def train_epoch(self, train_loader, epoch):
    #     self.model.train()
    #     epoch_loss = 0
    #     batch_step = 0
        
    #     total_batch_steps = len(train_loader.dataset) // train_loader.batch_size
    #     batch_progress_bar = tqdm(train_loader, total=total_batch_steps, leave=False)
    #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Trainer>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
    #     for batch_data in batch_progress_bar:
    #         inputs, labels = (
    #             batch_data["image"].to(self.device),
    #             batch_data["label"].to(self.device),
    #         )
            
    #         self.optimizer.zero_grad()
            
    #         with torch.cuda.amp.autocast():
    #             outputs = self.model(inputs)
    #             loss = self.loss_function(outputs, labels)
            
    #         self.scaler.scale(loss).backward()
    #         self.scaler.step(self.optimizer)
    #         self.scaler.update()
            
    #         epoch_loss += loss.item()
    #         batch_progress_bar.set_description(f"train_loss: {loss.item():.4f}")
            
    #         # Log batch-wise training loss to W&B
    #         wandb.log({"batch/batch_step": batch_step, "batch/train_loss": loss.item()})
    #         batch_step += 1
        
    #     self.lr_scheduler.step()
    #     epoch_loss /= total_batch_steps
        
    #     # Log epoch-wise metrics
    #     wandb.log({
    #         "epoch/epoch_step": epoch,
    #         "epoch/mean_train_loss": epoch_loss,
    #         "epoch/learning_rate": self.lr_scheduler.get_last_lr()[0],
    #     })
        
    #     return epoch_loss
    
    def validate(self, val_loader, post_trans, epoch):
        self.model.eval()
        validation_step = epoch // self.config.validation_intervals
        
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(self.device),
                    val_data["label"].to(self.device),
                )
                
                val_outputs = self.inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                
                self.dice_metric(y_pred=val_outputs, y=val_labels)
                self.dice_metric_batch(y_pred=val_outputs, y=val_labels)
            
            metric_value = self.dice_metric.aggregate().item()
            metric_batch = self.dice_metric_batch.aggregate()
            
            # Log validation metrics
            wandb.log({
                "validation/validation_step": validation_step,
                "validation/mean_dice": metric_value,
                "validation/mean_dice_tumor_core": metric_batch[0].item(),
                "validation/mean_dice_whole_tumor": metric_batch[1].item(),
                "validation/mean_dice_enhanced_tumor": metric_batch[2].item(),
            })
            print("--------------------I AM HERE TO CHECK VALIDATION IS WORKING OR NOT-----------------------------------")
            self.dice_metric.reset()
            self.dice_metric_batch.reset()
            
            return metric_value
    
    def inference(self, input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.config.inference_roi_size,
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5,
            )
        
        with torch.cuda.amp.autocast():
            return _compute(input)
    
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        
        # Create W&B artifact
        artifact = wandb.Artifact(
            name=f"{wandb.run.id}-checkpoint", 
            type="model"
        )
        artifact.add_file(local_path=path)
        wandb.log_artifact(artifact)
        artifact.wait()