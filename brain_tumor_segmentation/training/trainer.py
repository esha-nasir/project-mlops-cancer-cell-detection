import pytorch_lightning as pl
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from brain_tumor_segmentation.models.segresnet import get_model


class Trainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)  # Log hyperparameters
        self.model = get_model(config)

        self.loss_function = DiceLoss(
            smooth_nr=config.dice_loss.smoothen_numerator,
            smooth_dr=config.dice_loss.smoothen_denominator,
            squared_pred=config.dice_loss.squared_prediction,
            to_onehot_y=config.dice_loss.target_onehot,
            sigmoid=config.dice_loss.apply_sigmoid,
        )

        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.dice_metric_batch = DiceMetric(
            include_background=True, reduction="mean_batch"
        )
        self.post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log("batch/train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch["image"], batch["label"]
        outputs = self.inference(inputs)
        outputs = [self.post_trans(i) for i in decollate_batch(outputs)]

        self.dice_metric(y_pred=outputs, y=labels)
        self.dice_metric_batch(y_pred=outputs, y=labels)

    def validation_epoch_end(self, outputs):
        metric_value = self.dice_metric.aggregate().item()
        metric_batch = self.dice_metric_batch.aggregate()
        self.log("validation/mean_dice", metric_value)
        self.log("validation/mean_dice_tumor_core", metric_batch[0].item())
        self.log("validation/mean_dice_whole_tumor", metric_batch[1].item())
        self.log("validation/mean_dice_enhanced_tumor", metric_batch[2].item())
        self.dice_metric.reset()
        self.dice_metric_batch.reset()

    def inference(self, input):
        return sliding_window_inference(
            inputs=input,
            roi_size=self.config.inference_roi_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.5,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.initial_learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.max_train_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
