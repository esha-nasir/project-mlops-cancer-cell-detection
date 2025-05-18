from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import torch

def get_dice_metrics() -> tuple:
    """Return initialized dice metrics."""
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    return dice_metric, dice_metric_batch

def get_dice_loss(config) -> DiceLoss:
    """Return configured Dice loss."""
    return DiceLoss(
        smooth_nr=config.dice_loss.smoothen_numerator,
        smooth_dr=config.dice_loss.smoothen_denominator,
        squared_pred=config.dice_loss.squared_prediction,
        to_onehot_y=config.dice_loss.target_onehot,
        sigmoid=config.dice_loss.apply_sigmoid,
    )

def get_postprocessing_transforms() -> tuple:
    """Return postprocessing transforms for inference."""
    from monai.transforms import Compose, Activations, AsDiscrete
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    print("000000000000------------>",post_trans)
    return post_trans
