from monai.networks.nets import SegResNet
import torch

def get_model(config) -> SegResNet:
    """Create and return SegResNet model based on configuration."""
    return SegResNet(
        blocks_down=config.blocks_down,
        blocks_up=config.blocks_up,
        init_filters=config.init_filters,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        dropout_prob=config.dropout_prob,
    )

def load_model_from_checkpoint(config, checkpoint_path: str) -> SegResNet:
    """Load model from checkpoint."""
    model = get_model(config)
    model.load_state_dict(torch.load(checkpoint_path))
    return model
