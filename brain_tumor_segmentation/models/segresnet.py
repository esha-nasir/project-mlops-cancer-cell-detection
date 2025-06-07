import torch
from monai.networks.nets import SegResNet


def get_model(config) -> SegResNet:
    """Create and return SegResNet model based on configuration."""
    print("-------------------->To check model")
    print("blocks_down ####--", config.blocks_down)
    print("locks_up ###--", config.blocks_up)
    print("init_filters ###--", config.init_filters)
    print("in_channels ###--", config.in_channels)
    print("out_channels ###--", config.out_channels)
    print("dropout_prob ###--", config.dropout_prob)
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
