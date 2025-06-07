from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)

from .base import ConvertToMultiChannelBasedOnBratsClassesd


def get_train_transforms(config):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=config.pixdim,
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=config.roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0, lazy=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1, lazy=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2, lazy=True),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
