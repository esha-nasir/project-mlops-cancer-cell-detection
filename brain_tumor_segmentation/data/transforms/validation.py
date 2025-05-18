# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     EnsureTyped,
#     Orientationd,
#     Spacingd,
#     NormalizeIntensityd
# )
# from .base import ConvertToMultiChannelBasedOnBratsClassesd

# def get_val_transforms(config):
#     return Compose([
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=config.pixdim,
#             mode=("bilinear", "nearest"),
#         ),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#     ])


from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    SpatialPadd,
    CenterSpatialCropd,  # <-- New import
)
from .base import ConvertToMultiChannelBasedOnBratsClassesd

def get_val_transforms(config):
    return Compose([
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
        # NEW: Force validation samples to match training ROI size
        SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size),  # Pad if smaller
        CenterSpatialCropd(keys=["image", "label"], roi_size=config.roi_size),  # Crop if larger
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])