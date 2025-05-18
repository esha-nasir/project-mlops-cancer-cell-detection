# from monai.transforms import MapTransform
# import torch

# class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
#     """
#     Convert labels to multi channels based on brats classes
#     """
#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             result = []
#             # merge label 2 and label 3 to construct TC
#             result.append(torch.logical_or(d[key] == 2, d[key] == 3))
#             # merge labels 1, 2 and 3 to construct WT
#             result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
#             # label 2 is ET
#             result.append(d[key] == 2)
#             d[key] = torch.stack(result, axis=0).float()
#         return d

from monai.transforms import MapTransform
import torch

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on BraTS classes.
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # TC: Tumor Core = label 2 and 3
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # WT: Whole Tumor = label 1, 2, and 3
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # ET: Enhancing Tumor = label 2
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
            print(f"Converted label shape: {d[key].shape}")
        return d
