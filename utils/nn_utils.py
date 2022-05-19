"""
 @author   Maksim Penkin
"""


import torch


def get_image_tensor(img, transform):
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t
