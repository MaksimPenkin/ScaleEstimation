"""
 @author   Maksim Penkin
"""


import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms

from utils.nn_utils import get_image_tensor


def get_scale_search_net(scale_net=[-3.0, -2.0, -1.6, 0, 1.5, 2.0, 2.5, 3.0, 4.0],
                         original_size=224):
    size_net = []
    for s in scale_net:
        if s < 0:
            s_n = abs(float(original_size) / float(s))
            size_net.append(int(s_n))
        elif s == 0:
            size_net.append(int(original_size))
        else:
            s_n = abs(float(original_size) * float(s))
            size_net.append(int(s_n))

    return OrderedDict(zip(scale_net, size_net))


def get_scale_search_inblock_matrix(img,
                                    acts_grads_engine,
                                    scale_search_net):
    """
    inblock_matrix: [
                        [{blob_scale-1, fmap_1}, ..., {blob_scale-1, fmap_M}],
                        ...
                        [{blob_scale-N, fmap_1}, ..., {blob_scale-N, fmap_M}],
                    ]
    """

    scale_net = list(scale_search_net.keys())
    value_net = [[] for _ in range(len(scale_net))]

    inblock_matrix = OrderedDict(zip(scale_net, value_net))

    for scale_curr, size_curr in tqdm(scale_search_net.items()):
        transform = transforms.Compose([
            transforms.Resize(size_curr),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        batch_t = get_image_tensor(img, transform)
        acts, grads = acts_grads_engine(input_tensor=batch_t, targets=None)
        assert len(acts) == len(grads)
        assert len(grads) == 1  # Ensure target_layers was chosen with len == 1.

        blob_t = nn.ReLU()(torch.from_numpy(acts[0]))
        blob_t = torch.nn.AdaptiveAvgPool2d((1, 1))(blob_t)

        blob = blob_t.detach().numpy().flatten()
        inblock_matrix[scale_curr].extend(blob)

    return scale_net, np.array(list(inblock_matrix.values()))


def get_filtered_inblock_matrix(inblock_matrix):
    scales_num, fmap_num = inblock_matrix.shape
    fmap_save = []

    for i in range(fmap_num):
        fmap = inblock_matrix[:, i]

        # Filtering procedure.
        v_max, i_max, v_min, i_min = fmap.max(), np.argmax(fmap), fmap.min(), np.argmin(fmap)
        v_left, v_right = fmap[0], fmap[-1]

        if np.isclose(v_max, v_min):
            # print('Skip {}, due to 1'.format(i))
            # Пропускаем случаи констатной связи.
            continue
        elif not ((i_max not in [0, scales_num - 1]) or (i_min not in [0, scales_num - 1])):
            # print('Skip {}, due to 2'.format(i))
            # Пропускаем случаи строгой монотонности.
            continue
        else:
            fmap_save.append(i)

    return fmap_save, inblock_matrix[:, fmap_save]
