import numpy as np

COEF_8 = 2**8 - 1
COEF_16 = 2**16 - 1

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

cifar10_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
cifar10_std = np.array([0.2471, 0.2435, 0.2616], dtype=np.float32)
