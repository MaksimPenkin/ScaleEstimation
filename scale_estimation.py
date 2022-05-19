"""
 @author   Maksim Penkin
"""


import numpy as np
import os
from PIL import Image

from pytorch_grad_cam import GradCAM

from utils.cmd_utils import parse_args
from utils.dir_utils import create_folder
from utils.image_utils import plot_line

from models.vgg19 import VGG19
from utils.base_utils import get_scale_search_net
from utils.base_utils import get_scale_search_inblock_matrix
from utils.base_utils import get_filtered_inblock_matrix


print("Welcome to scale_estimation.py!\n")
args = parse_args()

# Read input image.
print("[*] Reading input image: {}...".format(os.path.split(args.image)[-1]))
img = Image.open(args.image)
print("[*] Done.\n")

# Prepare result-directory.
base_res_dir = "./results"
res_dir = os.path.join(base_res_dir,
                       os.path.splitext(os.path.split(args.image)[-1])[0])
create_folder(base_res_dir, force=False, raise_except_if_exists=False)
create_folder(res_dir, force=False, raise_except_if_exists=False)

# Read CNN model (VGG19 only for now).
model = None
if args.cnn == 'VGG19':
    print("[*] Building VGG19 network...")
    model = VGG19()
    print("[*] Done.\n")

# Define Activations-&-Gradients extractor (GradCAM-based)
target_layers = []
if args.block2analyze == '1':
    target_layers = [model.block1[-1]]
elif args.block2analyze == '2':
    target_layers = [model.block2[-1]]
elif args.block2analyze == '3':
    target_layers = [model.block3[-1]]
elif args.block2analyze == '4':
    target_layers = [model.block4[-1]]
elif args.block2analyze == '5':
    target_layers = [model.block5[-1]]
acts_grads_engine = GradCAM(model=model, target_layers=target_layers)

# Define Scale-Search-Net
scale_search_net = get_scale_search_net(scale_net=[-3.0,
                                                   -2.75, -2.5, -2.25, -2.0,
                                                   -1.75, -1.5, -1.25,
                                                   0,
                                                   1.25, 1.5, 1.75, 2.0,
                                                   2.25, 2.5, 2.75, 3.0,
                                                   3.25, 3.5, 3.75, 4.0],
                                        original_size=224)

SCALES_NUM = len(scale_search_net.keys())

print("[*] Step 1 / 3: Calculating feature-maps ({0} Block #{1}) upon multiple scales...".format(args.cnn, args.block2analyze))
scale_net, inblock_matrix = get_scale_search_inblock_matrix(img, acts_grads_engine, scale_search_net)
print("    Got matrix of shape: {}.".format(inblock_matrix.shape))
print("[*] Done.\n")

plot_line(x_values=scale_net,
          y_values=inblock_matrix,
          x_label="Input scale: \n2 means x2-UpSample, \n -2 means x2-DownSample",
          y_label="AvgPool[ReLU(...)]",
          save_path=os.path.join(res_dir,
                                 'out_block{}.png'.format(args.block2analyze)))

print("[*] Step 2 / 3: Choosing meaningful feature-maps...")
fmap_save, filtered_inblock_matrix = get_filtered_inblock_matrix(inblock_matrix)
fmap_save_perc = len(fmap_save) / inblock_matrix.shape[1] * 100
print("    Got matrix of shape: {}.".format(filtered_inblock_matrix.shape))
print("    Found {0} appropriate feature maps out of {1} ({2:.2f} %).".format(len(fmap_save),
                                                                              inblock_matrix.shape[1],
                                                                              fmap_save_perc))
print("[*] Done.\n")

plot_line(x_values=scale_net,
          y_values=filtered_inblock_matrix,
          x_label="Input scale: \n2 means x2-UpSample, \n -2 means x2-DownSample",
          y_label="AvgPool[ReLU(...)]",
          save_path=os.path.join(res_dir,
                                 'out_block{}_filtered.png'.format(args.block2analyze)))

print("[*] Step 3 / 3: Merging meaningful feature_maps...")
filtered_merged_inblock_matrix = np.mean(filtered_inblock_matrix, axis=1)
print("    Got matrix of shape: {}.".format(filtered_merged_inblock_matrix.shape))
plot_line(x_values=scale_net,
          y_values=filtered_merged_inblock_matrix,
          x_label="Input scale: \n2 means x2-UpSample, \n -2 means x2-DownSample",
          y_label="AvgPool[ReLU(...)]",
          save_path=os.path.join(res_dir,
                                 'out_block{}_filtered_merged.png'.format(args.block2analyze)))
print("[*] Done.\n")

# ---------------------------------------- #

# Saving info as TXT
create_folder(os.path.join(res_dir, "block_logs"), force=False, raise_except_if_exists=False)
with open(os.path.join(res_dir,
                       "block_logs",
                       "block_{}.txt".format(args.block2analyze)),
          "w") as f:
    i_max = np.argmax(filtered_merged_inblock_matrix)
    i_min = np.argmin(filtered_merged_inblock_matrix)
    if not ((i_max not in [0, SCALES_NUM-1]) or (i_min not in [0, SCALES_NUM-1])):
        f.write("None")
    elif (i_max not in [0, SCALES_NUM-1]) and (i_min not in [0, SCALES_NUM-1]):
        a = abs(filtered_merged_inblock_matrix[i_max])
        b = abs(filtered_merged_inblock_matrix[i_min])
        if a > b:
            f.write(str(scale_net[i_max]))
        else:
            f.write(str(scale_net[i_min]))

    elif i_max not in [0, SCALES_NUM-1]:
        f.write(str(scale_net[i_max]))
    else:
        f.write(str(scale_net[i_min]))
    f.write("\n{0:.2f}".format(fmap_save_perc))

print("[*] Find results in: {}\n".format(res_dir))

print("Implementation is finished.")
