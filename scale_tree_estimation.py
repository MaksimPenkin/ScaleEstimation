"""
 @author   Maksim Penkin
"""


import os
import numpy as np
import pandas as pd

from utils.cmd_utils import parse_args
from utils.image_utils import plot_scatter


print("Welcome to scale_tree_estimation.py!\n")
args = parse_args()

# Prepare result-directory.
base_dir = "./results"
block_logs_dir = os.path.join(base_dir,
                              os.path.splitext(os.path.split(args.image)[-1])[0],
                              "block_logs")

block_net = []
scale_net = []
df_dict = dict()
df_dict['image'] = [os.path.splitext(os.path.split(args.image)[-1])[0]]

for txt_name in os.listdir(block_logs_dir):
    name = os.path.splitext(txt_name)[0]
    block_id = int(name.split('_')[-1])

    with open(os.path.join(block_logs_dir, txt_name), 'rt') as f:
        txt_lines = f.read().splitlines()
    assert len(txt_lines) == 2

    scale, fm_perc = txt_lines
    if scale == 'None':
        scale = None
    else:
        scale = float(scale)
        block_net.append(block_id)
        scale_net.append(scale)
    fm_perc = float(fm_perc)

    df_dict['b{}_s'.format(block_id)] = [scale]
    df_dict['b{}_fm'.format(block_id)] = [fm_perc]
df_dict['s'] = [np.mean(scale_net)]

# ---------------------------------------- #

# Saving results as PLT
plot_scatter(x_values=block_net,
             y_values=scale_net,
             x_label="Block",
             y_label="Chosen Scale",
             x_lim=(0, 6),
             y_lim=(-6, 6),
             save_path=os.path.join(base_dir,
                                    os.path.splitext(os.path.split(args.image)[-1])[0],
                                    "tree_solution.png"))
# Saving results as CSV
df = pd.DataFrame.from_dict(df_dict)
csv_log = os.path.join(base_dir,
                       "dataset_log.csv")
df.to_csv(csv_log, index=False, mode='a', header=not os.path.exists(csv_log))

print("Implementation is finished.")
