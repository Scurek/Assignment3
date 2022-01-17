import os
import pickle
from itertools import cycle

import matplotlib.pyplot as plt

# dataset = "perfectly_detected_ears"
dataset = "my_ears"

# title = "Perfectly Detected Ears - Comparison"
# CMC_list = {
#     "LBP": 'LBP_v_13.res',
#     "Bloc LBP": 'BlocLBP_v_12.res',
#     "pix2pix": 'pix2pix_v_13.res'
# }
# title = "CNN Comparison"
# CMC_list = {
#     "No Changes": 'best_model_v_9_r_1_e_50.res',
#     "Grayscale": 'best_model_v_10_r_1_e_50.res',
#     "Rotation": 'best_model_v_11_r_1_e_50.res',
#     "Blur": 'best_model_v_12_r_1_e_50.res',
#     "Brightness": 'best_model_v_13_r_1_e_50.res',
#     "All Augmentations": 'best_model_v_14_r_1_e_50.res',
#     "Grayscale (Repeated)": 'best_model_v_10_r_2_e_50.res',
#     "Rotation (Repeated)": 'best_model_v_11_r_2_e_50.res',
#     "Rotation (Repeated, depth 152)": 'best_model_d_ears---ibb_v_11_r_2_e_50.res'
# }
title = "CNN Comparison"
CMC_list = {
    "No Changes": 'best_model_d_myears_v_1_r_1_e_50.res',
    "Grayscale": 'best_model_d_myears_v_2_r_1_e_50.res',
    "Rotation": 'best_model_d_myears_v_3_r_1_e_50.res',
    "Blur": 'best_model_d_myears_v_4_r_1_e_50.res',
    "Brightness": 'best_model_d_myears_v_5_r_1_e_50.res',
    "All Augmentations": 'best_model_d_myears_v_6_r_1_e_50.res',
    "Grayscale (Repeated)": 'best_model_d_myears_v_2_r_2_e_50.res',
    "Rotation (Repeated)": 'best_model_d_myears_v_3_r_2_e_50.res',
    "Rotation (Repeated, depth 152)": 'best_model_d_myears_v_3_r_2_e_50_152.res'
}
output = title.replace(" ", "_")
plt.figure(figsize=(7.5, 3.75), layout='constrained')
lines = ["--", "-.", ":"]
linecycler = cycle(lines)

for name, CMC in CMC_list.items():
    with open(os.path.join(dataset, CMC), 'rb') as f:
        acc = pickle.load(f)
    plt.plot(range(1, len(acc) + 1), acc, next(linecycler), label=name)
plt.ylabel('Recognition rate[%]')
plt.xlabel('Rank')
plt.title(title)
plt.legend()
plt.savefig(os.path.join(dataset, output + ".png"))
plt.show()