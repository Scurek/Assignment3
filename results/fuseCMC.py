import os
import pickle
from itertools import cycle

import matplotlib.pyplot as plt

# dataset = "my_ears"
# title = "My Ears - Feature Extraction Methods Comparison"
# CMC_list = {
#     "LBP (rotation)": 'LBP_v_3.res',
#     "Bloc LBP (rotation)": 'BlocLBP_v_3.res',
#     "Bloc LBP with DNN (rotation)": 'DNN_BlocLBP_v_3.res',
#     "pix2pix (brightness)": 'pix2pix_v_5.res'
# }

# dataset = "my_ears"
# title = "My Ears - pix2pix"
# CMC_list = {
#     "No Augmentation": 'pix2pix_v_2.res',
#     "Rotation": 'pix2pix_v_3.res',
#     "Blur": 'pix2pix_v_4.res',
#     "Brightness": 'pix2pix_v_5.res',
#     "All Augmentations": 'pix2pix_v_6.res'
# }

# dataset = "my_ears"
# title = "My Ears - LBP"
# CMC_list = {
#     "No Augmentation": 'LBP_v_2.res',
#     "Rotation": 'LBP_v_3.res',
#     "Blur": 'LBP_v_4.res',
#     "Brightness": 'LBP_v_5.res',
#     "All Augmentations": 'LBP_v_6.res'
# }

# dataset = "my_ears"
# title = "My Ears - Bloc LBP"
# CMC_list = {
#     "No Augmentation": 'BlocLBP_v_2.res',
#     "Rotation": 'BlocLBP_v_3.res',
#     "Blur": 'BlocLBP_v_4.res',
#     "Brightness": 'BlocLBP_v_5.res',
#     "All Augmentations": 'BlocLBP_v_6.res'
# }

dataset = "my_ears"
title = "My Ears - Bloc LBP with MLP"
CMC_list = {
    "No Augmentation": 'DNN_BlocLBP_v_2.res',
    "Rotation": 'DNN_BlocLBP_v_3.res',
    "Blur": 'DNN_BlocLBP_v_4.res',
    "Brightness": 'DNN_BlocLBP_v_5.res',
    "All Augmentations": 'DNN_BlocLBP_v_6.res'
}

# dataset = "my_ears"
# title = "My Ears - CNN Comparison"
# CMC_list = {
#     "No Changes": 'best_model_d_myears_v_1_r_1_e_50.res',
#     "Grayscale": 'best_model_d_myears_v_2_r_1_e_50.res',
#     "Rotation": 'best_model_d_myears_v_3_r_1_e_50.res',
#     "Blur": 'best_model_d_myears_v_4_r_1_e_50.res',
#     "Brightness": 'best_model_d_myears_v_5_r_1_e_50.res',
#     "All Augmentations": 'best_model_d_myears_v_6_r_1_e_50.res',
#     "Grayscale (Repeated)": 'best_model_d_myears_v_2_r_2_e_50.res',
#     "Rotation (Repeated)": 'best_model_d_myears_v_3_r_2_e_50.res',
#     "Rotation (Repeated, depth 152)": 'best_model_d_myears_v_3_r_2_e_50_152.res'
# }

# dataset = "my_ears"
# title = "My Ears - All Methods"
# CMC_list = {
#     "LBP (rotation)": 'LBP_v_3.res',
#     "Bloc LBP (rotation)": 'BlocLBP_v_3.res',
#     "Bloc LBP with MLP (rotation)": 'DNN_BlocLBP_v_3.res',
#     "pix2pix (brightness)": 'pix2pix_v_5.res',
#     "ResNet18 (rotation)": 'best_model_d_myears_v_3_r_1_e_50.res',
#     "ResNet18 - repeated (rotation)": 'best_model_d_myears_v_3_r_2_e_50.res',
#     "ResNet152 - repeated (rotation)": 'best_model_d_myears_v_3_r_2_e_50_152.res'
# }



# dataset = "perfectly_detected_ears"
# title = "Perfectly Detected Ears - pix2pix"
# CMC_list = {
#     "No Augmentation": 'pix2pix_v_6.res',
#     "Rotation": 'pix2pix_v_11.res',
#     "Blur": 'pix2pix_v_12.res',
#     "Brightness": 'pix2pix_v_13.res',
#     "All Augmentations": 'pix2pix_v_14.res'
# }

# dataset = "perfectly_detected_ears"
# title = "Perfectly Detected Ears - LBP"
# CMC_list = {
#     "No Augmentation": 'LBP_v_6.res',
#     "Rotation": 'LBP_v_11.res',
#     "Blur": 'LBP_v_12.res',
#     "Brightness": 'LBP_v_13.res',
#     "All Augmentations": 'LBP_v_14.res'
# }

# dataset = "perfectly_detected_ears"
# title = "Perfectly Detected Ears - Bloc LBP"
# CMC_list = {
#     "No Augmentation": 'BlocLBP_v_6.res',
#     "Rotation": 'BlocLBP_v_11.res',
#     "Blur": 'BlocLBP_v_12.res',
#     "Brightness": 'BlocLBP_v_13.res',
#     "All Augmentations": 'BlocLBP_v_14.res'
# }

# dataset = "perfectly_detected_ears"
# title = "Perfectly Detected Ears - Bloc LBP with MLP"
# CMC_list = {
#     "No Augmentation": 'DNN_BlocLBP_v_6.res',
#     "Rotation": 'DNN_BlocLBP_v_11.res',
#     "Blur": 'DNN_BlocLBP_v_12.res',
#     "Brightness": 'DNN_BlocLBP_v_13.res',
#     "All Augmentations": 'DNN_BlocLBP_v_14.res'
# }

# dataset = "perfectly_detected_ears"
# title = "Perfectly Detected Ears - CNN Comparison"
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

# dataset = "perfectly_detected_ears"
# title = "Perfectly Detected Ears - All Methods"
# CMC_list = {
#     "LBP (brightness)": 'LBP_v_13.res',
#     "Bloc LBP (blur)": 'BlocLBP_v_12.res',
#     "Bloc LBP with MLP (rotation)": 'DNN_BlocLBP_v_11.res',
#     "pix2pix (brightness)": 'pix2pix_v_13.res',
#     "ResNet18 (rotation)": 'best_model_v_11_r_1_e_50.res',
#     "ResNet18 - repeated (rotation)": 'best_model_v_11_r_2_e_50.res',
#     "ResNet152 - repeated (rotation)": 'best_model_d_ears---ibb_v_11_r_2_e_50.res'
# }


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