import os
import pickle
from itertools import cycle

import matplotlib.pyplot as plt

dataset = "my_ears"
data_list = {
    "DPC": {
        "No Augmentation": 'pix2pix_v_2.res',
        "Rotation": 'pix2pix_v_3.res',
        "Blur": 'pix2pix_v_4.res',
        "Brightness": 'pix2pix_v_5.res',
        "All Augmentations": 'pix2pix_v_6.res'
    },
    "LBP": {
        "No Augmentation": 'LBP_v_2.res',
        "Rotation": 'LBP_v_3.res',
        "Blur": 'LBP_v_4.res',
        "Brightness": 'LBP_v_5.res',
        "All Augmentations": 'LBP_v_6.res'
    },
    "BLPB": {
        "No Augmentation": 'BlocLBP_v_2.res',
        "Rotation": 'BlocLBP_v_3.res',
        "Blur": 'BlocLBP_v_4.res',
        "Brightness": 'BlocLBP_v_5.res',
        "All Augmentations": 'BlocLBP_v_6.res'
    },
    "BLPB + MLP": {
        "No Augmentation": 'DNN_BlocLBP_v_2.res',
        "Rotation": 'DNN_BlocLBP_v_3.res',
        "Blur": 'DNN_BlocLBP_v_4.res',
        "Brightness": 'DNN_BlocLBP_v_5.res',
        "All Augmentations": 'DNN_BlocLBP_v_6.res'
    },
    "ResNet": {
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
}


output = dataset + "_full_table"


with open(os.path.join(dataset, output), "w") as o:
    for method, results in data_list.items():
        o.write('\\multirow{' + str(len(results)) + '}*{\\rotatebox[origin=c]{90}{' + method + '}}')
        for name, location in results.items():
            with open(os.path.join(dataset, location), 'rb') as f:
                acc = pickle.load(f)
            o.write(f'& {name} & {"%.2f" % acc[0]} & {"%.2f" % acc[4]} \\\\ \n')
        o.write("\\midrule")