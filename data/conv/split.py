import glob
import os
import re

import cv2

data_path = "../my_ears"
folder = "train"


def get_annotations(annot_f):
    d = {}
    with open(annot_f) as f:
        lines = f.readlines()
        for line in lines:
            (key, val) = line.split(',')
            # keynum = int(self.clean_file_name(key))
            d[key] = int(val)
    return d


output_path = os.path.join(data_path, "split", folder)
annotations_path = os.path.join(data_path, "annotations", "recognition", "ids.csv")
images_path = os.path.join(data_path, folder)

cla_d = get_annotations(annotations_path)
im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))
for im_name in im_list:
    img = cv2.imread(im_name)
    label = cla_d['/'.join(re.split(r'/|\\', im_name)[2:])]
    output_folder = os.path.join(output_path, str(label))
    if not os.path.exists(output_folder):  # create folders if not exists
        os.makedirs(output_folder)
    cv2.imwrite(os.path.join(output_folder, os.path.basename(im_name)), img)
