import glob

import cv2, os
import numpy as np
from pathlib import Path

annot_path = "../ears/annotations/recognition/ids.csv"
base_path = "../../outside/yolov3/runs/detect"
folders = ["test", "train"]
img_path = "../ears/"
annot_output_path = "./annotations/ids.csv"

def get_annotations(annot_f):
    d = {}
    with open(annot_f) as f:
        lines = f.readlines()
        for line in lines:
            (key, val) = line.split(',')
            # keynum = int(self.clean_file_name(key))
            d[key] = int(val)
    return d

def yolo3_to_cv2(y_x, y_y, y_w, y_h, img):
    img_w = int(img.shape[1])
    img_h = int(img.shape[0])
    w = img_w * y_w
    h = img_h * y_h
    x_mid = y_x * img_w
    y_mid = y_y * img_h
    x = (2 * x_mid - w) / 2
    y = (2 * y_mid - h) / 2
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


cla_d = get_annotations(annot_path)


with open(annot_output_path, mode="w") as o:
    for folder in folders:
        img_counter = 1
        results = sorted(glob.glob(os.path.join(base_path, folder, 'labels') + '/*.txt'))
        for result in results:
            im_name, _ = os.path.splitext(os.path.basename(result))
            img = cv2.imread(os.path.join(img_path, folder, im_name + ".png"))
            label = cla_d[f'{folder}/{im_name}.png']
            with open(result, mode="r") as f:
                lines = f.readlines()
                for line in lines:
                    l_arr = line.rstrip().split(" ")
                    l_atr = l_arr[1:5]
                    l_atr = [float(i) for i in l_atr]
                    x, y, w, h = yolo3_to_cv2(*l_atr, img)
                    crop = img[y:y+h, x:x+w]
                    new_img_name = f'{img_counter:04d}.png'
                    cv2.imwrite(os.path.join(folder, new_img_name), crop)
                    o.write(f'{folder}/{new_img_name},{label}\n')
                    img_counter += 1