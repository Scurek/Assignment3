import glob
import os
from pathlib import Path
from preprocess import Preprocess

import cv2

data = "../data/ears/test"
#output = "../data/ears_bright/test"
output = "../outside/yolov3/datasets/ears_bright/images/test"
preprocess = Preprocess()

if not os.path.isdir(output):
    os.makedirs(output)

im_list = sorted(glob.glob(data + '/*.png', recursive=True))
for im_name in im_list:
    img = cv2.imread(im_name)
    img = preprocess.increase_brightness(img, 75)
    full_path = os.path.join(output, Path(os.path.basename(im_name)).stem) + ".png"
    cv2.imwrite(full_path, img)
