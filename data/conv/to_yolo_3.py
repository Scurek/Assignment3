import glob
import os
from pathlib import Path
import cv2
from tqdm import tqdm
inputs = [
    "../ears/annotations/detection/train_YOLO_format",
    "../ears/annotations/detection/test_YOLO_format"
]

inputImgs = [
    "../ears/train",
    "../ears/test"
]

outputs = [
    "./train",
    "./test"
]

# path = "../ears/annotations/detection/train_YOLO_format"
# pathImg = "../ears/train"
# output = "./yolo3"

# print(os.listdir(path))

for i in tqdm(range(len(inputs)), desc="Converting..."):
    path = inputs[i]
    pathImg = inputImgs[i]
    output = outputs[i]
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as f:
            with open(os.path.join(output, file), mode="w") as o:
                lines = f.readlines()
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    x, y, w, h = l_arr
                    # print(l_arr)
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    x_mid = (2*x + w) / 2
                    y_mid = (2*y + h) / 2
                    # print(os.path.join(pathImg, Path(os.path.basename(file)).stem) + '.png')
                    img = cv2.imread(os.path.join(pathImg, Path(os.path.basename(file)).stem) + '.png')
                    img_w = int(img.shape[1])
                    img_h = int(img.shape[0])
                    # print(img_w, img_h)
                    y_x = x_mid / img_w
                    y_y = y_mid / img_h
                    y_w = w / img_w
                    y_h = h / img_h

                    o.write("0 %f %f %f %f\n" % (y_x, y_y, y_w, y_h))
                    # print("0 %f %f %f %f" % (y_x, y_y, y_w, y_w))

# with open(os.path.join(path, "train.txt")) as f:
#     with open(os.path.join(path, output), mode="w") as o:
#         # "/ content / TrainYourOwnYOLO / Data / Source_Images"
#         lines = f.readlines()
#         for line in lines:
#             l_arr = line.split(" ")
#             l_arr_name = l_arr[0].split("/")[1]
#             l_arr_params = l_arr[1:5]
#             o.write(web_path + "/" + l_arr_name + " " + ",".join(l_arr_params) + ",0\n")
#