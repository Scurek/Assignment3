import cv2, os
import numpy as np
from pathlib import Path

class Detector:
    base_path = "./outside/yolov3/runs/val"

    def __init__(self, folder, min_conf=0):
        self.path = os.path.join(self.base_path, folder)
        self.min_conf = min_conf

    def detect(self, img, im_name):
        def yolo3_to_cv2(y_x, y_y, y_w, y_h):
            img_w = int(img.shape[1])
            img_h = int(img.shape[0])
            w = img_w * y_w
            h = img_h * y_h
            x_mid = y_x * img_w
            y_mid = y_y * img_h
            x = (2 * x_mid - w) / 2
            y = (2 * y_mid - h) / 2
            return x, y, w, h

        res = []
        try:
            with open(os.path.join(self.path, "labels", Path(os.path.basename(im_name)).stem) + ".txt", mode="r") as f:
                lines = f.readlines()
                for line in lines:
                    l_arr = line.split(" ")
                    conf = float(l_arr[5])
                    if conf < self.min_conf:
                        continue
                    l_atr = l_arr[1:5]
                    l_atr = [float(i) for i in l_atr]
                    l_atr = yolo3_to_cv2(*l_atr)
                    res.append([int(round(i)) for i in l_atr])
        finally:
            return np.array(res)


# if __name__ == '__main__':
#     fname = sys.argv[1]
#     img = cv2.imread(fname)
#     detector = CascadeDetector()
#     detected_loc = detector.detect(img)
#     for x, y, w, h in detected_loc:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
#     cv2.imwrite(fname + '.detected.jpg', img)
