import cv2, os
import numpy as np

class Detector:
    # This example of a detector detects faces. However, you have annotations for ears!

    # cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades',
    # 'lbpcascade_frontalface.xml'))
    left_ear = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades',
                                                  "haarcascade_mcs_leftear.xml"))
    right_ear = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades',
                                                   "haarcascade_mcs_rightear.xml"))

    def __init__(self, scale_factor, min_neighbors):
        self.scaleFactor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, img, im_name):
        det_list_left = self.left_ear.detectMultiScale(img, self.scaleFactor, self.min_neighbors)
        det_list_right = self.right_ear.detectMultiScale(img, self.scaleFactor, self.min_neighbors)
        # print(det_list_right, det_list_left)
        if len(det_list_left) > 0 and len(det_list_right) > 0:
            comb = np.concatenate((det_list_left, det_list_right), axis=0)
            # print("comb", comb)
            return comb
        elif len(det_list_left) > 0:
            return det_list_left
        elif len(det_list_right) > 0:
            return det_list_right
        else:
            return ()


# if __name__ == '__main__':
#     fname = sys.argv[1]
#     img = cv2.imread(fname)
#     detector = CascadeDetector()
#     detected_loc = detector.detect(img)
#     for x, y, w, h in detected_loc:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
#     cv2.imwrite(fname + '.detected.jpg', img)
