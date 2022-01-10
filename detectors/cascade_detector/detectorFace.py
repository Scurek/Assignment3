import cv2, os
import numpy as np


class Detector:
    # This example of a detector detects faces. However, you have annotations for ears!

    cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades',
                                                 'haarcascade_frontalface_default.xml'))

    def __init__(self, scale_factor, min_neighbors):
        self.scaleFactor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, img):
        return self.cascade.detectMultiScale(img, self.scaleFactor, self.min_neighbors)

# if __name__ == '__main__':
#     fname = sys.argv[1]
#     img = cv2.imread(fname)
#     detector = CascadeDetector()
#     detected_loc = detector.detect(img)
#     for x, y, w, h in detected_loc:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (128, 255, 0), 4)
#     cv2.imwrite(fname + '.detected.jpg', img)
