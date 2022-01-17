import cv2, sys
from skimage import feature
import numpy as np


class BlockLBP:
    def __init__(self, num_points=8, radius=1, eps=1e-6, resize=200, blocks=(8, 8), do_format=True):
        self.num_points = num_points * radius
        self.radius = radius
        self.eps = eps
        self.resize = resize
        self.blocks = blocks
        self.do_format = do_format

    def extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.do_format:
            img = cv2.resize(img, (self.resize, self.resize))
        x_size = int(self.resize / self.blocks[0])
        y_size = int(self.resize / self.blocks[1])
        n_bins = self.num_points + 2
        hist = np.zeros(n_bins * self.blocks[0] * self.blocks[1])
        for i in range(self.blocks[0]):
            for j in range(self.blocks[1]):
                sub_img = img[i * x_size:(i + 1) * x_size, j * y_size:(j + 1) * y_size]
                lbp = feature.local_binary_pattern(sub_img, self.num_points, self.radius, method="uniform")
                local_hist, _ = np.histogram(lbp.ravel(),
                                             density=True,
                                             bins=np.arange(0, n_bins + 1),
                                             range=(0, n_bins))

                # n_bins = int(lbp.max() + 1)
                # local_hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
                # normalize the histogram
                # local_hist = local_hist.astype("float")
                # local_hist /= local_hist.sum() + self.eps
                # hist = np.concatenate((hist, local_hist), axis=None)
                start = i * self.blocks[1] * n_bins + j * n_bins
                hist[start: start + n_bins] = local_hist
        return hist


if __name__ == '__main__':
    fname = sys.argv[1]
    img = cv2.imread(fname)
    extractor = BlockLBP()
    features = extractor.extract(img)
    print(features)
