import cv2, sys
from skimage import feature
import numpy as np

class Pix2Pix:
	def __init__(self, resize=100, do_format=True):
		self.resize = resize
		self.do_format = do_format

	def extract(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if self.do_format:
			img = cv2.resize(img, (self.resize, self.resize))

		img = img.ravel()
		
		return img

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	extractor = Pix2Pix()
	features = extractor.extract(img)
	print(features)