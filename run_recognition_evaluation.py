import math
import re

import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm

from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition_train_test import Evaluation


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config_recognition.json') as config_file:
            config = json.load(config_file)

        config = config['perfectly_detected_ears']
        self.train_path = config['train_path']
        self.test_path = config['test_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.train_path + '/*.png', recursive=True))
        im_list_test = sorted(glob.glob(self.test_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        cla_d = self.get_annotations(self.annotations_path)

        # Change the following extractors, modify and add your own

        # Pixel-wise comparison:
        import feature_extractors.pix2pix.extractor as p2p_ext
        pix2pix = p2p_ext.Pix2Pix()
        import feature_extractors.lbp.extractor as lbp_ext
        lbp = lbp_ext.LBP()
        import feature_extractors.block_lbp.extractor as block_lbp_ext
        block_lbp = block_lbp_ext.BlockLBP()

        feature_extractor = pix2pix

        train_features_arr = []
        y = []

        print("Extracting train features...")
        # for i in tqdm(range(len(im_list)), desc="Evaluating..."):
        for im_name in im_list:
            #     im_name = im_list[i]
            # Read an image
            img = cv2.imread(im_name)
            ann_name = '/'.join(re.split(r'/|\\', im_name)[2:])
            # ann_name = '/'.join(im_name.split('/', "\\\\")[2:])
            y.append(cla_d[ann_name])

            # Apply some preprocessing here
            # Run the feature extractors
            train_features = feature_extractor.extract(img)
            train_features_arr.append(train_features)
            # train_features = lbp.extract(img)
            # train_features_arr.append(train_features)
        print("Extracting test features...")
        x = []
        test_features_arr = []
        for im_name in im_list_test:
            #     im_name = im_list[i]
            # Read an image
            img = cv2.imread(im_name)
            ann_name = '/'.join(re.split(r'/|\\', im_name)[2:])
            # ann_name = '/'.join(im_name.split('/', "\\\\")[2:])
            x.append(cla_d[ann_name])
            test_features = feature_extractor.extract(img)
            test_features_arr.append(test_features)
        print("Calculating distances...")
        Y_plain = cdist(test_features_arr, train_features_arr, 'jensenshannon')
        print("Calculating measures...")
        r1 = eval.compute_rank1(Y_plain, x, y)
        print('Pix2Pix Rank 1[%]', r1)

        # r5 = eval.compute_rankX(Y_plain, y, 5)
        # print('Pix2Pix Rank 5[%]', r5)
        eval.CMC_plot(Y_plain, x, y, show=True)

        # Y_plain = cdist(train_features_arr, train_features_arr, 'jensenshannon')
        # r1 = eval.compute_rank1(Y_plain, y)
        # print('Pix2Pix Rank 1[%]', r1)


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
