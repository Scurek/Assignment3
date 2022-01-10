import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess


class TrainAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        train_config = config['training']
        self.images_path = train_config['images_path']
        self.annotations_path = train_config['annotations_path']
        self.output_path = train_config['output_path']

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_training(self):
        print("Preparing data")
        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        preprocess = Preprocess()

        imgs = []
        labels = []

        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            # img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse
            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)

            imgs.append(img)
            labels.append(annot_list)

        print("Data prepared")
        print(labels)
        import trainers.cascade.train as cascade_trainer
        cascade_trainer = cascade_trainer.Trainer()
        cascade_trainer.train(imgs, np.array(labels))
        cascade_trainer.save(self.output_path + "/LBPH.yml")


if __name__ == '__main__':
    ev = TrainAll()
    ev.run_training()
