import glob
import json
import re

import cv2
import os
import numpy as np


def get_annotations(annot_f):
    d = {}
    with open(annot_f) as f:
        lines = f.readlines()
        for line in lines:
            (key, val) = line.split(',')
            # keynum = int(self.clean_file_name(key))
            d[key] = int(val)
    return d


def prepare_training_data():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    images_path = config['train_path']
    annotations_path = config['annotations_path']
    cla_d = get_annotations(annotations_path)
    faces = []
    labels = []
    im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))

    for im_name in im_list:
        #     im_name = im_list[i]
        # Read an image
        img = cv2.imread(im_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (200, 200))
        faces.append(img)

        ann_name = '/'.join(re.split(r'/|\\', im_name)[2:])
        # ann_name = '/'.join(im_name.split('/', "\\\\")[2:])
        labels.append(cla_d[ann_name])

    return faces, labels


with open('config_recognition_train.json') as config_file:
    config = json.load(config_file)
config = config['my_ears']

print("Preparing data...")
faces, labels = prepare_training_data()
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

print("Training...")
face_recognizer.train(faces, np.array(labels))
print("Training complete!")


def predict(test_img):
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    label, confidence = face_recognizer.predict(gray)
    return label, confidence


def test_pred():
    images_path = config['test_path']
    annotations_path = config['annotations_path']
    cla_d = get_annotations(annotations_path)
    im_list = sorted(glob.glob(images_path + '/*.png', recursive=True))

    correct = 0
    all_pred = 0
    for im_name in im_list:
        #     im_name = im_list[i]
        # Read an image
        img = cv2.imread(im_name)
        img = cv2.resize(img, (200, 200))
        label, confidence = predict(img)

        ann_name = '/'.join(re.split(r'/|\\', im_name)[2:])
        # ann_name = '/'.join(im_name.split('/', "\\\\")[2:])

        if label == cla_d[ann_name]:
            correct += 1
        all_pred += 1
    return correct, all_pred


print("Testing...")
correct, all_pred = test_pred()
print("Testing complete!")
print(f"Results:\n Rank1[%]: {correct / all_pred * 100}")
