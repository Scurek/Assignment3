from datetime import datetime
import csv
import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
from save.save_annotated import saveImg, saveAll
from tqdm import tqdm


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        eval_config = config['evaluation']
        self.images_path = eval_config['images_path']
        self.annotations_path = eval_config['annotations_path']
        self.output_path = eval_config['output_path']

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_evaluation(self):
        LIMIT = -1
        SAVE = True
        MAXCOLUMNS = 2
        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()

        today = datetime.now()
        dirname = "res_" + today.strftime("%Y-%m-%d-%H-%M-%S")

        self.output_path = os.path.join(self.output_path, dirname)

        os.mkdir(self.output_path)

        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        import detectors.yolo3.detector as yolo3_detector
        import detectors.yolo3_heu.detector as yolo3_heu_detector
        # import detectors.your_super_detector.detector as super_detector

        from detector_configs import detectors_yolo

        detectors = detectors_yolo

        # Types: 0:TP 1:FP 2:FN
        # detectors = {
        #     "Cascade": {
        #         "detector": cascade_detector.Detector(1.10, 1),
        #         "color": (0, 255, 0),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Tiny300_ConfTresh-0.25": {
        #         "detector": yolo3_detector.Detector("Tiny300", 0.25),
        #         "color": (0, 0, 255),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Tiny300_ConfTresh-0.75": {
        #         "detector": yolo3_detector.Detector("Tiny300", 0.75),
        #         "color": (0, 0, 255),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Tiny50_ConfTresh-0.25": {
        #         "detector": yolo3_detector.Detector("Tiny50", 0.25),
        #         "color": (0, 0, 255),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Tiny50_ConfTresh-0.75": {
        #         "detector": yolo3_detector.Detector("Tiny50", 0.75),
        #         "color": (0, 0, 255),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Norm50_ConfTresh-0.25": {
        #         "detector": yolo3_detector.Detector("Norm50", 0.25),
        #         "color": (0, 255, 255),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Norm50_ConfTresh-0.75": {
        #         "detector": yolo3_detector.Detector("Norm50", 0.75),
        #         "color": (0, 255, 255),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "spp50_ConfTresh-0.25": {
        #         "detector": yolo3_detector.Detector("Spp50", 0.25),
        #         "color": (255, 255, 0),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
        #     "Spp50_ConfTresh-0.75": {
        #         "detector": yolo3_detector.Detector("Spp50", 0.75),
        #         "color": (255, 255, 0),
        #         "predictions": [],
        #         "iou_arr": [],
        #         "types": [0, 0, 0]
        #     },
            # "yolo3_squares": {
            #     "detector": yolo3_detector.Detector("Yolo3_Squares"),
            #     "color": (0, 0, 255),
            #     "predictions": [],
            #     "iou_arr": [],
            #     "types": [0, 0, 0]
            # },
            # "yolo3heu": {
            #     "detector": yolo3_heu_detector.Detector(),
            #     "color": (0, 255, 255),
            #     "predictions": [],
            #     "iou_arr": [],
            #     "types": [0, 0, 0]
            # },
            # "yolo3_c0.25": {
            #     "detector": yolo3_detector.Detector("Yolo3_c0.25"),
            #     "color": (0, 0, 255),
            #     "predictions": [],
            #     "iou_arr": [],
            #     "types": [0, 0, 0]
            # },
            # "yolo3_c0.75": {
            #     "detector": yolo3_detector.Detector("Yolo3_c0.75"),
            #     "color": (0, 255, 255),
            #     "predictions": [],
            #     "iou_arr": [],
            #     "types": [0, 0, 0]
            # },
        # }

        # for scale in [x / 100.0 for x in range(105, 130, 5)]:
        #     for min_neighbors in range(0, 3):
        #         detectors["cascade_sc%.2f_minn%d" % (scale, min_neighbors)] = {
        #             "detector": cascade_detector.Detector(scale, min_neighbors),
        #             "color": (0, 255, 0),
        #             "predictions": [],
        #             "iou_arr": [],
        #             "types": [0, 0, 0]
        #         }

        for i in tqdm(range(len(im_list)), desc="Evaluating..."):
        # for i in range(len(im_list)):
        #     print(i)
            if 0 < LIMIT < i:
                break
            im_name = im_list[i]
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            # img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse
            # img = preprocess.grayscale(img)
            # img = preprocess.automatic_brightness_and_contrast(img)
            img = preprocess.increase_brightness(img)

            saved_img_list = []

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = np.array(self.get_annotations(annot_name))
            #
            if SAVE:
                saved_img = saveImg(img, annot_list, self.output_path, "annotated", Path(os.path.basename(
                    im_name)).stem)
                saved_img_list.append(saved_img)
            # continue

            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            for name, detector in detectors.items():
                prediction_list = detector["detector"].detect(img, im_name)
                # print(name + " " + str(prediction_list))
                if SAVE:
                    saved_img = saveImg(img, prediction_list, self.output_path, name, Path(os.path.basename(
                        im_name)).stem, color=detector["color"])
                    saved_img_list.append(saved_img)
                p, gt = eval.prepare_for_detection(prediction_list, annot_list)
                iou, class_type = eval.iou_compute(p, gt, 0.5)
                detector["iou_arr"].append(iou)
                detector["types"][class_type] += 1

            if SAVE:
                saveAll(saved_img_list, self.output_path, "combined", Path(os.path.basename(im_name)).stem,
                        ["annotated"] + [label for label in detectors.keys()],
                        [(255, 0, 0)] + [detector["color"] for detector in detectors.values()],
                        max_columns=MAXCOLUMNS)

        if SAVE:
            with open(os.path.join(self.output_path, "results.csv"), 'a', newline='') as csvfile:
                csvfile.write("SEP=,\n")
                csvwriter = csv.writer(csvfile)
                columns = ['Algorithm', 'Average IOU', 'TP', 'FP', 'FN', 'Precision', 'Recall']
                csvwriter.writerow(columns)

        for name, detector in detectors.items():
            miou = np.average(detector["iou_arr"])
            print("(" + name + ") Average IOU: ", f"{miou:.2%}")
            print("(" + name + ") TP/FP/FN: " + str(detector["types"]))
            precision, recall = eval.compute_precision_recall(detector["types"])
            print("(" + name + ") Precision: %.4f" % precision)
            print("(" + name + ") Recall: %.4f" % recall)
            if SAVE:
                with open(os.path.join(self.output_path, "results.csv"), 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    row = [name, "%.2f" % miou, detector["types"][0], detector["types"][1], detector["types"][2],
                           "%.4f" % precision, "%.4f" % recall]
                    csvwriter.writerow(row)




if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
