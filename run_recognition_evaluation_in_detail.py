from itertools import cycle
import pickle
import cv2
import glob
import os
import json
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from metrics.evaluation_recognition_train_test import Evaluation
from roboflow import Roboflow

import feature_extractors.pix2pix.extractor as p2p_ext
import feature_extractors.lbp.extractor as lbp_ext
import feature_extractors.block_lbp.extractor as block_lbp_ext


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        base_dataset = "my_ears"

        self.results_folder = os.path.join('results', base_dataset)
        if not os.path.isdir(self.results_folder):
            os.makedirs(self.results_folder)

        with open('config_recognition_in_detail.json') as config_file:
            config = json.load(config_file)
        data_loc = config["data_location"]
        config = config[base_dataset]
        project_name = config['project_name']

        self.datasets = config['datasets']
        rf = Roboflow(api_key="vNXadpVSzZfevBW2gHPf")
        project = rf.workspace().project(project_name)

        for dataset in self.datasets:
            path = os.path.join(data_loc, project_name + "-" + str(dataset['version']))
            if os.path.isdir(path):
                dataset["dataset_path"] = path
                continue
            data = project.version(dataset['version']).download("folder", path)
            dataset["dataset_path"] = data.location

        feature_extractors = {
            "pix2pix": {
                "name": "pix2pix",
                "extractor": p2p_ext.Pix2Pix(do_format=True)
            },
            "LBP": {
                "name": "LBP",
                "extractor": lbp_ext.LBP(do_format=False)
            },
            "Bloc LBP": {
                "name": "BlocLBP",
                "extractor": block_lbp_ext.BlockLBP(do_format=False)
            }
        }

        self.feature_extractor = feature_extractors["LBP"]

    def run_evaluation(self):
        plt.figure(figsize=(7.5, 3.75), layout='constrained')
        lines = ["--", "-.", ":"]
        linecycler = cycle(lines)
        for dataset in self.datasets:
            print("\n-->Evaluating dataset", dataset["name"])
            dataset_path = dataset["dataset_path"]
            im_list = sorted(glob.glob(dataset_path + '/train/*/*.jpg', recursive=True))
            im_list_test = sorted(glob.glob(dataset_path + '/test/*/*.jpg', recursive=True))
            eval = Evaluation()
            # Change the following extractors, modify and add your own
            # Pixel-wise comparison:

            feature_extractor = self.feature_extractor["extractor"]

            train_features_arr = []
            y = []

            print("Extracting train features...")
            # for i in tqdm(range(len(im_list)), desc="Evaluating..."):
            for im_name in im_list:
                img = cv2.imread(im_name)
                class_name = int(os.path.basename(os.path.dirname(im_name)))
                y.append(class_name)

                train_features = feature_extractor.extract(img)
                train_features_arr.append(train_features)
            print("Extracting test features...")
            x = []
            test_features_arr = []
            for im_name in im_list_test:
                img = cv2.imread(im_name)
                class_name = int(os.path.basename(os.path.dirname(im_name)))
                x.append(class_name)
                test_features = feature_extractor.extract(img)
                test_features_arr.append(test_features)

            print("Calculating distances...")
            Y_plain = cdist(test_features_arr, train_features_arr, 'jensenshannon')
            print("Calculating measures...")
            r1 = eval.compute_rank1(Y_plain, x, y)
            print(dataset["name"], 'Rank 1[%]', r1)

            # r5 = eval.compute_rankX(Y_plain, y, 5)
            # print('Pix2Pix Rank 5[%]', r5)
            ranks, acc = eval.CMC_plot(Y_plain, x, y, show=False)
            with open(
                    os.path.join(self.results_folder,
                                 self.feature_extractor["name"] + "_v_" + str(dataset["version"]) + '.res'),
                    'wb') as f:
                pickle.dump(acc, f)
            plt.plot(ranks, acc, next(linecycler), label=dataset["name"])

        plt.ylabel('Recognition rate[%]')
        plt.xlabel('Rank')
        plt.title(self.feature_extractor["name"])
        plt.legend()
        plt.savefig(os.path.join(self.results_folder, self.feature_extractor["name"] + "_datasets_eval.png"))
        plt.show()


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
