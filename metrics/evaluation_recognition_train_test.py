import math
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:

    def compute_rank1(self, Y, y, x):
        classes = np.unique(sorted(y))
        count_all = 0
        count_correct = 0
        for cla1 in classes:
            idx1 = y == cla1
            idx2 = x == cla1
            # if (list(idx1).count(True)) <= 1:
            #     continue
            # Compute only for cases where there is more than one sample:
            Y1 = Y[idx1 == True, :]
            Y1[Y1 == 0] = math.inf
            for y1 in Y1:
                s = np.argsort(y1)
                smin = s[0]
                imin = idx2[smin]
                count_all += 1
                if imin:
                    count_correct += 1
        return count_correct / count_all * 100

    # Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

    def compute_rankX(self, Y, y, x, rank):
        # First loop over classes in order to select the closest for each class.
        classes = np.unique(sorted(y))
        count_all = 0
        count_correct = 0
        for i, cla1 in enumerate(classes):
            idx1 = y == cla1
            # if (list(idx1).count(True)) <= 1:
            #     continue
            Y1 = Y[idx1 == True, :]
            Y1[Y1 == 0] = math.inf
            for y1 in Y1:
                class_vector = np.ndarray(len(classes))
                for j, cla2 in enumerate(classes):
                    # Select the closest that is higher than zero:
                    idx2 = x == cla2
                    if (list(idx2).count(True)) <= 1:
                        class_vector[j] = math.inf
                        continue
                    y2 = y1[idx2 == True]
                    y2[y2 == 0] = math.inf
                    class_vector[j] = np.min(y2)
                s = np.argsort(class_vector)
                count_all += 1
                if i in s[0:rank]:
                    count_correct += 1
        return count_correct / count_all * 100

    def CMC_plot(self, Y, y, x, show=True):
        # First loop over classes in order to select the closest for each class.
        classes = np.unique(sorted(y))
        count_correct = np.zeros(len(classes), dtype=int)
        count_all = 0
        for i, cla1 in enumerate(classes):
            idx1 = y == cla1
            samples = list(idx1).count(True)
            if samples <= 1:
                continue
            count_all += samples
            Y1 = Y[idx1 == True, :]
            Y1[Y1 == 0] = math.inf
            for y1 in Y1:
                class_vector = np.ndarray(len(classes))
                for j, cla2 in enumerate(classes):
                    # Select the closest that is higher than zero:
                    idx2 = x == cla2
                    if (list(idx2).count(True)) <= 1:
                        class_vector[j] = math.inf
                        continue
                    y2 = y1[idx2 == True]
                    y2[y2 == 0] = math.inf
                    class_vector[j] = np.min(y2)
                s = np.argsort(class_vector)
                found = False
                for rank, index in enumerate(s):
                    if i == index:
                        found = True
                    if found:
                        count_correct[rank] += 1

        x = range(1, len(classes) + 1)
        y = np.zeros(len(classes), dtype=float)
        for i, count in enumerate(count_correct):
            y[i] = count / count_all * 100

        if show:
            plt.plot(x, y)
            plt.xlabel('Rank')
            plt.ylabel('Recognition rate')
            plt.show()

        return x, y
