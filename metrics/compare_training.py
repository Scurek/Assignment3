import csv
import os
import matplotlib.pyplot as plt


def compare_training(entries, base_path):
    fig_dur, ax_dur = plt.subplots()
    fig_epo, ax_epo = plt.subplots()
    for key, entry in entries.items():
        duration = entry["duration"] * 60
        times = []
        mAPs = []
        with open(os.path.join(base_path, key, "results.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvfile)
            for row in csvreader:
                time = ((int(row[0]) + 1) / 50.0) * duration
                times.append(time)
                mAPs.append(float(row[6]))
        ax_dur.plot(times, mAPs, label=key)
        ax_epo.plot(range(1, 51), mAPs, label=key)
    ax_dur.set_xlabel('Čas (min)')
    ax_dur.set_ylabel('mAP')
    ax_dur.legend()
    fig_dur.savefig('output/comp_time.png')

    ax_epo.set_xlabel('Epohi')
    ax_epo.set_ylabel('mAP')
    ax_epo.legend()
    fig_epo.savefig('output/comp_epochs.png')

if __name__ == '__main__':
    entries = {
        # "Tiny300": {
        #     "duration": 2.120
        # },
        "Tiny50": {
            "duration": 0.367
        },
        "Norm50": {
            "duration": 3.167
        },
        "Spp50": {
            "duration": 3.378
        }
    }
    compare_training(entries, "../outside/yolov3/runs/train")