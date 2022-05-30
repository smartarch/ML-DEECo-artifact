import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

from matplotlib.lines import Line2D

font = {'size': 12}

matplotlib.rc('font', **font)


def createLogPlot(records, averageRecords, filename, title, size, show=False, baseline100=None, figsize=None):
    subtitles = [
        'Survived Drones',
        'Damage Rate'
    ]
    colors = [
        'tab:blue',
        'tab:orange',
        'tab:olive',
    ]
    allX = np.arange(1, (size[0] * size[1]) + 1)
    allXLabels = [int(record[-1]) for record in records]

    avX = np.concatenate([
        np.array([1]),
        np.linspace(size[0] + 0.5, ((size[1] - 1) * size[0]) + 0.5, size[1] - 1)
    ])
    avXLabels = ['Baseline'] + [f"Training {i + 1}" for i in range(size[1] - 1)]

    mainY = [
        [record[0] for record in records],
        [record[3] for record in records]
    ]

    averageY = []
    averageSurvived = []
    averageDamage = []
    for record in averageRecords:
        for t in range(size[0]):
            averageSurvived.append(record[0])
            averageDamage.append(record[3])
    averageY.append(averageSurvived)
    averageY.append(averageDamage)

    if baseline100:
        baseline100main = [
            [record[0] for record in baseline100[0]],
            [record[3] for record in baseline100[0]]
        ]
        baseline100avg = [
            [record[0] for record in baseline100[1]] * 2,
            [record[3] for record in baseline100[1]] * 2,
        ]

    if not figsize:
        figsize = (size[1] + 4, size[1] + 4)
    fig, axs = plt.subplots(2, 1, figsize=figsize)

    for i in range(2):
        legend = []
        twin = axs[i].twiny()
        if size[1] > 1:
            legend.append(axs[i].plot(allX, mainY[i], color=colors[0], label="ML-based", marker="o", linestyle="None"))
            twin.plot(allX, averageY[i], color=colors[0], label="ML-based – Average", linestyle="dashed")
            yLines = np.linspace(size[0] + 0.5, ((size[1] - 1) * size[0]) + 0.5, size[1] - 1)
            axs[i].vlines(x=yLines, colors='black', ymin=0, ymax=max(mainY[i]), linestyle='dotted')

        legend.append(axs[i].plot(allX[:size[0]], mainY[i][:size[0]], color=colors[1], label="Baseline", marker="o", linestyle="None"))
        twin.plot(allX[[0, -1]], averageY[i][:2], color=colors[1], label="Baseline – Average", linestyle="dashed")

        if baseline100:
            legend.append(axs[i].plot(allX[:size[0]], baseline100main[i], color=colors[2], label="Baseline 100", marker="o", linestyle="None"))
            twin.plot(allX[[0, -1]], baseline100avg[i], color=colors[2], label="Baseline 100 – Average", linestyle="dashed")

        axs[i].set_xticks(allX, labels=allXLabels)
        axs[i].set_xlabel("Runs")
        twin.set_xticks(avX, labels=avXLabels)
        twin.set_xlabel("Trainings")

        axs[i].set_ylabel(subtitles[i])
        # axs[i].legend(loc='lower right')
        # twin.legend(loc='lower left')
        legend.append([Line2D([0], [0], color="lightgrey", label="Average", linestyle="dashed")])
        lines = [line for lines in legend for line in lines]
        labels = [line.get_label() for line in lines]
        twin.legend(lines, labels)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)


def createChargerPlot(logs, filename, title):
    colors = [
        'green',
        'yellowgreen',
        'lightcoral',
        'lightblue',
    ]
    fig, axs = plt.subplots(len(logs), figsize=(10, 12))
    if len(logs) == 1:
        axs = [axs]
    x = np.arange(1, len(logs[0].records) + 1)
    labels = logs[0].header
    for i in range(len(logs)):
        array = np.array(logs[i].records)
        axs[i].bar(x, array[:, 0], color=colors[0], label=labels[0], width=1)
        bottom = array[:, 0]
        for j in range(1, array.shape[1]):
            axs[i].bar(x, array[:, j], color=colors[j], label=labels[j], bottom=bottom, width=1)
            bottom = bottom + array[:, j]
        # axs[i].stackplot(x, array.T, labels=labels, colors=colors)  # alternative
        axs[i].legend()
        axs[i].set_title(f"Charger {i + 1}")
        axs[i].set_xlabel("Time Steps")
        axs[i].set_ylabel("Drones")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.savefig(filename + ".png", dpi=600)
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('--baseline', '-b', action='store_true', default=False)
    parser.add_argument('--show', '-s', action='store_true', default=False)
    args = parser.parse_args()

    folder = Path(args.folder)
    for file in folder.glob("*.csv"):
        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            if "average" in file.name:
                average = list(reader)
            else:
                log = list(reader)
                world = file.name.split("_")[0]

    if args.baseline:
        baselineFolder = folder.parent / "baseline_100"
        for file in baselineFolder.glob("*.csv"):
            with open(file, newline="") as csvfile:
                reader = csv.reader(csvfile)
                if "average" in file.name:
                    baselineAverage = list(reader)
                else:
                    baselineLog = list(reader)
        baselineLog = np.array(baselineLog[1:], dtype=np.float32)
        baselineAverage = np.array(baselineAverage[1:], dtype=np.float32)
        baseline = (baselineLog, baselineAverage)
    else:
        baseline = None

    log = np.array(log[1:], dtype=np.float32)
    average = np.array(average[1:], dtype=np.float32)
    size = (int(log[:, -1].max()), int(log[:, -2].max()))
    createLogPlot(log, average, folder / "plot", f"World: {world}\nEstimator: Neural network [256, 256]", size, args.show, baseline, figsize=(12, 9))
