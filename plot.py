import matplotlib.pyplot as plt
import os
import argparse
import pickle
from copy import deepcopy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--load-dir", default="", help="Load directory")
parser.add_argument("--metric", default="acc", help="Performance metric")
parser.add_argument("--eval_at", default="num_samples", help="X-axis, num_evals, num_samples etc.")
parser.add_argument("--format", default="eps", help="File format, e.g png, pdf")


def plot_learning_curves(data_x, data_y, config, STRATEGY, SEED, title="Learning curve", x_label="Num samples", y_label="Accuracy"):
    '''
    Plot result of AL/ML,
    param data_x: Number of samples/Percentage samples,
    param data_y: Accuracy,
    param plot_dir: Dir for saving plot,
    '''
    plt.figure("Learning curves")
    plt.plot(data_x, data_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    filename = f"{config['DATA_SET']}_{config['NET']}_q{config['NUM_QUERY']}_{STRATEGY}_{SEED}.eps"
    plt.savefig(os.path.join(config['PLOT_DIR'], filename))


if __name__ == "__main__":
    pass