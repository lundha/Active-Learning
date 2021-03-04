
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from copy import deepcopy
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--load-dir", default="", help="Load directory")
parser.add_argument("--metric", default="ll", help="Performance metric")
parser.add_argument("--eval_at", default="num_samples", help="X-axis, num_evals, num_samples etc.")
parser.add_argument("--format", default="png", help="File format, e.g png, pdf")


def plot_learning_curves(data_x, data_y, plot_dir, filename, title="Learning curve", x_label="Num samples", y_label="Accuracy"):
    '''
    Plot result of AL/ML
    :params: data_x: Number of samples/Percentage samples
    :params: data_y: Accuracy
    :plot_dir: Dir for saving plot
    :filename: Name of plot file
    '''
    plt.figure("Learning curves")
    plt.plot(data_x, data_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.join(plot_dir, filename))


if __name__ == "__main__":

    plot_learning_curves([i for i in range(10)], [i for i in range(10)], "/Users/martin.lund.haug/Documents/Masteroppgave/plot/", "plot3.png")


    


