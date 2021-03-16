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


def plot_learning_curves(data_x, data_y, plot_dir, config, title="Learning curve", x_label="Num samples", y_label="Accuracy"):
    '''
    Plot result of AL/ML
    :params data_x: Number of samples/Percentage samples
    :params data_y: Accuracy
    :params plot_dir: Dir for saving plot
    '''
    plt.figure("Learning curves")
    plt.plot(data_x, data_y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    filename = f"{config['DATA_SET']}_{config['NET']}_q{config['NUM_QUERY']}_{config['STRATEGY']}.eps"
    plt.savefig(os.path.join(plot_dir, filename))


if __name__ == "__main__":


    acc = [0.0897, 0.4862, 0.5721, 0.5959, 0.6357, 0.6488, 0.6544, 0.6644, 0.672, 0.6504, 0.6749, 0.7043, 0.6997, 0.6965, 0.6885, 0.6988, 0.7106, 0.7361, 0.6704, 0.6938, 0.713, 0.7076, 0.7098, 0.6721, 0.7202, 0.7142, 0.7266, 0.7057, 0.7265, 0.7284, 0.7285, 0.6673, 0.7327, 0.7351, 0.7343, 0.7322, 0.7317, 0.7403, 0.7431, 0.7384, 0.7442, 0.7371, 0.7415, 0.7418, 0.7424, 0.7316, 0.742, 0.7342, 0.7413, 0.7324, 0.7157]
    num_samples = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000]

    acc = [0.0922, 0.4909, 0.5943, 0.6157, 0.6101, 0.6518, 0.6453, 0.6637, 0.6817, 0.6739, 0.6541]
    num_samples = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    PLOT_DIR = "/Users/martin.lund.haug/Documents/Masteroppgave/core-set/plots"

    plot_learning_curves(num_samples, acc, PLOT_DIR, "cifar-coreset.png")

    plot_learning_curves(num_samples, acc, "/Users/martin.lund.haug/Documents/Masteroppgave/plot/", "fraction_cifar_round.png")


    
