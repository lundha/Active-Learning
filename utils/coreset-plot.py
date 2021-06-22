import matplotlib.pyplot as plt
import os
import argparse
import pickle
from copy import deepcopy
import numpy as np
import sys
import re

def plot():
    ailaron_coreset_w_opt = [0.4907000000000001, 0.6662333333333333, 0.7303000000000001, 0.7893666666666667, 0.8269333333333333, 0.8483999999999999, 0.8711000000000001, 0.8798, 0.886, 0.8872333333333334, 0.9037333333333333, 0.8955000000000001, 0.9062333333333333, 0.9083, 0.9112, 0.9083]
    ailaron_coreset = 


    f, ax = plt.subplots(1)
    labels = ['Core-set with optimization', 'Core-set without optimization', 'K-means++', 'K-means']
    colors = ['tab:green', 'tab:red', 'tab:blue', 'tab:olive']
    linestyle = ['solid', 'solid', 'solid', 'solid']
    fraction = [100*elem/n_pool for elem in fraction]
    for accuracy, label, color, linestyle in zip(times, labels, colors, linestyle):
        ax.plot(fraction, accuracy, label=label, color=color, linestyle=linestyle)

    ax.set_xlim(5, 90)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Number labeled samples[%]")
    ax.set_ylabel("Computation time[min]")
    ax.set_facecolor('aliceblue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.legend(loc=4, framealpha=0)
    plt.xticks(ticks=fraction)
    plt.grid(color='white')

    plt.show() 