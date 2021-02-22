import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

def get_dataset(name, Fraction):
    if name == 'CIFAR10':
        return get_CIFAR10(Fraction)


def get_CIFAR10(Fraction):
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = data_tr.targets
    X_te = data_te.data
    Y_te = data_te.targets
    Y_tr, Y_te = torch.from_numpy(np.array(Y_tr)), torch.from_numpy(np.array(Y_te))
    X_tr, Y_tr = make_subset(X_tr, Y_tr, Fraction)
    X_te, Y_te = make_subset(X_te, Y_te, Fraction)
    return X_tr, Y_tr, X_te, Y_te


def make_subset(X, Y, Fraction):

    size = int(Fraction*len(X))
    index = {'include': np.arange(0), 'exclude': np.arange(len(X))} 
    shuffled_indices = np.random.permutation(index['exclude'])
    index['include'], index['exclude'] = shuffled_indices[:size], shuffled_indices[size:]
    return X[index['include']], Y[index['include']]


