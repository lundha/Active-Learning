import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image

def get_dataset(name):
    if name == 'CIFAR10':
        return get_CIFAR10()


def get_CIFAR10():
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = data_tr.targets
    X_te = data_te.data
    Y_te = data_te.targets
    return X_tr, Y_tr, X_te, Y_te