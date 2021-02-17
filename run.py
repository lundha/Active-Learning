
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
import click
from autoencoder import Autoencoder
from utils import load_data_pool, print_image
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from coreset import Coreset
from utils import load_data
import os 
from activelearningdataset import ActiveLearningDataset
from get_net import get_net


# Load data
data_dir = "/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/"
header_file = data_dir + "header.tfl.txt"
filename = data_dir + "image_set.data"
file_ending = ".png"
num_classes = 10

DATA_SET = 'CIFAR10'
NET = 'resnet18'
NUM_INIT_LABELED = 10000
NUM_QUERY = 1000
BUDGET = 1000

'''

load_data_args = {'CIFAR10':
            {'data_dir': "/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/",
            'num_classes': 10,
            'file_ending': ".png"
            }
        }

args_dict = {'CIFAR10': 
        {'n_epoch': 10, 
        'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), 
        'loader_tr_args': {'batch_size': 32, 'num_workers': 1},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
        }
    }

args = args_dict[DATA_SET]
data_args = load_data_args[DATA_SET]


train_data = load_data_pool(train=True, arg=data_args)
test_data = load_data_pool(train=False, arg=data_args)


X_tr, Y_tr = load_data(data_args['data_dir'], train = True)
X_te, Y_te = load_data(data_args['data_dir'], train = False)
'''

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_SET)

print(len(Y_tr))
print(len(Y_te))

# Generate initially labeled pool 
ALD = ActiveLearningDataset(X_tr, Y_tr, NUM_INIT_LABELED)

# Load network 
net = get_net(NET)

# Load strategy
strategy = Coreset(ALD, net, args)
idxs = strategy.query(NUM_QUERY)
print(len(idxs))

if __name__ == "__main__":
    pass




