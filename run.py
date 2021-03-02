
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
import click
import torch
from autoencoder import Autoencoder
from utils import load_data_pool, print_image, sub_sample_dataset
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
from download_dataset import get_dataset
from config import update_config, load_config
from plot import plot_learning_curves
# Load data

config = load_config()

DATA_DIR = config['DATA_DIR']
PLOT_DIR = config['PLOT_DIR']
HEADER_FILE = DATA_DIR + "header.tfl.txt"
FILENAME = DATA_DIR + "image_set.data"

DATA_SET = config['DATA_SET']
NET = config['NET']
NUM_INIT_LABELED = config['NUM_INIT_LABELED']
NUM_QUERY = config['NUM_QUERY']
BUDGET = config['BUDGET']
NUM_WORKERS = config['NUM_WORKERS']
FRACTION = config['FRACTION']

load_data_args = {'CIFAR10':
            {'data_dir': "/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/",
            'num_classes': 10,
            'file_ending': ".png",
            'num_channels': 3,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            }
        }

learning_args = {'CIFAR10': 
        {'n_epoch': 10, 
        'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), 
        'loader_tr_args': {'batch_size': 64, 'num_workers': NUM_WORKERS},
        'loader_te_args': {'batch_size': 1000, 'num_workers': NUM_WORKERS},
        }
    }

data_args = load_data_args[DATA_SET]
args = learning_args[DATA_SET]


tic = datetime.now()

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_SET, Fraction=FRACTION)


print(f"Number of training samples: {len(Y_tr)}")
print(f"Number of testing samples: {len(Y_te)}")

# Generate initially labeled pool 
ALD = ActiveLearningDataset(X_tr, Y_tr, NUM_INIT_LABELED)
# Load network 
net = get_net(NET, data_args)
# Load strategy
strategy = Coreset(ALD, net, args)

# Round 0 accuracy
rnd = 0
print(f"Round: {rnd}")
strategy.train()
P = strategy.predict(X_te, Y_te)

acc = []
num_labeled_samples = []
num_labeled_samples.append(len(ALD.index['unlabeled']))
acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))

print(f"Testing accuracy {acc[rnd]}")
print(f"Computation time: {datetime.now()-tic}")



while len(ALD.index['labeled']) < BUDGET + NUM_INIT_LABELED:

    rnd += 1
    n_pool = len(ALD.index['unlabeled'])

    
    queried_idxs = strategy.query(NUM_QUERY, n_pool)
    print(f"Num queried indexes: {len(queried_idxs)}")
    ALD.move_from_unlabeled_to_labeled(queried_idxs)


    strategy.train()
    P = strategy.predict(X_te, Y_te)
    acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))
    num_labeled_samples.append(len(ALD.index['labeled']))

    print(f"Round: {rnd}, Testing accuracy: {acc[rnd]}, Samples labeled: {num_labeled_samples[rnd]}, Pool size: {n_pool}")

    
print(acc)
print(num_labeled_samples)
print(type(strategy).__name__)
if len(acc) == len(num_labeled_samples):
    plot_learning_curves(num_labeled_samples, acc, PLOT_DIR, "cifar-coreset.png")
else:
    print("Acc is not same length as num labeled samples")




