import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
import argparse

from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from utils.utils import load_data_pool, print_image, sub_sample_dataset, load_data
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from query_strategies import Coreset, Random_Strategy, Uncertainty_Strategy, Max_Entropy_Strategy, Bayesian_Sparse_Set_Strategy, \
                            Strategy, DFAL, BUDAL
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from activelearningdataset import ActiveLearningDataset
from get_net import get_net
from download_dataset import get_dataset
from config import update_config, load_config
from plot import plot_learning_curves
from tsne_compare_strategies import plot_tsne
from copy import deepcopy
from keras.datasets import cifar10


config = load_config()

DATA_DIR = config['DATA_DIR']
PLOT_DIR = config['PLOT_DIR']
HEADER_FILE = DATA_DIR + "header.tfl.txt"
FILENAME = DATA_DIR + "image_set.data"

DATA_SET = config['DATA_SET']
NET = config['NET']
STRATEGY = config['STRATEGY']
NUM_INIT_LABELED = config['NUM_INIT_LABELED']
NUM_QUERY = config['NUM_QUERY']
BUDGET = config['BUDGET']
NUM_WORKERS = config['NUM_WORKERS']
FRACTION = config['FRACTION']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', default=STRATEGY, type=str, help='Choose different strategy than specified in config')
args = parser.parse_args()

STRATEGY = args.strategy

load_data_args = {'CIFAR10':
            {
                'data_dir': "../datasets/cifar10/",
                'num_classes': 10,
                'file_ending': ".png",
                'num_channels': 3,
                'device': DEVICE
            },
            'PLANKTON':
            {
                'data_dir': "",
                'num_classes': 0,
                'file_ending': ".",
                'num_channels': 3,
                'device': DEVICE
            }
        }

learning_args = {'CIFAR10': 
        {
            'n_epoch': 10, 
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), 
            'loader_tr_args': {'batch_size': 64, 'num_workers': NUM_WORKERS},
            'loader_te_args': {'batch_size': 1000, 'num_workers': NUM_WORKERS},
        },
        'PLANKTON':
        {
            'n_epoch': 10, 
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), 
            'loader_tr_args': {'batch_size': 64, 'num_workers': NUM_WORKERS},
            'loader_te_args': {'batch_size': 1000, 'num_workers': NUM_WORKERS},         
        }

    }


data_args = load_data_args[DATA_SET]
args = learning_args[DATA_SET]


X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_SET, Fraction=FRACTION)
(X_tr_keras, Y_tr_keras), (X_te_keras, Y_te_keras) = cifar10.load_data()

Y_tr_keras, Y_te_keras = torch.from_numpy(np.array(Y_tr_keras)), torch.from_numpy(np.array(Y_te_keras))
Y_tr_keras, Y_te_keras = torch.squeeze(Y_tr_keras), torch.squeeze(Y_tr_keras)

X_te_tsne, Y_te_tsne = deepcopy(X_te), deepcopy(Y_te)

# Generate initially labeled pool 
ALD = ActiveLearningDataset(X_tr, Y_tr, NUM_INIT_LABELED)

# Load network 
net = get_net(NET, data_args)

list_queried_idxs = []

if STRATEGY == 'coreset':
    strategy = Coreset(ALD, net, args)
elif STRATEGY == 'uncertainty':
    strategy = Uncertainty_Strategy(ALD, net, args)
elif STRATEGY == 'max_entropy':
    strategy = Max_Entropy_Strategy(ALD, net, args)
elif STRATEGY == 'bayesian_sparse_set':
    strategy = Bayesian_Sparse_Set_Strategy(ALD, net, args)
elif STRATEGY == 'DFAL':
    strategy = DFAL(ALD, net, args)
elif STRATEGY == 'BUDAL':
    strategy = BUDAL(ALD, net, args)
else:
    strategy = Random_Strategy(ALD, net, args)

# Number of unlabeled samples (pool)
n_pool = len(ALD.index['unlabeled'])
print(type(strategy).__name__)
print(f"Number of training samples: {n_pool}, Number initially labeled: {len(ALD.index['labeled'])}, Number of testing samples: {len(Y_te)}")


# Round 0 accuracy
rnd = 0
print(f"Round: {rnd}")

strategy.train()
P = strategy.predict(X_te, Y_te)

acc = []
num_labeled_samples = []
num_labeled_samples.append(len(ALD.index['labeled']))
acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))

print(f"Testing accuracy {acc[rnd]}")

'''
print(f"Query indexes and plotting")
queried_idxs = strategy.query(NUM_QUERY, n_pool)
queried_idxs = np.asarray(queried_idxs)
list_queried_idxs.append(queried_idxs)
print(f"Num queried indexes: {len(queried_idxs)}")
ALD.move_from_unlabeled_to_labeled(queried_idxs)
'''
init_tic = datetime.now()

while len(ALD.index['labeled']) < BUDGET + NUM_INIT_LABELED:

    tic = datetime.now()

    rnd += 1
    n_pool = len(ALD.index['unlabeled'])

    NUM_QUERY = min(NUM_QUERY, NUM_INIT_LABELED + BUDGET - len(ALD.index['labeled']))

    queried_idxs = strategy.query(NUM_QUERY, n_pool)
    #print(f"Queried idxs: {queried_idxs}")
    #print(f"Num queried indexes: {len(queried_idxs)}")
    #ALD.on_value_move_from_unlabeled_to_labeled(queried_idxs)
    ALD.move_from_unlabeled_to_labeled(queried_idxs)
    #print(f"Labeled indexes: {ALD.index['labeled']}")
    #print(f"Unlabeled indexes: {ALD.index['unlabeled']}")
    strategy.train()
    P = strategy.predict(X_te, Y_te)
    acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))
    num_labeled_samples.append(len(ALD.index['labeled']))

    print(f"Round: {rnd}, Testing accuracy: {acc[rnd]}, Samples labeled: {num_labeled_samples[rnd]}, Pool size: {len(ALD.index['unlabeled'])}, Iteration time: {datetime.now()-tic}")

    
print(acc)
print(num_labeled_samples)
print(type(strategy).__name__)
print(f"Total run time: {datetime.now() - init_tic}")
if len(acc) == len(num_labeled_samples):
    plot_learning_curves(num_labeled_samples, acc, config, STRATEGY)
else:
    print("Acc is not same length as num labeled samples")
