import click
import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from utils.utils import load_data_pool, print_image, sub_sample_dataset, load_data
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from query_strategies import Coreset, Random_Strategy, Uncertainty_Strategy, Max_Entropy_Strategy, Strategy
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

#X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_SET, Fraction=0.5)
from keras.datasets import cifar10
_, (X_te, Y_te) = cifar10.load_data()
X_te_tsne, Y_te_tsne = deepcopy(X_te), deepcopy(Y_te)

# Generate initially labeled pool 
ALD = ActiveLearningDataset(X_te, Y_te, NUM_INIT_LABELED)
# Load network 
net = get_net(NET, data_args)
# Load strategy
strategy = Coreset(ALD, net, args)
# strategy = Uncertainty_Strategy(ALD, net, args)
# strategy = Random_Strategy(ALD, net, args)
# strategy = Max_Entropy_Strategy(ALD, net, args)
# Number of unlabeled samples (pool)
n_pool = len(ALD.index['unlabeled'])

print(f"Shape of training data: {ALD.index['unlabeled'].shape}, Data type: {ALD.index['unlabeled'].dtype}")
print(f"Number of training samples: {n_pool}, Number of testing samples: {len(Y_te)}")


# Round 0 accuracy
rnd = 0
print(f"Round: {rnd}")
strategy.train()
P = strategy.predict(X_te, Y_te)

acc = []
num_labeled_samples = []
num_labeled_samples.append(len(ALD.index['labeled']))
acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))

print(f"Testing accuracy {acc[rnd]}, Computation time: {datetime.now()-tic}")


print(f"Query indexes and plotting")
queried_idxs = strategy.query(NUM_QUERY, n_pool)
queried_idxs = np.asarray(queried_idxs)
print(f"Num queried indexes: {len(queried_idxs)}")


##### 
print(f"len X: {len(X_te)}, shape X: {X_te.shape}, type X: {type(X_te)}")
print(f"len Y: {len(Y_te)}, shape Y: {Y_te.shape}, type Y: {type(Y_te)}")
print(f"len Q_idxs: {len(queried_idxs)}, shape Y: {queried_idxs.shape}, type Q_idxs: {type(queried_idxs)}")
#####

plot_tsne(X_te_tsne, Y_te_tsne, queried_idxs, num_classes=10)
print(f"Total run time: {datetime.now()-tic}")

'''

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
print(f"Total run time: {datetime.now() - tic}")
if len(acc) == len(num_labeled_samples):
    plot_learning_curves(num_labeled_samples, acc, PLOT_DIR, "cifar-coreset.png")
else:
    print("Acc is not same length as num labeled samples")


'''

