import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from random import randint
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
SEED = randint(1,100)
STRATEGY = args.strategy

LOG_FILE = f"./LOGS/{DATA_SET}_{STRATEGY}_q{NUM_QUERY}_f{FRACTION}.log"


load_data_args = {'CIFAR10':
            {
                'data_dir': "../datasets/cifar10/",
                'num_classes': 10,
                'file_ending': ".png",
                'num_channels': 3,
                'device': DEVICE
            },
            'PLANKTON10':
            {
                'data_dir': "/home/martlh/masteroppgave/datasets/train10",
                'img_dim': 32, 
                'num_classes': 10,
                'file_ending': ".jpg",
                'num_channels': 3,
                'device': DEVICE
            }
        }

learning_args = {'CIFAR10': 
        {
            'data_set': 'CIFAR10',
            'n_epoch': 10,
            'img_dim': 32,  
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), 
            'loader_tr_args': {'batch_size': 32, 'num_workers': NUM_WORKERS},
            'loader_te_args': {'batch_size': 1000, 'num_workers': NUM_WORKERS},
        },
        'PLANKTON10':
        {
            'data_set': 'PLANKTON10',
            'n_epoch': 10,
            'img_dim': 32,
            'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), 
            'loader_tr_args': {'batch_size': 64, 'num_workers': NUM_WORKERS},
            'loader_te_args': {'batch_size': 1000, 'num_workers': NUM_WORKERS},         
        }

    }


load_data_args = load_data_args[DATA_SET]
learning_args = learning_args[DATA_SET]


X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_SET, load_data_args, Fraction=FRACTION)

print("**"*5+"Data info"+"**"*5)
img_dim = X_tr.shape[1]

print(f"X_tr shape: {X_tr.shape}, X_tr dtype: {X_tr.dtype}, X_tr type: {type(X_tr)} \n \
    X_tr[1] shape: {X_tr[1].shape}, X_tr[1] dtype: {X_tr[1].dtype}, X_tr[1] type: {type(X_tr[1])}\n")

print(f"Y_tr shape: {Y_tr.shape},Y_tr dtype: {Y_tr.dtype}, Y_tr type: {type(Y_tr)} \n \
    Y_tr[1] shape: {Y_tr[1].shape}, Y_tr[1] dtype: {Y_tr[1].dtype}, Y_tr[1] type: {type(Y_tr[1])}\n")

print("**"*10)



# Generate initially labeled pool 
ALD = ActiveLearningDataset(X_tr, Y_tr, NUM_INIT_LABELED)

# Load network 
net = get_net(NET, load_data_args)

list_queried_idxs = []

if STRATEGY == 'coreset':
    strategy = Coreset(ALD, net, learning_args)
elif STRATEGY == 'uncertainty':
    strategy = Uncertainty_Strategy(ALD, net, learning_args)
elif STRATEGY == 'max_entropy':
    strategy = Max_Entropy_Strategy(ALD, net, learning_args)
elif STRATEGY == 'bayesian_sparse_set':
    strategy = Bayesian_Sparse_Set_Strategy(ALD, net, learning_args)
elif STRATEGY == 'DFAL':
    strategy = DFAL(ALD, net, learning_args)
elif STRATEGY == 'BUDAL':
    strategy = BUDAL(ALD, net, learning_args)
elif STRATEGY == 'random':
    strategy = Random_Strategy(ALD, net, learning_args)
else: 
    sys.exit("A valid strategy is not specified, terminating execution..")

# Number of unlabeled samples (pool)
n_pool = len(ALD.index['unlabeled'])
print(type(strategy).__name__)
print(f"Number of training samples: {n_pool}, Number initially labeled: {len(ALD.index['labeled'])}, Number of testing samples: {len(Y_te)}")


##### LOGGING 
fh = open(LOG_FILE, 'a+')
fh.write('\n \t\t ***** NEW AL SESSION ***** \n')
fh.write('*** INFO ***\n')
fh.write(f'Strategy: {type(strategy).__name__}, Dataset: {DATA_SET}, Number of training samples: {n_pool}, Number initially labeled samples: {len(ALD.index["labeled"])}\n'
        f'Number of testing samples: {len(Y_te)}, Learning network: {NET}, Num query: {NUM_QUERY}, Budget: {BUDGET}, SEED: {SEED}\n')
fh.write(f'Num epochs: {learning_args["n_epoch"]}, Training batch size: {learning_args["loader_tr_args"]["batch_size"]}, Testing batch size: {learning_args["loader_te_args"]["batch_size"]}\n')
fh.write('--'*10)
fh.close()

# Round 0 accuracy
init_tic = datetime.now()
rnd = 0
print(f"Round: {rnd}")

strategy.train()
P = strategy.predict(X_te, Y_te)

acc = []
num_labeled_samples = []
num_labeled_samples.append(len(ALD.index['labeled']))
acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))

print(f"Testing accuracy {acc[rnd]}")

##### LOGGING #####
fh = open(LOG_FILE, 'a+')
fh.write(f'\nRound: {rnd}, Testing accuracy: {acc[rnd]}, Number of labeled samples: {len(ALD.index["labeled"])}/{n_pool}, Iteration time: {datetime.now() - init_tic}\n')
fh.close()
###################

while len(ALD.index['labeled']) < BUDGET + NUM_INIT_LABELED:

    tic = datetime.now()

    rnd += 1

    NUM_QUERY = min(NUM_QUERY, NUM_INIT_LABELED + BUDGET - len(ALD.index['labeled']))

    queried_idxs = strategy.query(NUM_QUERY)

    if (type(strategy).__name__ == 'DFAL' or type(strategy).__name__ == 'Random_Strategy' or type(strategy).__name__ == 'Uncertainty_Strategy' \
        or type(strategy).__name__ == 'Max_Entropy_Strategy'):
        ALD.on_value_move_from_unlabeled_to_labeled(queried_idxs)
    else: 
        ALD.move_from_unlabeled_to_labeled(queried_idxs)

    strategy.train()
    P = strategy.predict(X_te, Y_te)
    acc.append(1.0 * (Y_te==P).sum().item() / len(Y_te))
    num_labeled_samples.append(len(ALD.index['labeled']))

    print(f"Round: {rnd}, Testing accuracy: {acc[rnd]}, Samples labeled: {num_labeled_samples[rnd]}, Pool size: {len(ALD.index['unlabeled'])}, Iteration time: {datetime.now()-tic}\n")

    ##### LOGGING #####
    fh = open(LOG_FILE, 'a+')
    fh.write(f'Round: {rnd}, Testing accuracy: {acc[rnd]}, Number of labeled samples: {len(ALD.index["labeled"])}/{n_pool}, Iteration time: {datetime.now() - tic}\n')
    fh.close()
    ###################
    
print(acc)
print(num_labeled_samples)
print(type(strategy).__name__)
print(f"Total run time: {datetime.now() - init_tic}")

##### LOGGING #####
fh = open(LOG_FILE, 'a+')
fh.write(f'\n \t\t **** FINISHED RUNNING **** \n')
fh.write(f'Testing accuracy: {acc}, Number of labeled samples: {num_labeled_samples}, Strategy: {type(strategy).__name__}, Dataset: {DATA_SET}, Total iteration time: {datetime.now() - init_tic}\n')
fh.close()
###################

if len(acc) == len(num_labeled_samples):
    plot_learning_curves(num_labeled_samples, acc, config, STRATEGY, SEED)
else:
    print("Acc is not same length as num labeled samples")

