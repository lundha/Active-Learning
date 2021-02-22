
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


# Load data
data_dir = "/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/"
header_file = data_dir + "header.tfl.txt"
filename = data_dir + "image_set.data"
file_ending = ".png"
num_classes = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_SET = 'CIFAR10'
NET = 'CIFAR_NET'
NUM_INIT_LABELED = 0
NUM_QUERY = 100
BUDGET = 500
NUM_WORKERS = 4


load_data_args = {'CIFAR10':
            {'data_dir': "/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/",
            'num_classes': 10,
            'file_ending': ".png",
            'num_channels': 3,
            'device': device
            }
        }

args_dict = {'CIFAR10': 
        {'n_epoch': 10, 
        'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]), 
        'loader_tr_args': {'batch_size': 64, 'num_workers': NUM_WORKERS},
        'loader_te_args': {'batch_size': 1000, 'num_workers': NUM_WORKERS},
        }
    }

args = args_dict[DATA_SET]
data_args = load_data_args[DATA_SET]
'''

train_data = load_data_pool(train=True, arg=data_args)
test_data = load_data_pool(train=False, arg=data_args)

X_tr, Y_tr = load_data(data_args['data_dir'], train = True)
X_te, Y_te = load_data(data_args['data_dir'], train = False)
'''

tic = datetime.now()

X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_SET, Fraction=0.01)


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
acc = np.zeros(10)
acc[rnd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
print(f"Testing accuracy {acc[rnd]}")
print(f"Computation time: {datetime.now()-tic}")



while len(ALD.index['labeled']) < BUDGET + NUM_INIT_LABELED:

    rnd += 1
    print(f"Round: {rnd}")
    print(ALD)
    n_pool = len(ALD.index['unlabeled'])
    print(f"pool size: {n_pool}")

    queried_idxs = strategy.query(NUM_QUERY, n_pool)
    ALD.move_from_unlabeled_to_labeled(queried_idxs)

    strategy.train()
    P = strategy.predict(X_te, Y_te)
    acc[rnd] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
    print(f"Testing accuracy {acc[rnd]}")
    
print(acc)
print(type(strategy).__name__)




