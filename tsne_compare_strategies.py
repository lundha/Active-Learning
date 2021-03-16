import click
import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
import argparse

from tsne import tsne_model, tsne_feature_extractor, plot_tsne_categories
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from utils.utils import load_data_pool, print_image, sub_sample_dataset, load_data
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from query_strategies import Coreset, Random_Strategy, Uncertainty_Strategy, Max_Entropy_Strategy, Bayesian_Sparse_Set_Strategy, \
                            Strategy
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from activelearningdataset import ActiveLearningDataset
from get_net import get_net
from download_dataset import get_dataset
from config import update_config, load_config
from plot import plot_learning_curves
from copy import deepcopy
from keras.datasets import cifar10


def compare_tsne():

    config = load_config()

    DATA_DIR = config['DATA_DIR']
    PLOT_DIR = config['PLOT_DIR']
    HEADER_FILE = DATA_DIR + "header.tfl.txt"
    FILENAME = DATA_DIR + "image_set.data"

    DATA_SET = config['DATA_SET']
    NET = config['NET']
    STRATEGY = config['STRATEGY']
    NUM_QUERY = config['NUM_QUERY']
    NUM_WORKERS = config['NUM_WORKERS']
    NUM_INIT_LABELED = config['NUM_INIT_LABELED']
    STRATEGIES = ['bayesian_sparse_set', 'coreset', 'uncertainty', 'max_entropy']


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

    tsne_args = {'dataset': 'CIFAR10',
                'strategy': ['bayesian_sparse_set', 'coreset', 'uncertainty', 'max_entropy']
            }

    data_args = load_data_args[DATA_SET]
    args = learning_args[DATA_SET]


    (X_tr, Y_tr), (X_te, Y_te) = cifar10.load_data()
    Y_tr, Y_te = np.asarray(Y_tr), np.asarray(Y_te)


    X_te_tsne, Y_te_tsne = deepcopy(X_te), deepcopy(Y_te)

    # Generate initially labeled pool 
    ALD = ActiveLearningDataset(X_tr, Y_tr, NUM_INIT_LABELED)

    # Load network 
    net = get_net(NET, data_args)

    list_queried_idxs = []
    tic = datetime.now()

    for STRATEGY in STRATEGIES:

        if STRATEGY == 'coreset':
            strategy = Coreset(ALD, net, args)
        elif STRATEGY == 'uncertainty':
            strategy = Uncertainty_Strategy(ALD, net, args)
        elif STRATEGY == 'max_entropy':
            strategy = Max_Entropy_Strategy(ALD, net, args)
        elif STRATEGY == 'bayesian_sparse_set':
            strategy = Bayesian_Sparse_Set_Strategy(ALD, net, args)
        else:
            strategy = Random_Strategy(ALD, net, args)

        # Number of unlabeled samples (pool)
        n_pool = len(ALD.index['unlabeled'])
        print(type(strategy).__name__)

        print(f"Query indexes and plotting")
        queried_idxs = strategy.query(NUM_QUERY, n_pool)
        queried_idxs = np.asarray(queried_idxs)
        list_queried_idxs.append(queried_idxs)
        print(f"Num queried indexes: {len(queried_idxs)}")

        num_classes = 10
        plot_tsne(X_te_tsne, Y_te_tsne, list_queried_idxs, num_classes, tsne_args)
        print(f"Total run time: {datetime.now()-tic}")



def plot_tsne(x: list, y: list, queried_idxs: list, num_classes: int, tsne_args: dict) -> None:
    '''
    Create T-SNE plot based on data pool. Highlight queried data points with black color
    '''
    weight_path = '/Users/martin.lund.haug/Documents/Masteroppgave/core-set/tsne/v5-weights.48-0.4228.hdf5'
    out_dir = 'v5-features'
    x = x.astype('float32')
    x /= 255
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = tnse_model(x=x, num_classes=num_classes, weight_path=weight_path)
    tx, ty = tsne_feature_extractor(model, x, out_dir)
    plot_tsne_categories(x, y, tx, ty, queried_idxs, out_dir, tsne_args)



if __name__ == "__main__":
    '''
    # This time we will only use the test set:
    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    queried_idxs = [i for i in range(10)]
    # Convert class vectors to binary class matrices.
    num_classes = 10
    
    weight_path = '/Users/martin.lund.haug/Documents/Masteroppgave/core-set/tsne/v5-weights.48-0.4228.hdf5'
    out_dir = 'v5-features'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = tsne_model(x=x_test, num_classes=num_classes, weight_path=weight_path)
    tx, ty = tsne_feature_extractor(model, x_test, out_dir)
    
    plot_tsne_categories(x_test, y_test, tx, ty, queried_idxs, out_dir)
    '''
    compare_tsne()