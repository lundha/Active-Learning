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
                            Strategy, DFAL, BUDAL
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from activelearningdataset import ActiveLearningDataset
from get_net import get_net
from get_dataset import get_dataset
from config import update_config, load_config
from plot import plot_learning_curves
from copy import deepcopy
from keras.datasets import cifar10
from random import randint


def compare_tsne():

    config = load_config()

    DATA_DIR = config['DATA_DIR']
    PLOT_DIR = config['PLOT_DIR']


    DATA_SET = config['DATA_SET']
    NET = config['NET']
    STRATEGY = config['STRATEGY']
    NUM_QUERY = 300 
    NUM_WORKERS = config['NUM_WORKERS']
    NUM_INIT_LABELED = 0 
    STRATEGIES = ['coreset', 'DFAL']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    FRACTION = config['FRACTION']
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='DFAL', type=str, help='Choose different strategy for TSNE')
    args = parser.parse_args()
    STRATEGY = args.strategy

    load_data_args = {'CIFAR10':
                {
                    'data_dir': "../datasets/cifar10/",
                    'num_classes': 10,
                    'file_ending': ".png",
                    'num_channels': 3,
                },
                'PLANKTON10':
                {
                    'data_dir': "../datasets/train10",
                    'img_dim': 32, 
                    'num_classes': 10,
                    'file_ending': ".jpg",
                    'num_channels': 3,
                }
            }

    learning_args = {'CIFAR10': 
            {
                'data_set': 'CIFAR10',
                'n_epoch': 20,
                'img_dim': 32,  
                'transform': transforms.Compose(
                        [transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(size=32,padding=4),
                        transforms.ToTensor(), 
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), 
                'tr_args': {'batch_size': 64, 'lr': 0.004, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
                'te_args': {'batch_size': 1000, 'lr': 0.004, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
            },
            'PLANKTON10':
            {
                'data_set': 'PLANKTON10',
                'n_epoch': 10,
                'img_dim': 32,
                'transform': transforms.Compose(
                            [transforms.Grayscale(num_output_channels=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=7,translate=(0.1,0.1),fillcolor=255),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.95,), std=(0.2,))]),  
                'tr_args': {'batch_size': 64, 'lr': 0.004, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
                'te_args': {'batch_size': 1000, 'lr': 0.004, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},         
            }

        }

    tsne_args = {'dataset': DATA_SET,
                'strategy': STRATEGIES
            }

    data_args = load_data_args[DATA_SET]
    args = learning_args[DATA_SET]

    HEADER_FILE = data_args['data_dir'] + "header.tfl.txt"
    FILENAME = data_args['data_dir'] + "image_set.data"
    num_classes = data_args['num_classes']

    #(_, _), (X_te, Y_te) = cifar10.load_data()
    X_tr, Y_tr, X_te, Y_te, _, _ = get_dataset(DATA_SET, data_args, Fraction=FRACTION)

    X_te_tsne, Y_te_tsne = deepcopy(X_tr), deepcopy(Y_tr)

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
        elif STRATEGY == 'DFAL':
            strategy = DFAL(ALD, net, args)
        elif STRATEGY == 'BUDAL':
            strategy = BUDAL(ALD, net, args)
        else:
            strategy = Random_Strategy(ALD, net, args)

        # Number of unlabeled samples (pool)
        print(type(strategy).__name__)
        print(f"Query indexes and plotting")
        queried_idxs = strategy.query(NUM_QUERY)
        queried_idxs = np.asarray(queried_idxs)
        list_queried_idxs.append(queried_idxs)
        print(f"Num queried indexes: {len(queried_idxs)}")
        weight_path = '/home/martlh/masteroppgave/core-set/tsne/plankton-v5-weights.55-0.3445.hdf5'
        seed = randint(1,1000)
        print(seed)
        np.save('./queried_idxs', list_queried_idxs)
        #plot_tsne(X_te_tsne, Y_te_tsne, list_queried_idxs, num_classes, tsne_args, weight_path, data_set='plankton10', seed=seed)    
    
    print(f"Total run time: {datetime.now()-tic}")



def plot_tsne(x: list, y: list, queried_idxs: list, num_classes: int, tsne_args: dict, weight_path: str, data_set: str, seed: int) -> None:
    '''
    Create T-SNE plot based on data pool. Highlight queried data points with black color
    '''

    out_dir = 'tsne_plots'
    x = x.astype('float32')
    x /= 255
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
  
    model = tsne_model(x=x, num_classes=num_classes, weight_path=weight_path)
    tx, ty = tsne_feature_extractor(model, x, out_dir, data_set, new_weights=True)
    plot_tsne_categories(x, y, tx, ty, queried_idxs, out_dir, tsne_args, seed)



if __name__ == "__main__":

    compare_tsne()