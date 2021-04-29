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
                            Strategy, DFAL, BUDAL, Softmax_Hybrid_Strategy, BadgeSampling, LearningLoss, MarginSampling, KMeansSampling, \
                            ActiveLearningByLearning
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from activelearningdataset import ActiveLearningDataset
from get_net import get_net
from get_dataset import get_dataset
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
CUDA_N = config['CUDA_N']
DEVICE = torch.device(f"cuda:{CUDA_N}" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', default=STRATEGY, type=str, help='Choose different strategy than specified in config')
args = parser.parse_args()
config['STRATEGY'] = STRATEGY = args.strategy

TRIALS = 3

acc_all = []

for TRIAL in range(TRIALS):

    SEED = randint(1,1000)

    LOG_FILE = f"./V3_LOGS/{DATA_SET}_{STRATEGY}_q{NUM_QUERY}_f{FRACTION}.log"

    ##### LOGGING #####
    fh = open(LOG_FILE, 'a+')
    fh.write('\n \t\t ***** NEW AL SESSION ***** \n')
    fh.close()
    ###################

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
                                transforms.RandomVerticalFlip(),
                                transforms.RandomAffine(degrees=7,translate=(0.1,0.1),fillcolor=255),
                                transforms.RandomCrop(size=32,padding=4),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), 
                'tr_args': {'batch_size': 64, 'lr': 0.001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
                'valid_args': {'batch_size': 2000, 'num_workers': NUM_WORKERS},
                'te_args': {'batch_size': 1000, 'lr': 0.001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
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
                'tr_args': {'batch_size': 32, 'lr': 0.004, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
                'valid_args': {'batch_size': 379, 'num_workers': NUM_WORKERS},
                'te_args': {'batch_size': 1000, 'lr': 0.004, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},         
            }

        }


    load_data_args = load_data_args[DATA_SET]
    learning_args = learning_args[DATA_SET]


    X_tr, Y_tr, X_te, Y_te, X_valid, Y_valid = get_dataset(DATA_SET, load_data_args, Fraction=FRACTION)
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
    elif STRATEGY == 'softmax_hybrid':
        strategy = Softmax_Hybrid_Strategy(ALD, net, learning_args)
    elif STRATEGY == 'BADGE':
        strategy = BadgeSampling(ALD, net, learning_args)
    elif STRATEGY == 'learning_loss':
        strategy = LearningLoss(ALD, net, learning_args)
    elif STRATEGY == 'all':
        strategy_list = [DFAL(ALD, net, learning_args),
                        Coreset(ALD, net, learning_args)]
        strategy = ActiveLearningByLearning(ALD, net, learning_args, strategy_list)
    else: 
        sys.exit("A valid strategy is not specified, terminating execution..")

    # Number of unlabeled samples (pool)
    n_pool = len(ALD.index['unlabeled'])
    print(type(strategy).__name__)
    print(f"Number of training samples: {n_pool}, Number initially labeled: {len(ALD.index['labeled'])}, Number of testing samples: {len(Y_te)}")


    ##### LOGGING #####
    fh = open(LOG_FILE, 'a+')
    fh.write(f'\n \t\t ***** Start trial {TRIAL+1}/{TRIALS} ***** \n')
    fh.write(f'*** INFO *** Starting time: {datetime.now()}\n')
    fh.write(f'Strategy: {type(strategy).__name__}, Dataset: {DATA_SET}, Number of training samples: {n_pool}, Number initially labeled samples: {len(ALD.index["labeled"])}\n'
            f'Number of testing samples: {len(Y_te)}, Learning network: {NET}, Num query: {NUM_QUERY}, Budget: {BUDGET}, SEED: {SEED}\n')
    fh.write(f'Num epochs: {learning_args["n_epoch"]}, Training batch size: {learning_args["tr_args"]["batch_size"]}, Testing batch size: {learning_args["te_args"]["batch_size"]}\n'
            f'Learning rate: {learning_args["tr_args"]["lr"]}, Weight decay: {learning_args["te_args"]["weight_decay"]}\n')
    fh.write('--'*10)
    fh.close()

    # Round 0 accuracy
    init_tic = datetime.now()
    rnd = 0
    print(f"Round: {rnd}")
    #strategy.train()
    P = strategy.predict(X_te, Y_te)

    acc = []
    num_labeled_samples = []
    num_labeled_samples.append(len(ALD.index['labeled']))
    acc.append(round(1.0 * (Y_te==P).sum().item() / len(Y_te),4))

    print(f"Testing accuracy {acc[rnd]}")

    ##### LOGGING #####
    fh = open(LOG_FILE, 'a+')
    fh.write(f'\nRound: {rnd}, Testing accuracy: {acc[rnd]}, Number of labeled samples: {len(ALD.index["labeled"])}/{n_pool}, Iteration time: {datetime.now() - init_tic}\n')
    fh.close()
    ###################

    while len(ALD.index['labeled']) < BUDGET + NUM_INIT_LABELED:

        tic = datetime.now()

        rnd += 1
        NUM_QUERY = min(NUM_QUERY, NUM_INIT_LABELED + BUDGET - len(ALD.index['labeled']), len(ALD.index['unlabeled']))

        queried_idxs = strategy.query(NUM_QUERY)

        ### TEST CLASS COUNTING ###
        class_count = ALD.count_class_query(queried_idxs)
        class_count_prct = [round(elem/sum(class_count),2) for elem in class_count]

        ALD.move_from_unlabeled_to_labeled(queried_idxs, strategy)

        strategy.train(X_valid, Y_valid)
        P = strategy.predict(X_te, Y_te)
        acc.append(round(1.0 * (Y_te==P).sum().item() / len(Y_te),4))
        num_labeled_samples.append(round(len(ALD.index['labeled']),4))

        print(f"Round: {rnd}, Testing accuracy: {acc[rnd]}, Samples labeled: {num_labeled_samples[rnd]}, Pool size: {len(ALD.index['unlabeled'])}, Iteration time: {datetime.now()-tic}\n")

        ##### LOGGING #####
        fh = open(LOG_FILE, 'a+')
        fh.write(f'Round: {rnd}, Testing accuracy: {acc[rnd]}, Number of labeled samples: {num_labeled_samples[rnd]}/{n_pool}, Iteration time: {datetime.now() - tic}\n'
                f'Queried class distribution: {class_count_prct}\n\n')
        fh.close()
        ###################
        

    print(f"Acc: {acc}, Num labeled samples: {num_labeled_samples}, Strategy: {type(strategy).__name__}, Total run time: {datetime.now() - init_tic}")

    ##### LOGGING #####
    fh = open(LOG_FILE, 'a+')
    fh.write(f'\n \t\t **** FINISHED RUNNING trial {TRIAL+1}/{TRIALS+1} **** \n')
    fh.write(f'Testing accuracy: {acc}, Number of labeled samples: {num_labeled_samples}, Strategy: {type(strategy).__name__}, Dataset: {DATA_SET}, Total iteration time: {datetime.now() - init_tic}\n')
    fh.close()
    ###################

    acc_all.append(acc)

acc_all = np.asarray(acc_all)
mean_acc = np.mean(acc_all, axis=0)
std_acc = np.std(acc_all, axis=0)
var_acc = np.var(acc_all, axis=0)

##### LOGGING #####
fh = open(LOG_FILE, 'a+')
fh.write(f'\n \t\t **** FINISHED RUNNING **** \n')
fh.write(f'Mean accuracy: {mean_acc}, Std accuracy: {std_acc}, Var accuracy: {var_acc} Strategy: {type(strategy).__name__}, Dataset: {DATA_SET}, Total iteration time: {datetime.now() - init_tic}\n'
        f'All results: {acc_all}')
fh.close()
###################
