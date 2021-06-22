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
from query_strategies import Strategy, Coreset_Strategy, Random_Strategy, Uncertainty_Strategy, Max_Entropy_Strategy, Bayesian_Sparse_Set_Strategy, \
                            DFAL_Strategy, CIRAL_Strategy, Softmax_Hybrid_Strategy, BADGE_Strategy, Learning_Loss_Strategy, Margin_Strategy, \
                            KMeans_Strategy, ActiveLearningByLearning_Strategy, All_Data_Strategy
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from activelearningdataset import ActiveLearningDataset
from get_net import get_net
from sklearn import metrics
from get_dataset import get_dataset
from plot import plot_learning_curves
from visualize_meta_results import plot_tsne
from copy import deepcopy
from config import args, STRATEGY, DATASET, NUM_WORKERS, TRIALS, CUDA_N
import logging


al_args = args[DATASET]['al_args']
learning_args = args[DATASET]['learning_args']
data_args = args[DATASET]['data_args']

NUM_INIT_LABELED = al_args['num_init_labeled']
NUM_QUERY = al_args['num_query']
BUDGET = al_args['budget']
NET = learning_args['net']
FRACTION = data_args['fraction']
DEVICE = torch.device(f"cuda:{CUDA_N}" if torch.cuda.is_available() else "cpu")
LOG_PATH = f"./V16_LOGS"
LOG_FILE = f"{LOG_PATH}/{DATASET}_{STRATEGY}_q{NUM_QUERY}_f{FRACTION}.log"

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


### CONFIGURE LOGGING ###
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
### OUTPUT STREAM
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
### TO FILE
ch2 = logging.FileHandler(filename=LOG_FILE, mode='a+')
ch2.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s - %(message)s')

ch.setFormatter(formatter)
ch2.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(ch2)
########################

acc_all = []

for TRIAL in range(TRIALS):

    SEED = randint(1,1000)

    acc = []
    num_labeled_samples = []
    list_queried_idxs = []
    
    ##### LOGGING #####
    logger.info('\t\t ***** NEW AL SESSION ***** ')
    ###################


    X_tr, Y_tr, X_te, Y_te, X_val, Y_val = get_dataset(DATASET, data_args)

    ##### LOGGING #####
    logger.debug("**"*5+"Data info"+"**"*5)
    img_dim = X_tr.shape[1]
    logger.debug(f"X_tr shape: {X_tr.shape}, X_tr dtype: {X_tr.dtype}, X_tr type: {type(X_tr)} \n \
        X_tr[1] shape: {X_tr[1].shape}, X_tr[1] dtype: {X_tr[1].dtype}, X_tr[1] type: {type(X_tr[1])}\n")

    logger.debug(f"Y_tr shape: {Y_tr.shape},Y_tr dtype: {Y_tr.dtype}, Y_tr type: {type(Y_tr)} \n \
        Y_tr[1] shape: {Y_tr[1].shape}, Y_tr[1] dtype: {Y_tr[1].dtype}, Y_tr[1] type: {type(Y_tr[1])}\n")
    logger.debug(f"X_valid shape: {X_val.shape}")
    logger.debug("**"*10)
    ###################

    # Generate initially labeled pool 
    ALD = ActiveLearningDataset(X_tr, Y_tr, X_te, Y_te, X_val, Y_val, NUM_INIT_LABELED)

    # Load learning network 
    net = get_net(NET, data_args, strategy=STRATEGY)

    # Load active learning strategy
    strategies = {
        'ALL': ActiveLearningByLearning_Strategy,
        'ALL-DATA': All_Data_Strategy,
        'BAYESIAN_SPARSE_SET': Bayesian_Sparse_Set_Strategy,
        'BADGE': BADGE_Strategy,
        'CIRAL': CIRAL_Strategy,
        'CORESET': Coreset_Strategy,
        'DFAL': DFAL_Strategy,
        'KMEANS': KMeans_Strategy,
        'LEARNING_LOSS': Learning_Loss_Strategy,
        'MAX_ENTROPY': Max_Entropy_Strategy,
        'RANDOM': Random_Strategy,
        'SOFTMAX_HYBRID': Softmax_Hybrid_Strategy,
        'UNCERTAINTY': Uncertainty_Strategy,
    }

    kwargs = {"strategy_list" : [DFAL_Strategy(ALD, net, learning_args, logger),
                                Coreset_Strategy(ALD, net, learning_args, logger)], 
            "log_file" : LOG_FILE, 
            "n_epochs" : 100}
    
    strategy = strategies[STRATEGY](ALD, net, learning_args, logger, **kwargs)
    init_pool = len(ALD.index["unlabeled"])

    ##### LOGGING #####
    logger.info(type(strategy).__name__)
    logger.info(f"Number of training samples: {len(ALD.index['unlabeled'])}, Number initially labeled: {len(ALD.index['labeled'])}, Number of testing samples: {len(Y_te)}, "
                f"Number of validation samples: {len(ALD.X_valid)}")
    ###################


    ##### LOGGING #####
    logger.info(f'\t\t ***** Start trial {TRIAL+1}/{TRIALS} *****')
    logger.info(f'Starting time: {datetime.now()}')
    logger.info(f'Strategy: {type(strategy).__name__}, Dataset: {DATASET}, Number of training samples: {len(ALD.index["unlabeled"])}, Number initially labeled samples: {len(ALD.index["labeled"])}\n'
            f'Number of testing samples: {len(Y_te)}, Learning network: {NET}, Num query: {NUM_QUERY}, Budget: {BUDGET}, SEED: {SEED}')
    logger.info(f'Num epochs: {learning_args["n_epoch"]}, Training batch size: {learning_args["tr_args"]["batch_size"]}, Testing batch size: {learning_args["te_args"]["batch_size"]}, '
            f'Learning rate: {learning_args["tr_args"]["lr"]}, Weight decay: {learning_args["te_args"]["weight_decay"]}')
    logger.info('--'*10)
    ###################

    init_tic = datetime.now()
    rnd = 0
    strategy.train()
    
    P = strategy.predict(X_te, Y_te)

    ### CALCULATE BALANCED ACCURACY / ACCURACY ###
    if DATASET == 'PLANKTON10':
        accuracy = metrics.balanced_accuracy_score(Y_te, P)
    else:
        accuracy = float(1.0 * (Y_te==P).sum().item() / len(Y_te))
    ###############################################

    acc.append(round(accuracy, 4))
    num_labeled_samples.append(len(ALD.index['labeled']))
    
    logger.debug(f"Round: {rnd}, Testing accuracy {acc[rnd]}")

    ##### LOGGING #####
    logger.info(f'Round: {rnd}, Testing accuracy: {acc[rnd]}, Number of labeled samples: {len(ALD.index["labeled"])}/{init_pool}, Iteration time: {datetime.now() - init_tic}')
    ###################

    while len(ALD.index['labeled']) < BUDGET + NUM_INIT_LABELED:

        tic = datetime.now()

        rnd += 1
        num_query = min(NUM_QUERY, NUM_INIT_LABELED + BUDGET - len(ALD.index['labeled']), len(ALD.index['unlabeled']))
        
        ### QUERY SAMPLES FOR LABELING ###
        queried_idxs = strategy.query(num_query)
        list_queried_idxs.append(np.asarray(queried_idxs))

        ### CLASS COUNTING ###
        class_count, class_count_percent = ALD.count_class_query(queried_idxs)

        ### MOVE SAMPLES FROM UNLABELED TO LABELED POOL ###
        ALD.move_from_unlabeled_to_labeled(queried_idxs, strategy)

        ### TRAIN CLASSIFIER ON UPDATED LABELED POOL ###
        strategy.train()

        ### PREDICT ON UNLABELED POOL ###
        P = strategy.predict(X_te, Y_te)

        ### CONFUSION MATRIX ###
        from plotcm import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        if rnd % 2:
            stacked = torch.stack((Y_te, P),dim=1)
            cmt = torch.zeros(data_args['num_classes'], data_args['num_classes'], dtype=torch.int64)
            for p in stacked:
                tl, pl = p.tolist()
                cmt[tl,pl] = cmt[tl,pl] + 1
            print(cmt)
            cm = confusion_matrix(Y_te, P) 
            plt.figure(figsize=(25,25))
            classes = data_args['class_names']
            #classes = [str(i) for i in range(data_args['num_classes'])]
            plot_confusion_matrix(cm, classes, rnd)
        ########################

        ### CALCULATE BALANCED ACCURACY / ACCURACY ###
        if DATASET == 'PLANKTON10':
            accuracy = metrics.balanced_accuracy_score(Y_te, P)
        else:
            accuracy = float(1.0 * (Y_te==P).sum().item() / len(Y_te))
        ###############################################

        acc.append(round(accuracy, 4))
        num_labeled_samples.append(round(len(ALD.index['labeled']),4))
        logger.debug(f'Round: {rnd}, Testing accuracy: {acc[rnd]}, Samples labeled: {num_labeled_samples[rnd]}, Pool size: {len(ALD.index["unlabeled"])}, Iteration time: {datetime.now()-tic}')


        ##### LOGGING #####
        logger.info(f'Round: {rnd}, Testing accuracy: {acc[rnd]}, Number of labeled samples: {num_labeled_samples[rnd]}/{init_pool}, Iteration time: {datetime.now() - tic}\n'
                f'Queried class distribution: {class_count_percent}')
        ###################
        

    logger.info(f"Acc: {acc}, Num labeled samples: {num_labeled_samples}, Strategy: {type(strategy).__name__}, Total run time: {datetime.now() - init_tic}")

    ##### LOGGING #####
    logger.info(f'\t\t **** FINISHED RUNNING trial {TRIAL+1}/{TRIALS} ****')
    logger.info(f'Testing accuracy: {acc}, Number of labeled samples: {num_labeled_samples}, Strategy: {type(strategy).__name__}, Dataset: {DATASET}, Total iteration time: {datetime.now() - init_tic}\n')
    ###################
    np.save(f'./queried_idxs/{DATASET}_{STRATEGY}_q{NUM_QUERY}_{SEED}', list_queried_idxs)
    acc_all.append(acc)

acc_all = np.asarray(acc_all)
mean_acc = list(np.mean(acc_all, axis=0))
std_acc = list(np.std(acc_all, axis=0))
var_acc = list(np.var(acc_all, axis=0))

##### LOGGING #####
logger.info(f'\t\t **** FINISHED RUNNING ****')
logger.info(f'Mean accuracy: {mean_acc}, Std accuracy: {std_acc}, Var accuracy: {var_acc} Strategy: {type(strategy).__name__}, Dataset: {DATASET}, Total iteration time: {datetime.now() - init_tic}\n'
        f'All results: {acc_all}')
###################
