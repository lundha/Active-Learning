import torch
import os 
import matplotlib.pyplot as plt
import numpy as np
import argparse

from tsne import tsne_model, tsne_feature_extractor, plot_tsne_categories, plot_tsne_images
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from utils.utils import load_data_pool, print_image, sub_sample_dataset, load_data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils
from query_strategies import Strategy, Coreset_Strategy, Random_Strategy, Uncertainty_Strategy, Max_Entropy_Strategy, Bayesian_Sparse_Set_Strategy, \
                            DFAL_Strategy, CIRAL_Strategy, Softmax_Hybrid_Strategy, BADGE_Strategy, Learning_Loss_Strategy, Margin_Strategy, \
                            KMeans_Strategy, ActiveLearningByLearning_Strategy, All_Data_Strategy
from datetime import datetime
from kcenter_greedy import KCenterGreedy
from skimage import io, transform
from activelearningdataset import ActiveLearningDataset
from get_net import get_net
from get_dataset import get_dataset
from config import args, STRATEGY, DATASET, NUM_WORKERS, TRIALS, CUDA_N
from plot import plot_learning_curves
from copy import deepcopy
from keras.datasets import cifar10
from random import randint
import logging
from config import DATASET, STRATEGY



def imshow(before_aug, after_aug=None, img_name=DATASET, title1=None, title2=None):
    """Imshow for Tensor."""

    if after_aug is not None:

        fig=plt.figure(figsize=(8, 8))
        columns = 1
        rows = 2

        before_aug, after_aug = denormalise(before_aug), denormalise(after_aug)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(before_aug) 
        fig.tight_layout()
        plt.title(title1)
        plt.axis('off')
        plt.title(title1)
        fig.add_subplot(rows, columns, 2)
        fig.tight_layout()

        plt.imshow(after_aug)
        plt.title(title2)
        plt.axis('off')
        plt.title(title2)
        plt.savefig(f"./aug_plots/{img_name}.png")
    else:

        img = denormalise(img)

        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"./dataset_plots/{img_name}.png")
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
    

def denormalise(img):
    img = img.numpy().transpose((1, 2, 0))
    if DATASET in ('PLANKTON10', 'AILARON', 'PASTORE', 'MNIST'):
        mean = np.array([0.95, 0.95, 0.95])
        std = np.array([0.2, 0.2, 0.2])
    elif DATASET == 'CIFAR10':
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    else:
        raise NotImplementedError('Dataset not valid')
    img = std * img + mean
    
    img = np.clip(img, 0, 1)
    return img


def visual_augmentation():

    learning_args = args[DATASET]['learning_args']
    data_args = args[DATASET]['data_args']
    transform = learning_args['transform']
    X_tr, Y_tr, _, _, _, _ = get_dataset(DATASET, data_args)

    transform_no_aug = transforms.Compose(
                            [
                            transforms.Grayscale(num_output_channels=3),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.95,), std=(0.2,))
                            ]) 
    data_handler = DataHandler(X_tr,Y_tr, transform)
    data_handler_no_aug = DataHandler(X_tr, Y_tr, transform_no_aug)
    data_loader = DataLoader(data_handler, batch_size=8, num_workers=4)
    data_loader_no_aug = DataLoader(data_handler_no_aug, batch_size=8, num_workers=4)

    # Get a batch of training data
    inputs, _, _ = next(iter(data_loader))
    inputs_no_aug, _, _ = next(iter(data_loader_no_aug))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    out_no_aug = torchvision.utils.make_grid(inputs_no_aug)

    imshow(out_no_aug, out, img_name='comparison')

def visual_dataset():

    data_args = args[DATASET]['data_args']
 
    config = {
        'CIFAR10' : {'class_name': ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'],
                    'transform': transforms.Compose(
                            [transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
                    },
        'PLANKTON10': {'class_name': ['Trichodesmium puff', 'Protist', 'Acantharia protist', 'Appendicularian s-shape', \
                                        'Hydromedusae solmaris', 'Trichodesmium bowtie', 'Chaetognath sagitta', 'Copepod cyclopoid oithona eggs', \
                                        'Detritus', 'Echinoderm larva seastar brachiolaria'],
                        'transform': transforms.Compose(
                            [transforms.ToTensor(), 
                            transforms.Normalize(mean=(0.95,), std=(0.2,))])
                        },
        'AILARON': {'class_name': ['fish_egg', 'copepod', 'diatom_chain', 'other', 'faecal_pellets', 'bubble'],
                    'transform': transforms.Compose(
                            [transforms.ToTensor(), 
                            transforms.Normalize(mean=(0.95,), std=(0.2,))
                            ])
                    },
        'PASTORE': {'class_name': ['volvox', 'spirostomum_ambigum', 'blepharisma_americanum', 'actinosphaerium_nucleofilum', 'euplotes_eurystomus', 'stentor_coeruleus', \
                    'dileptus', 'didinum_nasutum', 'paramecium_bursaria', 'arcella_vulgaris'],
                    'transform': transforms.Compose(
                            [transforms.ToTensor(), 
                            transforms.Normalize(mean=(0.95,), std=(0.2,))])
                    }
        }
    class_names = config[DATASET]['class_name']
    transform = config[DATASET]['transform']
                        
    X_tr, Y_tr, _, _, _, _ = get_dataset(DATASET, data_args)

    data_handler = DataHandler(X_tr,Y_tr, transform)
    data_loader = DataLoader(data_handler, batch_size=100, num_workers=4)

    indices_plotted = [0 for _ in range(len(class_names))]
    print(len(indices_plotted))
    images, indices, _ = next(iter(data_loader))
    example_rows = 2
    example_cols = int(len(class_names)//2)
    # Show a grid of example images    
    fig, axes = plt.subplots(example_rows, example_cols, figsize=(11, 5)) #  sharex=True, sharey=True)
    axes = axes.flatten()
    for image, index in zip(images, indices):
        if (indices_plotted[index] == 1):
            continue
        else:
            ax = axes[index]
            ax.imshow(denormalise(image))
            ax.set_axis_off()
            ax.set_title(class_names[index], fontsize=7)
            indices_plotted[index] = 1
    
    fig.subplots_adjust(wspace=0.06, hspace=1)
    #fig.subtitle(DATASET, fontsize=20)
    plt.savefig(f'./dataset_plots/{DATASET}.png')
    #out = torchvision.utils.make_grid(inputs)
    #imshow_single(out, img_name=DATASET)

def compare_tsne():

    al_args = args[DATASET]['al_args']
    learning_args = args[DATASET]['learning_args']
    data_args = args[DATASET]['data_args']

    tsne_args = {'dataset': DATASET, 'strategy': STRATEGY}
    NET = learning_args['net']
    FRACTION = data_args['fraction']
    NUM_QUERY = 200 
    NUM_INIT_LABELED = 0 

    num_classes = data_args['num_classes']

    #(_, _), (X_te, Y_te) = cifar10.load_data()
    X_tr, Y_tr, X_te, Y_te, _, _ = get_dataset(DATASET, data_args)

    #X_tr_tsne, Y_tr_tsne = deepcopy(X_tr), deepcopy(Y_tr)
    X_te_tsne, Y_te_tsne = deepcopy(X_te), deepcopy(Y_te)

    # Generate initially labeled pool 
    #ALD = ActiveLearningDataset(X_tr, Y_tr, _, _, _, _, NUM_INIT_LABELED)
    ALD = ActiveLearningDataset(X_te, Y_te, _, _, _, _, NUM_INIT_LABELED)
    # Load network 
    net = get_net(NET, data_args, strategy=STRATEGY)
    list_queried_idxs = []
    tic = datetime.now()

    ### LOGGER

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    LOG_PATH = f"./TSNE_LOGS"
    LOG_FILE = f"{LOG_PATH}/{DATASET}_{STRATEGY}_q{NUM_QUERY}_f{FRACTION}.log"

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    
    # Load active learning strategy
    strategies = {
        'ALL': ActiveLearningByLearning_Strategy,
        'ALL-DATA': All_Data_Strategy,
        'BAYESIAN_SPARSE_SET': Bayesian_Sparse_Set_Strategy,
        'BADGE': BADGE_Strategy,
        'CIRAL': CIRAL_Strategy,
        'CORESET': Coreset_Strategy,
        'DFAL': DFAL_Strategy,
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

    logger.info(f'Strategy: {type(strategy).__name__}, dataset: {DATASET}')    
    queried_idxs = strategy.query(NUM_QUERY)
    logger.info(f'Queried {len(queried_idxs)} samples')
    list_queried_idxs.append(np.asarray(queried_idxs))

    weight_config = {
        'CIFAR10': '/home/martlh/masteroppgave/core-set/tsne/weights/CIFAR10-weights.97-0.5117.hdf5',
        'PASTORE': '/home/martlh/masteroppgave/core-set/tsne/weights/PASTORE-weights.94-0.0986.hdf5',
        'KAGGLE': '/home/martlh/masteroppgave/core-set/tsne/weights/PLANKTON10-weights.91-0.3682.hdf5',
        'AILARON': '/home/martlh/masteroppgave/core-set/tsne/weights/AILARON-weights.95-0.2208.hdf5'
    }
    weight_path = weight_config[DATASET]

    seed = randint(1,1000)
    logger.info(seed)
    np.save('./queried_idxs', list_queried_idxs)

    plot_tsne(X_te_tsne, Y_te_tsne, list_queried_idxs, num_classes, tsne_args, weight_path, data_set=DATASET, seed=seed)    
    logger.info(f"Total run time: {datetime.now()-tic}")



def plot_tsne(x: list, y: list, queried_idxs: list, num_classes: int, tsne_args: dict, weight_path: str, data_set: str, seed: int) -> None:
    '''
    Create T-SNE plot based on data pool. Highlight queried data points with black color
    '''

    out_dir = 'tsne_plots_new'
    x = x.astype('float32')
    x /= 255
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
  
    model = tsne_model(x=x, num_classes=num_classes, weight_path=weight_path)
    tx, ty = tsne_feature_extractor(model, x, out_dir, data_set, new_weights=True)
    #plot_tsne_images(x, y, tx, ty, out_dir, data_set, seed)
    plot_tsne_categories(x, y, tx, ty, queried_idxs, out_dir, tsne_args, seed)


if __name__ == "__main__":

    compare_tsne()
    #visual_augmentation()
    #visual_dataset()