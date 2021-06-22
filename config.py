import json
import argparse
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--strategy', default='CORESET', help='AL strategy')
parser.add_argument('--dataset', default='CIFAR10', help='Dataset')
parser.add_argument('--num_query', default=1000, type=int, help='Number of samples to query each round')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers in torch')
parser.add_argument('--cuda_n', default=1, type=int, help='Cuda to use (0 or 1)')
args = vars(parser.parse_args())

STRATEGY = args['strategy']
DATASET = args['dataset']
NUM_QUERY = args['num_query']
CUDA_N = args['cuda_n']
NUM_WORKERS = args['num_workers']

TRIALS = 3

args = {
    "CIFAR10": {
        "al_args": {
            "num_init_labeled": 3900,
            "num_query": NUM_QUERY,
            "budget": 3000,
        },
        "data_args": {
            "data_dir": "../datasets/cifar10/",
            "plot_dir": "./plots",
            "file_ending": ".png",
            "class_names": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            "img_dim": 32,
            "num_classes": 10,
            "num_channels": 3,
            "fraction": 0.08,
        },
        "learning_args": {
            "n_epoch": 30,
            "net": "resnet18",
            "img_dim": 32,
            "num_classes": 10,
            'transform': transforms.Compose(
                            [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=20,translate=(0.1,0.1),fillcolor=(255,255,255)),
                            transforms.RandomCrop(size=32,padding=4),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]), 
            'tr_args': {'batch_size': 64, 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9, 'num_workers': NUM_WORKERS},
            'valid_args': {'batch_size': 400, 'num_workers': NUM_WORKERS},
            'te_args': {'batch_size': 1000, 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9, 'num_workers': NUM_WORKERS},
        }
    }, 
    "MNIST": {
        "al_args": {
            "strategy": "RANDOM",
            "num_init_labeled": 100,
            "num_query": NUM_QUERY,
            "budget": 3500,
        },
        "data_args": {
            "data_dir": "../datasets/mnist/",
            "plot_dir": "./plots",
            "file_ending": ".png",
            "class_names": ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', \
                            'NINE'],
            "img_dim": 32,
            "num_classes": 10,
            "num_channels": 3,
            "fraction": 0.08,
        },
        "learning_args": {
            "n_epoch": 30,
            "net": "net3",
            "img_dim": 32,
            "num_classes": 10,
            "transform": transforms.Compose(
                            [transforms.Grayscale(num_output_channels=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=7,translate=(0.1,0.1),fillcolor=(255,255,255)),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=(0.95,), std=(0.2,))]), 
            'tr_args': {'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9, 'num_workers': NUM_WORKERS},
            'valid_args': {'batch_size': 400, 'num_workers': NUM_WORKERS},
            'te_args': {'batch_size': 1000, 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9, 'num_workers': NUM_WORKERS},
        }
    },
    "PLANKTON10": {
        "al_args": {
            "num_init_labeled": 7484,
            "num_query": NUM_QUERY,
            "budget": 6500,
        },
        "data_args": {
            "data_dir": "../datasets/train10",
            "plot_dir": "./plots",
            "file_ending": ".jpg",
            "class_names": ['trichodesmium puff', 'protist other', 'acantharia protist', 'appendicularian s-shape', \
                            'hydromedusae solmaris', 'trichodesmium bowtie', 'chaetognat sagitta', 'copepod cyclopoid oithona eggs', \
                            'detritus other', 'echinoderm larva seastar brachiolaria'],
            "img_dim": 32, 
            "num_classes": 10,
            "num_channels": 3,
            "fraction": 1,
        },
        "learning_args": {
            'n_epoch': 30,
            'net': 'net3',
            "img_dim": 32,
            "num_classes": 10,
            'transform': transforms.Compose(
                            [transforms.Grayscale(num_output_channels=3),
                            #transforms.RandomHorizontalFlip(),
                            #transforms.RandomVerticalFlip(),
                            #transforms.RandomAffine(degrees=30,translate=(0.1,0.1),fillcolor=(255,255,255)),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.95,), std=(0.2,))]), 
            'tr_args': {'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
            'valid_args': {'batch_size': 379, 'num_workers': NUM_WORKERS},
            'te_args': {'batch_size': 1000, 'lr': 0.0001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},    
        }
    },
    "AILARON": {
        "al_args": {
            "num_init_labeled": 100,
            "num_query": NUM_QUERY,
            "budget": 3500,
        },
        "data_args": {
            "data_dir": "../datasets/dataset_balanced_new",
            "plot_dir": "./plots",
            "file_ending": ".tiff",
            "class_names": ['FISH EGG', 'COPEPOD', 'DIATOM CHAIN', 'OTHER', 'FAECAL PELLETS', 'BUBBLE'],
            "img_dim": 32, 
            "num_classes": 6,
            "num_channels": 3,
            "fraction": 1,
        },
        "learning_args": {
            'n_epoch': 30,
            'net': 'net3',
            "img_dim": 32,
            "num_classes": 6,
            'transform': transforms.Compose(
                            [transforms.Grayscale(num_output_channels=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=30,translate=(0.1,0.1),fillcolor=(255,255,255)),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.95,), std=(0.2,))]), 
            'tr_args': {'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
            'valid_args': {'batch_size': 201, 'num_workers': NUM_WORKERS},
            'te_args': {'batch_size': 1000, 'lr': 0.0001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},  
        }
    },
    "PASTORE": {
        "al_args": {
            "num_init_labeled": 3900,
            "num_query": NUM_QUERY,
            "budget": 500,
        },
        "data_args": {
            "data_dir": "../datasets/Pastore_Training",
            "plot_dir": "./plots",
            "file_ending": ".jpg",
            "class_names": ['VOLVOX', 'SPIROSTOMUM AMBIGUUM', 'BLEPHARISMA AMERICANUM', 'ACTINOSPHAERIUM NUCLEOFILUM', \
                            'EUPLOTES EURYSTOMUS', 'STENTOR COERULEUS', 'DILEPTUS', 'DIDINIUM NASUTUM', 'PARAMECIUM BURSARIA', \
                            'ARCELLA VULGARIS'],
            "img_dim": 32, 
            "num_classes": 10,
            "num_channels": 3,
            "fraction": 1,
        },
        "learning_args": {
            'n_epoch': 30,
            'net': 'resnet18',
            "img_dim": 32,
            "num_classes": 10,
            'transform': transforms.Compose(
                            [transforms.Grayscale(num_output_channels=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=30,translate=(0.1,0.1),fillcolor=(255,255,255)),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.95,), std=(0.2,))]), 
            'tr_args': {'batch_size': 32, 'lr': 0.00001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},
            'valid_args': {'batch_size': 199, 'num_workers': NUM_WORKERS},
            'te_args': {'batch_size': 1000, 'lr': 0.00001, 'weight_decay': 0.0001, 'num_workers': NUM_WORKERS},  
        }
    }
}

