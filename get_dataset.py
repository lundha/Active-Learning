import numpy as np
import torch
import sys
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from dataloader import DataSet, Resize, Normalize, ToTensor, Convert2RGB
from sklearn.model_selection import KFold
from utils.utils import my_subset
import os

def get_dataset(name, load_data_args):

    if name == 'CIFAR10':
        return get_CIFAR10(load_data_args)
    elif name == 'PLANKTON10':
        return get_PLANKTON10(load_data_args)
    elif name == 'MNIST':
        return get_MNIST(load_data_args)
    elif name == 'AILARON':
        return get_AILARON(load_data_args)
    elif name == 'PASTORE':
        return get_PASTORE(load_data_args)
    else:
        raise NotImplementedError('Dataset not supported')

def get_MNIST(load_data_args):

    data_tr = datasets.MNIST('./MNIST', train=True, download=True)
    data_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = np.array(data_tr.data)
    Y_tr = data_tr.targets
    X_te = np.array(data_te.data)
    Y_te = data_te.targets
    Y_tr, Y_te = torch.from_numpy(np.array(Y_tr)), torch.from_numpy(np.array(Y_te))
    X_tr, Y_tr = make_subset(X_tr, Y_tr, Fraction=load_data_args['fraction'])
    X_te, Y_te = make_subset(X_te, Y_te, Fraction=load_data_args['fraction'])
    
    split = int(len(X_te)//5)
    X_valid, Y_valid = X_te[:split], Y_te[:split]
    X_te, Y_te = X_te[:split], Y_te[:split]

    return X_tr, Y_tr, X_te, Y_te, X_valid, Y_valid
    


def get_CIFAR10(load_data_args):
    data_tr = datasets.CIFAR10('./CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = data_tr.targets
    X_te = data_te.data
    Y_te = data_te.targets
    Y_tr, Y_te = torch.from_numpy(np.array(Y_tr)), torch.from_numpy(np.array(Y_te))
    X_tr, Y_tr = make_subset(X_tr, Y_tr, Fraction=load_data_args['fraction'])
    X_te, Y_te = make_subset(X_te, Y_te, Fraction=load_data_args['fraction'])
    
    split = int(len(X_te)//5)
    X_valid, Y_valid = X_te[:split], Y_te[:split]
    X_te, Y_te = X_te[split:], Y_te[split:]

    return X_tr, Y_tr, X_te, Y_te, X_valid, Y_valid


def get_PLANKTON10(load_data_args):
    data_dir = "../"+load_data_args['data_dir']+"/"
    print(data_dir)
    header_file = data_dir + 'header.tfl.txt'
    filename = data_dir + 'image_set.data'
    file_ending = load_data_args['file_ending']
    num_classes = load_data_args['num_classes']
    img_dim = load_data_args['img_dim']
    X_tr, X_te = [], []
    Y_tr, Y_te = [], []

    if not (os.path.exists(f"{data_dir}data.npy") and os.path.exists(f"{data_dir}labels.npy")):
        
        try:
            dataset = DataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                        transform=None, num_classes=num_classes, train=True, img_dim=img_dim)
            
            _images = dataset.images
            _labels = dataset.labels
    
        except Exception as e:
            print(f"Could not load dataset, exception: {e}")
            sys.exit('No dataset')

    else:
        _images = np.load(f"{data_dir}data.npy", allow_pickle=True)
        _labels = np.load(f"{data_dir}labels.npy", allow_pickle=True)
    
    print(f"Shape of images: {_images.shape}")

    shuffled_indices = np.random.permutation([_ for _ in range(len(_images))])
    split = int(len(_images)/5)

    X_valid, Y_valid = _images[shuffled_indices[:int(split//5)]], _labels[shuffled_indices[:int(split//5)]]
    X_te, Y_te = _images[shuffled_indices[int(split//5):split]], _labels[shuffled_indices[int(split//5):split]]
    X_tr, Y_tr = _images[shuffled_indices[split:]], _labels[shuffled_indices[split:]] 

    Y_tr, Y_te, Y_valid = torch.from_numpy(Y_tr), torch.from_numpy(Y_te), torch.from_numpy(Y_valid)

    return X_tr, Y_tr, X_te, Y_te, X_valid, Y_valid

def get_AILARON(load_data_args):
    data_dir = "../"+load_data_args['data_dir']+"/"
    header_file = data_dir + 'header.tfl.txt'
    filename = 'image_set.data'
    file_ending = load_data_args['file_ending']
    num_classes = load_data_args['num_classes']
    img_dim = load_data_args['img_dim']
    X_tr, X_te = [], []
    Y_tr, Y_te = [], []

    if not (os.path.exists(f"{data_dir}data.npy") and os.path.exists(f"{data_dir}labels.npy")):
        
        try:
            dataset = DataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                        transform=None, num_classes=num_classes, train=True, img_dim=img_dim)
            
            _images = dataset.images
            _labels = dataset.labels
    
        except Exception as e:
            print(f"Could not load dataset, exception: {e}")
            sys.exit('No dataset')

    else:
        _images = np.load(f"{data_dir}data.npy", allow_pickle=True)
        _labels = np.load(f"{data_dir}labels.npy", allow_pickle=True)
    
    print(f"Shape of images: {_images.shape}")

    shuffled_indices = np.random.permutation([_ for _ in range(len(_images))])
    split = int(len(_images)/5)

    X_valid, Y_valid = _images[shuffled_indices[:int(split//5)]], _labels[shuffled_indices[:int(split//5)]]
    X_te, Y_te = _images[shuffled_indices[int(split//5):split]], _labels[shuffled_indices[int(split//5):split]]
    X_tr, Y_tr = _images[shuffled_indices[split:]], _labels[shuffled_indices[split:]] 

    Y_tr, Y_te, Y_valid = torch.from_numpy(Y_tr), torch.from_numpy(Y_te), torch.from_numpy(Y_valid)

    return X_tr, Y_tr, X_te, Y_te, X_valid, Y_valid
    
def get_PASTORE(load_data_args):

    data_dir = load_data_args['data_dir']+"/"
    header_file = data_dir + 'header.tfl.txt'
    filename = 'image_set.data'
    file_ending = load_data_args['file_ending']
    num_classes = load_data_args['num_classes']
    img_dim = load_data_args['img_dim']
    X_tr, X_te = [], []
    Y_tr, Y_te = [], []

    if not (os.path.exists(f"{data_dir}data.npy") and os.path.exists(f"{data_dir}labels.npy")):
        
        try:
            dataset = DataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                        transform=None, num_classes=num_classes, train=True, img_dim=img_dim)
            
            _images = dataset.images
            _labels = dataset.labels
    
        except Exception as e:
            print(f"Could not load dataset, exception: {e}")
            sys.exit('No dataset')

    else:
        _images = np.load(f"{data_dir}data.npy", allow_pickle=True)
        _labels = np.load(f"{data_dir}labels.npy", allow_pickle=True)
    
    print(f"Shape of images: {_images.shape}")

    shuffled_indices = np.random.permutation([_ for _ in range(len(_images))])
    split = int(len(_images)/5)

    X_valid, Y_valid = _images[shuffled_indices[:int(split//5)]], _labels[shuffled_indices[:int(split//5)]]
    X_te, Y_te = _images[shuffled_indices[int(split//5):split]], _labels[shuffled_indices[int(split//5):split]]
    X_tr, Y_tr = _images[shuffled_indices[split:]], _labels[shuffled_indices[split:]] 

    Y_tr, Y_te, Y_valid = torch.from_numpy(Y_tr), torch.from_numpy(Y_te), torch.from_numpy(Y_valid)

    return X_tr, Y_tr, X_te, Y_te, X_valid, Y_valid



def make_subset(X, Y, Fraction):

    size = int(Fraction*len(X))
    index = {'include': np.arange(0), 'exclude': np.arange(len(X))} 
    shuffled_indices = np.random.permutation(index['exclude'])
    index['include'], index['exclude'] = shuffled_indices[:size], shuffled_indices[size:]
    return X[index['include']], Y[index['include']]


