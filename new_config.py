import click
import json
import argparse
from torchvision import transforms

args = {
    "CIFAR10": {
        "al_args": {
            "num_init_labeled": 100,
            "num_query": 1000,
            "budget": 3000,
        },
        "load_data_args": {
            "data_dir": "../datasets/cifar10/",
            "plot_dir": "./plots",
            "file_ending": ".png",
            "num_classes": 10,
            "num_channels": 3,
            "fraction": 0.2,
        },
        "learning_args": {
            'n_epoch': 10,
            'transform': transforms.Compose(
                            [transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=7,translate=(0.1,0.1),fillcolor=255),
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
            "num_query": 1000,
            "budget": 3000,
        },
        "load_data_args": {
            "data_dir": "../datasets/mnist/",
            "plot_dir": "./plots",
            "file_ending": ".png",
            "num_classes": 10,
            "num_channels": 1,
            "fraction": 1,
        },
        "learning_args": {
            "n_epoch": 30,
            "transform": transforms.Compose(
                            [transforms.Grayscale(num_output_channels=3),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(degrees=7,translate=(0.1,0.1),fillcolor=255),
                            transforms.RandomCrop(size=32,padding=4),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=(0.95,), std=(0.2,))]), 
            'tr_args': {'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9, 'num_workers': NUM_WORKERS},
            'valid_args': {'batch_size': 400, 'num_workers': NUM_WORKERS},
            'te_args': {'batch_size': 1000, 'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9, 'num_workers': NUM_WORKERS},
        }
    },
    "PLANKTON10": {
        "al_args": {
            "num_init_labeled": 100,
            "num_query": 1000,
            "budget": 3000,
        },
        "load_data_args": {
            "data_dir": "../datasets/train10",
            "plot_dir": "./plots",
            "img_dim": 32, 
            "num_classes": 10,
            "file_ending": ".jpg",
            "num_channels": 3,
            "fraction": 1,
        },
        "learning_args": {
            'data_set': 'PLANKTON10',
            'n_epoch': 30,
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
    },
    "AILARON": {
        "al_args": {
            "num_init_labeled": 100,
            "num_query": 1000,
            "budget": 3000,
        },
        "load_data_args": {
            "data_dir": "../datasets/dataset_balanced_new",
            "plot_dir": "./plots",
            "img_dim": 32, 
            "num_classes": 6,
            "file_ending": ".tiff",
            "num_channels": 1,
            "fraction": 1,
        },
        "learning_args": {
            'n_epoch': 30,
            'img_dim': 32,
            'num_classes': 6,
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
}

'''
def update_config(args) -> None:

    new_config = {}
    prev_config = load_config()
    for key, value in prev_config.items():
        new_config.update({key : return_latest_updated_value(value, args.get(key.lower()))})
    print(new_config)


    with open("config.json", 'w') as f:
        json.dump(new_config, f)
    f.close()


def load_config() -> dict:
    
    with open("config.json") as f:
        config = json.load(f)
    f.close()
    
    return config

def return_latest_updated_value(old_value, new_value):

    if old_value == new_value:
        return old_value
    else:
        return new_value


if __name__ == "__main__":
    update_config(args)
'''