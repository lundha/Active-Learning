import click
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../datasets/cifar10', help='Location of data set')
parser.add_argument('--plot_dir', default='./plots', help='Dir for saving plot')
parser.add_argument('--net', default='resnet18', help='Learning network')
parser.add_argument('--strategy', default='coreset', help='AL strategy')
parser.add_argument('--data_set', default='CIFAR10', help='Dataset')
parser.add_argument('--num_init_labeled', default=0, type=int, help='number initial labeled samples')
parser.add_argument('--num_query', default=1000, type=int, help='Number of samples to query each round')
parser.add_argument('--budget', default=10000, type=int, help='Budget for sample annotation')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers in torch')
parser.add_argument('--fraction', default=1, type=float, help='fraction of samples to use')
parser.add_argument('--cuda_n', default=1, type=int, help='Cuda to use (0 or 1)')
args = vars(parser.parse_args())

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
