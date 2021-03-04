import click
import json

@click.command()
@click.option('--data_dir', default='/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10', help='Location of data set')
@click.option('--plot_dir', default='/Users/martin.lund.haug/Documents/Masteroppgave/core-set/plots', help='Dir for saving plot')
@click.option('--net', default='resnet18', help='Learning network')
@click.option('--data_set', default='CIFAR10', help='Dataset')
@click.option('--num_init_labeled', default=0, help='number initial labeled samples')
@click.option('--num_query', default=1000, help='Number of samples to query each round')
@click.option('--budget', default=10000, help='Budget for sample annotation')
@click.option('--num_workers', default=4, help='number of workers in torch')
@click.option('--fraction', default=1, help='fraction of samples to use')


def update_config(data_dir, plot_dir, net: str, data_set: str, num_init_labeled: int, num_query: int, budget: int,
                num_workers: int, fraction: int) -> None:

    json_file = {}

    json_file['NET'] = net
    json_file['DATA_SET'] = data_set
    json_file['NUM_INIT_LABELED'] = num_init_labeled
    json_file['NUM_QUERY'] = num_query
    json_file['BUDGET'] = budget
    json_file['NUM_WORKERS'] = num_workers
    json_file['FRACTION'] = fraction

    with open("config.json", 'w') as f:
        json.dump(json_file, f)
    f.close()

def load_config() -> list:

    with open("config.json") as f:
        config = json.load(f)
    f.close()
    
    return config

if __name__ == "__main__":
    update_config()