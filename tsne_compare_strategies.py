from tsne import keras_features as kf 
from keras.datasets import cifar10
import numpy as np
import os

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

    model = kf.model(x=x, num_classes=num_classes, weight_path=weight_path)
    tx, ty = kf.feature_extractor(model, x, out_dir)
    kf.plot_tsne_categories(x, y, tx, ty, queried_idxs, out_dir, tsne_args)



if __name__ == "__main__":

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

    model = kf.model(x=x_test, num_classes=num_classes, weight_path=weight_path)
    tx, ty = kf.feature_extractor(model, x_test, out_dir)
    
    kf.plot_tsne_categories(x_test, y_test, tx, ty, queried_idxs, out_dir)