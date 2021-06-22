from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os
from random import randint
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from get_dataset import get_dataset
from utils.utils import map_list, add_border
from config import args, STRATEGY, DATASET, NUM_WORKERS, TRIALS, CUDA_N


def tsne_model(x, num_classes=10, weight_path="/tsne/weights/cifar-v5-weights.48-0.4228.hdf5"):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x.shape[1:], name='conv1'))
    model.add(BatchNormalization(axis=3, name='bn_conv1'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name='conv2'))
    model.add(BatchNormalization(axis=3, name='bn_conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization(axis=3, name='bn_conv3'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), name='conv4'))
    model.add(BatchNormalization(axis=3, name='bn_conv4'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(50, name='fc1'))
    model.add(BatchNormalization(axis=1, name='bn_fc1'))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, name='output'))
    model.add(BatchNormalization(axis=1, name='bn_outptut'))
    model.add(Activation('softmax'))

    model.load_weights(weight_path)

    return model 


def tsne_feature_extractor(model, data_x, out_dir, data_set, new_weights=None):
    batch_size = 32

    # Get featues
    if os.path.isfile(os.path.join(out_dir, f'{data_set}_fc1_features.npy')) and new_weights == False:
        features = np.load(os.path.join(out_dir, f'{data_set}_fc1_features.npy'))
    else:
        feat_extractor = Model(inputs=model.input,
                            outputs=model.get_layer('fc1').output)
        features = feat_extractor.predict(data_x, batch_size=batch_size)
        np.save(os.path.join(out_dir, f'{data_set}_fc1_features.npy'), features)
    
    # TSNE transfrom features
    if os.path.isfile(os.path.join(out_dir, f'{data_set}_fc1_features_pca_tsne_default.npy')) and new_weights == False:
        tsne = np.load(os.path.join(out_dir, f'{data_set}_fc1_features_pca_tsne_default.npy'))
    else:
        tsne = TSNE().fit_transform(features)    
        np.save(os.path.join(out_dir, f'{data_set}_fc1_features_pca_tsne_default.npy'), tsne)

    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    return tx, ty


def plot_tsne_images(data_x, data_y, tx, ty, out_dir, DATASET, args, seed):

    width = 1333
    height = 1000
    max_dim = 250

    colors = ['red', 'yellow', 'green', 'blue', 'orange', 'indianred']#, 'cyan', 'pink', 'orange', 'brown']

    full_image = Image.new('RGB', (width, height), color=(255,255,255))
    for idx, x in enumerate(data_x):
        tile = Image.fromarray(np.uint8(x * 255))
        #tile = Image.open(img)
        rs = max(4, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs),
                            int(tile.height / rs)),
                        Image.ANTIALIAS)
        tile = add_border(tile, border=2, color=colors[data_y[idx]])
        full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                                int((height-max_dim) * ty[idx])))

    plt.figure(figsize = (8,6))
    plt.imshow(full_image)

    full_image.save(os.path.join(out_dir, f"{DATASET}_tsne_images_{seed}.png"))

# TSNE with categories
def plot_tsne_categories(data_x, data_y, tx, ty, queried_idxs, out_dir, args, seed):

    # have to re-load cifar to get y_test back in its original form
    # _, (x_test, y_test) = cifar10.load_data()
    dataset, strategy = args['dataset'], args['strategy']
    plt.figure(figsize = (8,6))

    colors = ['red', 'yellow', 'green', 'blue', 'orange', 'indianred', 'magenta', 'pink', 'orange', 'brown']

    if dataset.upper() == 'CIFAR10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset.upper() == 'PLANKTON10':
        classes = ['trichodesmium_puff', 'protist_other', 'acantharia_protist', 'appendicularian_s_shape', \
                    'hydromedusae_solmaris', 'trichodesmium_bowtie', 'chaetognath_sagitta', 'copepod_cyclopoid_oithona_eggs' \
                    'detritus_other', 'echinoderm_larva_seastar_brachiolaria'] 
    else: 
        print("Unknown dataset")

    for j in range(len(queried_idxs)):

        new_data_y = np.asarray(data_y)
        mapped_y = map_list(new_data_y, queried_idxs[j])
        mapped_y = np.asarray(mapped_y)

        for i in range(len(classes)):
            # y_i is a vector that is true on corresponding indexes with data_y for each class in classes
            # i.e true for all 'airplane' elements in data_y on first iteration. This is to correctly color the
            # scatter plot
            y_i = new_data_y == i

            plt.scatter(tx[y_i], ty[y_i], label=classes[i], color=colors[i])
            #plt.legend(loc=4)

        plt.scatter(tx[mapped_y], ty[mapped_y], marker="^", c='black')
        plt.gca().invert_yaxis()
        plt.axis('off')
        len_q_idx = len(queried_idxs[j])
        strat = strategy[j]
        
        plt.savefig(os.path.join(out_dir, f"TSNE_{DATASET}_q{len_q_idx}_{strat}_{seed}.png"))


def plot_tsne_category(data_x, data_y, tx, ty, out_dir, data_set, seed, q_idx):

    plt.figure(figsize = (8,6))
    new_data_y = np.asarray(data_y)

    mapped_y = map_list(new_data_y, q_idx)
    mapped_y = np.asarray(mapped_y)
    colors = ['red', 'yellow', 'green', 'blue', 'purple', 'indianred', 'cyan', 'pink', 'orange', 'brown']

    for i in range(len(np.unique(data_y))):
        # y_i is a vector that is true on corresponding indexes with data_y for each class in classes
        # i.e true for all 'airplane' elements in data_y on first iteration. This is to correctly color the
        # scatter plot
        y_i = new_data_y == i

        plt.scatter(tx[y_i], ty[y_i], color=colors[i])
        # An idea would be to do an equal masking with queried elements
    plt.scatter(tx[mapped_y], ty[mapped_y], marker="^", c='black')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, f"{data_set}_tsne_category_{seed}.png"))

def plot_tsne(data_x, data_y, tx, ty, out_dir, DATASET, args, seed):

    plt.figure(figsize = (8,6))
    new_data_y = np.asarray(data_y)

    colors = ['red', 'yellow', 'green', 'blue', 'purple', 'indianred', 'gray', 'pink', 'orange', 'brown']

    for i in range(args['num_classes']):
        # y_i is a vector that is true on corresponding indexes with data_y for each class in classes
        # i.e true for all 'airplane' elements in data_y on first iteration. This is to correctly color the
        # scatter plot
        y_i = new_data_y == i

        plt.scatter(tx[y_i], ty[y_i], color=colors[i])
        # An idea would be to do an equal masking with queried elements
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, f"{DATASET}_tsne_{seed}.png"))

if __name__ == "__main__":

    seed = randint(1,1000)
    print(seed)
    q_idxs = np.load('../queried_idxs.npy')
    al_args = args[DATASET]['al_args']
    learning_args = args[DATASET]['learning_args']
    data_args = args[DATASET]['data_args']
    X_tr, Y_tr, X_te, Y_te, _, _ = get_dataset(DATASET, data_args)
    X_te, Y_te = X_tr, Y_tr
    X_tr = X_tr.astype('float32')
    X_tr /= 255
    
    weight_config = {
        'CIFAR10': '/home/martlh/masteroppgave/core-set/tsne/weights/CIFAR10-weights.97-0.5117.hdf5',
        'PASTORE': '/home/martlh/masteroppgave/core-set/tsne/weights/PASTORE-weights.94-0.0986.hdf5',
        'PLANKTON10': '/home/martlh/masteroppgave/core-set/tsne/weights/PLANKTON10-weights.91-0.3682.hdf5',
        'AILARON': '/home/martlh/masteroppgave/core-set/tsne/weights/AILARON-weights.95-0.2208.hdf5'
    }

    weight_path = weight_config[DATASET]
    out_dir = f'../new_tsne_plots'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    q_idxs = np.load('../queried_idxs/PLANKTON10_CIRAL_q400_545.npy', allow_pickle=True)
    q_idx = q_idxs[4]
    model = tsne_model(X_tr, num_classes=data_args['num_classes'], weight_path=weight_path)
    tx, ty = tsne_feature_extractor(model, X_tr, out_dir, DATASET, new_weights=True)
    #plot_tsne_category(X_tr, Y_tr, tx, ty, out_dir, DATASET, seed, q_idx)
    #plot_tsne(X_tr, Y_tr, tx, ty, out_dir, DATASET, data_args, seed)
    plot_tsne_images(X_tr, Y_tr, tx, ty, out_dir, DATASET, data_args, seed)
