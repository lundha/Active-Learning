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
from utils.utils import map_list

def tsne_model(x, num_classes=10, weight_path="/tsne/v5-weights.48-0.4228.hdf5"):

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
    model.add(Dense(512, name='fc1'))
    model.add(BatchNormalization(axis=1, name='bn_fc1'))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, name='output'))
    model.add(BatchNormalization(axis=1, name='bn_outptut'))
    model.add(Activation('softmax'))

    model.load_weights(weight_path)

    return model 


def tsne_feature_extractor(model, data_x, out_dir):
    batch_size = 32

    # Get featues
    feat_extractor = Model(inputs=model.input,
                        outputs=model.get_layer('fc1').output)
    features = feat_extractor.predict(data_x, batch_size=batch_size)
    np.save(os.path.join(out_dir, 'fc1_features.npy'), features)

    features = np.load(os.path.join(out_dir, 'fc1_features.npy'))
    # TSNE transfrom features
    tsne = TSNE().fit_transform(features)
    np.save(os.path.join(out_dir, 'fc1_features_pca_tsne_default.npy'), tsne)

    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    return tx, ty


def plot_tsne_images(data_x, tx, ty):

    width = 4000
    height = 3000
    max_dim = 100

    full_image = Image.new('RGB', (width, height))
    for idx, x in enumerate(data_x):
        tile = Image.fromarray(np.uint8(x * 255))
        #tile = Image.open(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs),
                            int(tile.height / rs)),
                        Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim) * tx[idx]),
                                int((height-max_dim) * ty[idx])))

    plt.figure(figsize = (16,12))
    plt.imshow(full_image)

    full_image.save(os.path.join(out_dir, "fc1_features_tsne_default.jpg"))

# TSNE with categories
def plot_tsne_categories(data_x, data_y, tx, ty, queried_idxs, out_dir, args):

    # have to re-load cifar to get y_test back in its original form
    # _, (x_test, y_test) = cifar10.load_data()
    dataset, strategy = args['dataset'], args['strategy']
    seed = randint(1,100)

    plt.figure(figsize = (8,6))

    for j in range(len(queried_idxs)):

        new_data_y = np.asarray(data_y)
        mapped_y = map_list(new_data_y, queried_idxs[j])
        mapped_y = np.asarray(mapped_y)
        print(f"Length mapped y: {mapped_y}")
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(len(classes)):
            # y_i is a vector that is true on corresponding indexes with data_y for each class in classes
            # i.e true for all 'airplane' elements in data_y on first iteration. This is to correctly color the
            # scatter plot
            y_i = new_data_y == i
            plt.scatter(tx[y_i[:,0]], ty[y_i[:,0]], label=classes[i])
            # An idea would be to do an equal masking with queried elements
        #plt.legend(loc=4)
        plt.scatter(tx[mapped_y], ty[mapped_y], marker="^", c='black')
        plt.gca().invert_yaxis()

        len_q_idx = len(queried_idxs[j])
        strat = strategy[j]
        
        plt.savefig(os.path.join(out_dir, f"TSNE_{dataset}_q{len_q_idx}_{strat}_{seed}.eps"))

def plots_tsne():
    pass



if __name__ == "__main__":

    # This time we will only use the test set:
    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')
    x_test /= 255
    print(x_test.shape[0], 'test samples')
    print(type(y_test))
    print(type(x_test))
    # Convert class vectors to binary class matrices.
    num_classes = 10
    
    weight_path = '/Users/martin.lund.haug/Documents/Masteroppgave/core-set/tsne/v5-weights.48-0.4228.hdf5'
    out_dir = 'v5-features'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = model(weight_path=weight_path)
    tx, ty = feature_extractor(model, x_test)
    
    plot_tsne_categories(x_test, y_test, tx, ty)
    

