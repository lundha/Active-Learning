from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


# This time we will only use the test set:
_, (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
num_classes = 10

y_test = keras.utils.to_categorical(y_test, num_classes)

import os

weights_path = 'v5-weights.48-0.4228.hdf5'
out_dir = 'v5-features'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_test.shape[1:], name='conv1'))
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

model.load_weights("./v5-weights.17-0.4992.hdf5")

import numpy as np

batch_size = 32

feat_extractor = Model(inputs=model.input,
                       outputs=model.get_layer('fc1').output)

features = feat_extractor.predict(x_test, batch_size=batch_size)

np.save(os.path.join(out_dir, 'fc1_features.npy'), features)

# PCA - FC1

from sklearn.manifold import TSNE

features = np.load(os.path.join(out_dir, 'fc1_features.npy'))
tsne = TSNE().fit_transform(features)

np.save(os.path.join(out_dir, 'fc1_features_tsne_default.npy'), tsne)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))


import matplotlib.pyplot as plt
from PIL import Image

width = 4000
height = 3000
max_dim = 100

full_image = Image.new('RGB', (width, height))
for idx, x in enumerate(x_test):
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

# have to re-load cifar to get y_test back in its original form
_, (x_test, y_test) = cifar10.load_data()

y_test = np.asarray(y_test)

plt.figure(figsize = (16,12))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(len(classes)):
    y_i = y_test == i
    plt.scatter(tx[y_i[:, 0]], ty[y_i[:, 0]], label=classes[i])
plt.legend(loc=4)
plt.gca().invert_yaxis()
plt.savefig(os.path.join(out_dir, "fc1_features_tsne_default_pts.jpg"), bbox_inches='tight')
plt.show()
