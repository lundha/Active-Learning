from __future__ import print_function
import os
import sys
import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from get_dataset import get_dataset
from config import args, STRATEGY, DATASET, NUM_WORKERS, TRIALS, CUDA_N


def model(x_train, num_classes=10):

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], name='conv1'))
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

    model.summary()

    return model


def train_model(x_train, y_train, x_test, y_test, model):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # we will do some mild image pre-processing for augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    batch_size = 64
    epochs = 100

    # initiate Adam opt1mizer
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    filepath = './weights/'+DATASET+'-weights.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_chk = ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                save_best_only=True,
                                save_weights_only=True, mode='auto',
                                period=1)

    csv_log = CSVLogger(f'./weights/{DATASET}-training.log')

    model.fit_generator(datagen.flow(x_train, y_train,
                                    batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[model_chk, csv_log])

if __name__ == "__main__":
    
    data_args = args[DATASET]['data_args']
    X_tr, Y_tr, X_te, Y_te, X_val, Y_val = get_dataset(DATASET, data_args)

    Y_tr, Y_te = keras.utils.to_categorical(Y_tr), keras.utils.to_categorical(Y_te)
    model = model(X_tr, num_classes=data_args['num_classes'])
    train_model(X_tr, Y_tr, X_te, Y_te, model)