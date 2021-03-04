
from dataloader import DataSet, Resize, Normalize, ToTensor, Convert2RGB
from torchvision import transforms, utils
import torch
from torch.utils.data import DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from autoencoder import Autoencoder

def load_data_pool(train=False, arg=None) -> DataSet:
    '''
    Load data pool if it not exist, else return 
    '''
    if train:
        data_dir = arg['data_dir'] + "train/"
    else:
        data_dir = arg['data_dir'] + "test/"

    print(data_dir)
    num_classes = arg['num_classes']
    file_ending = arg['file_ending']
    print(file_ending)
    print(num_classes)

    header_file = data_dir + "header.tfl.txt"
    filename = data_dir + "image_set.data"
    
    
    try:
        dataset = DataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                    num_classes=num_classes, train=train)
    except Exception as e:
        print(f"Could not load dataset, exception: {e}")

    return dataset

def print_images(idxs, cols, rows):

  
    fig=plt.figure(figsize=(8, 8))
    columns = cols
    rows = rows

    for i, idx in enumerate(idxs):
        
        img_name = dataset.dataset.iloc[idx, 0].split(' ')[0]
        label  = dataset.dataset.iloc[idx, 0].split(' ')[1]
        label_name = dataset.classlist[int(label)]

        img = io.imread(img_name)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        plt.title(label_name)

    plt.show()


def print_image(dataset, idx):
    '''
    Print image from dataset. Args: dataset and index in dataset
    '''
    img_name = dataset.dataset.iloc[idx, 0].split(' ')[0]
    label  = dataset.dataset.iloc[idx, 0].split(' ')[1]
    label_name = dataset.classlist[int(label)]
    image = io.imread(img_name)
    plt.imshow(image)
    plt.title(label_name)
    plt.show()

def sub_sample_dataset(x, y, new_size):

    sample = np.random.randint(0,len(y),new_size)
    new_x = [x[i] for i in sample]
    new_y = [y[i] for i in sample]

    return np.array(new_x), np.array(new_y)

def get_embedding(dataloader, embedding_dim) -> np.array:
    '''
    Create and save embedding, arg: handler 
    '''
    encoder = Autoencoder(embedding_dim)

    embedding = torch.zeros([len(dataloader.dataset), embedding_dim])
    
    with torch.no_grad():
        for x, y, idxs in dataloader:
            out, e1 = encoder(x)
            embedding[idxs] = e1.cpu()
    np.save('/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/embedding.npy', embedding)
    embedding = embedding.numpy()
    return embedding


def calculate_distance_matrix(self, embedding) -> np.array:
    '''
    Calculate and save distance matrix, input is embedding
    '''
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(embedding), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    np.save('/Users/martin.lund.haug/Documents/Masteroppgave/cifar10/distance_matrix.npy', dist_mat)
    print(f"Time taken to generate distance matrix: {datetime.now() - t_start}")
    return dist_mat


def save_to_file(filetype, filename, file):
    if filetype == 'plot':
        pass
    elif filetype == 'list':
        pass
    elif filetype == 'result':
        pass



def load_data(dir, train):
    '''
    Loading numpy arrays and their corresponding labels
    dir: Directory where the data is saved
    train: Boolean for train/test data
    '''
    if train:
        X_tr = np.load(f'{dir}/train_data.npy')
        Y_tr = np.load(f'{dir}/train_labels.npy')
        return X_tr, Y_tr    
    else:
        X_te = np.load(f'{dir}/test_data.npy')
        Y_te = np.load(f'{dir}/test_labels.npy')
        return X_te, Y_te



if __name__ == "__main__":
    pass