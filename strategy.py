
import torch
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

class Strategy():

    def __init__(self, ALD, net, args):
        self.ALD = ALD
        self.net = net
        self.args = args
        self.n_pool = len(ALD.index['unlabeled'])
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")


    def query(self, n_query):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb


    def _train(self, epoch, loader_tr, optimizer):
        pass

    def train(self):
        pass


    def predict(self, X, Y):
        pass


    def prepare_data(self, X_unlabeled, Y_unlabeled, transform):
        '''
        Creates a dataloader from unlabeled X and Y data points
        :param: X_unlabeled: Unlabeled datapoints Y_unlabeled: Unlabeled labels
        '''
        handler = DataHandler(X_unlabeled, Y_unlabeled, transform)
        loader = DataLoader(handler, shuffle=False, batch_size=32)
        return loader

    def get_embedding(self, dataloader, embedding_dim):
        '''
        Create and save embedding 
        :param: dataloader:dataloader, embedding_din:embedding dimension 
        '''
        encoder = Autoencoder()

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
        np.save('/Users/martin.lund.haug/Documents/Masteroppgave/datasets/cifar10/distance_matrix.npy', dist_mat)
        print(f"Time to generate distance matrix: {datetime.now() - t_start}")
        return dist_mat