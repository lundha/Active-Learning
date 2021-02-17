
import torch
from torch import nn
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from torch.utils.data import DataLoader
import torch.optim as optim
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

        for batch_idx, (x,y,idx) in enumerate(loader_tr):
            x,y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.classifier(x)
            loss = nn.CrossEntropyLoss()
            loss = (out, y)
            loss.backward()
            optimizer.step()


    def train(self):
        n_epoch = self.args['n_epoch']
        self.classifier = self.net().to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.0001)

        idx_lb = self.ALD.index['labeled']
        loader_tr = self.prepare_data(self.ALD.X[idx_lb], self.ALD.Y[idx_lb], self.args['transform'], args['loader_tr_args'])

        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)


    def predict(self, Xte, Yte):
        loader_te = self.prepare_data(Xte, Yte, args['transform'], args['loader_te_args'])
        
        with torch.no_grad():
            for x,y,idx in loader_te:
                x,y = x.to(self.device), y.to(self.device)
                out = self.classifier(x)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P


    def prepare_data(self, X_unlabeled, Y_unlabeled, transform, args):
        '''
        Creates a dataloader from unlabeled X and Y data points
        :param: X_unlabeled: Unlabeled datapoints Y_unlabeled: Unlabeled labels
        '''
        handler = DataHandler(X_unlabeled, Y_unlabeled, transform=)
        loader = DataLoader(handler, shuffle=False, batch_size=args['batch_size'], num_workers=args['num_workers'])
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
        np.save('./CIFAR10/embedding.npy', embedding)
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
        np.save('./CIFAR10/distance_matrix.npy', dist_mat)
        print(f"Time taken to generate distance matrix: {datetime.now() - t_start}")
        return dist_mat