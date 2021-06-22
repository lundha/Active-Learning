
import torch
from torch import nn
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from copy import deepcopy
import pdb
import sys
import numpy as np
import pickle
from scipy.spatial.distance import cosine
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
import torchfile
import resnet
import argparse
from collections import OrderedDict
from scipy import stats
import time
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from config import DATASET, STRATEGY


class Strategy():
    def __init__(self, ALD, net, args, logger):
        self.ALD = ALD
        self.net = net
        self.args = args
        self.logger = logger
        self.n_pool = len(ALD.index['unlabeled'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def query(self):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        
        self.classifier.train()
        acc, val_loss = 0., 0.

        for _, (x, y, _) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, _, self.ll_features = self.classifier(x)
            loss = self.criterion(out, y)

            val_loss += torch.sum(loss.data)  / loss.size(0)
            acc += torch.sum((torch.max(out,1)[1] == y).float()).data.item() / loss.size(0)

            loss.sum().backward()
            optimizer.step()

        return val_loss, acc

    
    def train(self):

        self.classifier.train()
        n_epoch = self.args['n_epoch']
        idx_lb = self.ALD.index['labeled']
        X_valid, Y_valid = self.ALD.X_valid, self.ALD.Y_valid
        optimizer = optim.Adam(self.net.parameters(), lr=self.args['tr_args']['lr'], weight_decay=self.args['tr_args']['weight_decay'])
        loader_tr = self.prepare_loader(self.ALD.X[idx_lb], self.ALD.Y[idx_lb], self.args['transform'], self.args['tr_args'])
        prev_loss, curr_loss, counter = 0, 0, 0
        

        for epoch in range(n_epoch):
            prev_loss = curr_loss

            tr_loss, tr_acc = self._train(epoch, loader_tr, optimizer)
            P = self.predict(X_valid, Y_valid)
            curr_loss, valid_acc, P = self.val_loss_accuracy(X_valid, Y_valid)
            valid_acc = round(1.0 * (Y_valid==P).sum().item() / len(Y_valid),4)

            self.logger.debug(f"{str(epoch)} Training accuracy: {str(tr_acc)}, Valid accuracy: {str(valid_acc)}, "
                    f"Valid loss: {str(round(curr_loss,3))}, Training loss: {str(round(tr_loss.item(),3))}")

            if curr_loss > prev_loss and STRATEGY != 'ALL-DATA':
                counter += 1
                if counter > 5:
                    self.logger.debug('Validation loss keeps increasing, stop training.')
                    break
            else:
                counter = 0
        

    def val_loss_accuracy(self, X: list, Y: list) -> (float, torch.Tensor):
        loader_val = self.prepare_loader(X, Y, self.args['transform'], self.args['valid_args'], drop_last=False)
        self.classifier.eval()
        Y = Y.type(torch.LongTensor)
        P = torch.zeros(len(Y), dtype=Y.dtype)

        val_acc, val_loss = 0. , 0.
        with torch.no_grad():
            for x, y, idx in loader_val:
                x, y = x.to(self.device), y.to(self.device)
                out, _, _ = self.classifier(x)
                loss = self.criterion(out, y)
                val_loss = loss.sum() / loss.size(0)
                val_acc += torch.sum((torch.max(out,1)[1] == y).float()).data.item() / loss.size(0)
                pred = out.max(1)[1]
                P[idx] = pred.cpu()

        return float(val_loss), val_acc, P


    def predict(self, X, Y):
        #X, Y = self.ALD.X_test, self.ALD.Y_test
        loader_te = self.prepare_loader(X, Y, self.args['transform'], self.args['te_args'], drop_last=False)
        self.classifier.eval()
        Y = Y.type(torch.LongTensor)    
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            try:
                for x,y,idx in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, _, _ = self.classifier(x)
                    pred = out.max(1)[1]
                    P[idx] = pred.cpu()
            except Exception as e:
                self.logger.error(f'Exception: {str(e)}')
        return P

    def predict_prob(self, X, Y):
        loader_te = self.prepare_loader(X, Y, self.args['transform'], self.args['te_args'])

        self.classifier.eval()
        probs = torch.zeros([len(Y), self.args['num_classes']])

        with torch.no_grad():
            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)
                out, _, _ = self.classifier(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs


    def get_embedding(self, dataloader, embedding_dim):
        '''
        Create and save embedding 
        :param: dataloader:dataloader, embedding_din:embedding dimension 
        '''
        print(f"embedding dim: {embedding_dim}")
        encoder = Autoencoder(embedding_dim, self.args['img_dim'])
        encoder.load_state_dict(torch.load(f'{DATASET}_weights.pt'))
        embedding = torch.zeros([len(dataloader.dataset), embedding_dim])
        encoder = encoder.to(self.device)
        with torch.no_grad():
            for x, _, idxs in dataloader:
                x = x.to(self.device)
                _, e1 = encoder(x)
                embedding[idxs] = e1.cpu()
        #np.save(f'./{DATASET}_EMBEDDING.npy', embedding)
        embedding = embedding.numpy()
        #embedding = np.load(f'./tsne_plots_new/{DATASET}_fc1_features.npy')
        print(embedding.shape)
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
        np.save(f'./{DATASET}_DIST_MATRIX.npy', dist_mat)
        self.logger.debug(f"Time taken to generate distance matrix: {datetime.now() - t_start}")
        return dist_mat

    def prepare_handler(self, X_unlabeled, Y_unlabeled, transform):
        return DataHandler(X_unlabeled, Y_unlabeled, transform)


    def prepare_loader(self, X, Y, transform, args, shuffle=False, drop_last=True, custom_batch_size=None):
        '''
        Creates a dataloader from unlabeled X and Y data points
        :param: X_unlabeled: Unlabeled datapoints Y_unlabeled: Unlabeled labels
        '''
        handler = DataHandler(X, Y, transform)
        if custom_batch_size is not None:
            loader = DataLoader(handler, shuffle=shuffle, drop_last=drop_last, batch_size=custom_batch_size, num_workers=args['num_workers'])
        else:
            loader = DataLoader(handler, shuffle=shuffle, drop_last=drop_last, batch_size=args['batch_size'], num_workers=args['num_workers'])
        return loader

    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y):
        model = self.classifier
        embDim = model.get_embedding_dim()

        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = self.prepare_loader(X, Y, self.args['transform'], self.args['te_args'])

        #loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
        #                    shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out, _ = self.classifier(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)



    # kmeans ++ initialization
    def init_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        self.logger.debug('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            self.logger.debug(str(len(mu)) + '\t' + str(sum(D2)))
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2)/ sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        gram = np.matmul(X[indsAll], X[indsAll].T)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        vgt = val[val > 1e-2]
        return indsAll
    