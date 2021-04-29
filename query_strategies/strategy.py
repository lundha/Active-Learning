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
from torch.utils.data import DataLoader
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

class Strategy():
    def __init__(self, ALD, net, args):
        self.ALD = ALD
        self.net = net
        self.args = args
        self.n_pool = len(ALD.index['unlabeled'])
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.classifier = self.net.to(self.device)

    def query(self):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb


    def _train(self, epoch, loader_tr, optimizer):
        self.classifier.train()
        accFinal = 0.
        val_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.classifier(x)
            loss = F.cross_entropy(out, y)
            val_loss += loss.item()
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            optimizer.step()
        val_loss /= len(loader_tr.dataset)
        return accFinal / len(loader_tr.dataset.X), val_loss

    
    def train(self, X_valid, Y_valid):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        n_epoch = self.args['n_epoch']
        self.classifier.train()
        optimizer = optim.Adam(self.net.parameters(), lr=self.args['tr_args']['lr'], weight_decay=self.args['tr_args']['weight_decay'])

        idx_lb = self.ALD.index['labeled']
        loader_tr = self.prepare_loader(self.ALD.X[idx_lb], self.ALD.Y[idx_lb], self.args['transform'], self.args['tr_args'])
        epoch = 1
        accPrev = 0.
        valid_acc = 0.
        counter = 0

        while valid_acc < 0.95: 
            accPrev = valid_acc
            tr_acc, tr_loss = self._train(epoch, loader_tr, optimizer)
            P = self.predict(X_valid, Y_valid)
            valid_acc = round(1.0 * (Y_valid==P).sum().item() / len(Y_valid),4)
            loss = self.evaluate_val(X_valid, Y_valid)
    
            epoch += 1
            print(str(epoch) + ' training accuracy: ' + str(round(tr_acc,3)) + ' val accuracy: ' + str(round(valid_acc,3)) + \
                        ' valid loss: ' + str(round(loss,3)) + ' training loss: ' + str(round(tr_loss,3)), flush=True)

            if accPrev == valid_acc:
                counter += 1
                if counter > 10:
                    print('Training has converged')
                    break
            else:
                counter = 0
        
            if epoch > 50:
                print('Training will not converge')
                break

            if (epoch % 50 == 0) and (valid_acc < 0.2): # reset if not converging
                self.classifier = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.classifier.parameters(), lr = 0.05, weight_decay=0)


    def evaluate_val(self, Xte, Yte):
        loader_te = self.prepare_loader(Xte, Yte, self.args['transform'], self.args['valid_args'])
        self.classifier.eval()
        Yte = Yte.type(torch.LongTensor)
        loss = 0

        with torch.no_grad():
            for x,y,idx in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.classifier(x)
                loss += F.cross_entropy(out, y, reduction="sum").item()

            loss /= len(Yte)
        return loss



    def predict(self, Xte, Yte):
        loader_te = self.prepare_loader(Xte, Yte, self.args['transform'], self.args['valid_args'])
        self.classifier.eval()
        Yte = Yte.type(torch.LongTensor)
      
        try:
            P = torch.zeros(len(Yte), dtype=Yte.dtype)
        except Exception as e:
            print(f"Exception: {str(e)}")
            P = np.zeros(len(Yte), dtype=Yte.dtype)
        with torch.no_grad():
            try:
                for x,y,idx in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, _ = self.classifier(x)
                    pred = out.max(1)[1]
                    P[idx] = pred.cpu()
            except Exception as e:
                print(f'Exception: {str(e)}')
        return P

    def predict_prob(self, X, Y):
        loader_te = self.prepare_loader(X, Y, self.args['transform'], self.args['te_args'])

        self.classifier.eval()
        try:
            probs = torch.zeros([len(Y), len(np.unique(Y))])
        except Exception as e:
            print(f"Exception: {str(e)}")
            P = np.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.classifier(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs


    def get_embedding(self, dataloader, embedding_dim):
        '''
        Create and save embedding 
        :param: dataloader:dataloader, embedding_din:embedding dimension 
        '''
        
        encoder = Autoencoder(embedding_dim, self.args['img_dim'])
        #encoder = self.classifier
        data_set = self.args['data_set']
        embedding = torch.zeros([len(dataloader.dataset), embedding_dim])
        with torch.no_grad():
            for x, y, idxs in dataloader:
                #x, y = x.to(self.device), y.to(self.device)
                out, e1 = encoder(x)
                embedding[idxs] = e1.cpu()
        np.save(f'./{data_set}_EMBEDDING.npy', embedding)
        embedding = embedding.numpy()
        #embedding = np.load('/home/martlh/masteroppgave/core-set/tsne_plots/plankton_fc1_features.npy')
        
        return embedding


    def calculate_distance_matrix(self, embedding) -> np.array:
        '''
        Calculate and save distance matrix, input is embedding
        '''
        data_set = self.args['data_set']
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(embedding), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        np.save(f'./{data_set}_DIST_MATRIX.npy', dist_mat)
        print(f"Time taken to generate distance matrix: {datetime.now() - t_start}")
        return dist_mat

    def prepare_handler(self, X_unlabeled, Y_unlabeled, transform):
        return DataHandler(X_unlabeled, Y_unlabeled, transform)


    def prepare_loader(self, X_unlabeled, Y_unlabeled, transform, args):
        '''
        Creates a dataloader from unlabeled X and Y data points
        :param: X_unlabeled: Unlabeled datapoints Y_unlabeled: Unlabeled labels
        '''
        handler = DataHandler(X_unlabeled, Y_unlabeled, transform)
        loader = DataLoader(handler, shuffle=False, drop_last=True, batch_size=args['batch_size'], num_workers=args['num_workers'])
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
                cout, out = self.classifier(x)
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

    