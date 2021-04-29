from kcenter_greedy import KCenterGreedy
import numpy as np
from gurobi_solver import gurobi_solver
from .strategy import Strategy
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
from torch import nn
import torchfile
from torch.autograd import Variable
import resnet
import torch.optim as optim
import pdb
from torch.nn import functional as F
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


class Softmax_Hybrid_Strategy(Strategy):
    def __init__(self, ALD, net, args,tor=1e-4):
        super().__init__(ALD, net, args)
    
        self.args = args
        self.ALD = ALD
        self.blending_constant = 1
        self.embedding_dim = net.get_embedding_dim()
        self.already_selected = []
        self.tor = tor
        self.max_iter = 50
 
    def query(self, num_query):

        # Deep Fool on n_pool
        # Return sorted list from most uncertain to least uncertain
        uncertain_samples = self.uncertain_query()
        blended_uncertain_list = uncertain_samples[0:int(len(uncertain_samples) * self.blending_constant)]
        
        # 0.9 is the blending delta
        self.blending_constant = self.blending_constant*0.9
        CIRAL_samples = self.diverse_query(num_query, blended_uncertain_list)

        return CIRAL_samples

    
    def diverse_query(self, num_query, samples):
        #idx_ulb = self.ALD.index['unlabeled']

        loader = self.prepare_loader(self.ALD.X[samples], self.ALD.Y[samples], self.args['transform'], self.args['tr_args'])
        embedding = self.get_embedding(loader, self.embedding_dim)
        dist_mat = self.calculate_distance_matrix(embedding)
        greedy_idx, _ = self.find_greedy_solution(dist_mat, num_query)
        #opt_idx = self.find_optimal_solution(dist_mat, min_dist, num_query, n_pool)
        #greedy_idx = init_centers(embedding, num_query)
        opt_idx = greedy_idx   
        return opt_idx
    
    # kmeans ++ initialization
    def init_centers(X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] >  newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
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



    def uncertain_query(self):
    
        idx_ulb = self.ALD.index['unlabeled']
        X, Y = self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb]
        P = self.predict_prob(X,Y)
        # .max(1) picks the highest element from each array and puts it into a list 
        U = P.max(1)[0]
        return idx_ulb[U.argsort()]

        
    # Find greedy solution
    def find_greedy_solution(self, dist_mat, num_query):
        
        kcg = KCenterGreedy(dist_mat)
        idx = kcg.select_batch(self.already_selected, num_query)
        return idx, kcg.min_distances
        
