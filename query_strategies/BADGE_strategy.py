import numpy as np
from torch.utils.data import DataLoader
from .Strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
import resnet
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.cluster import KMeans
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

class BADGE_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)
        
    def query(self, num_query):
        idx_ulb = self.ALD.index['unlabeled']
        gradEmbedding = self.get_grad_embedding(self.ALD.X[idx_ulb], self.ALD.Y.numpy()[idx_ulb]).numpy()
        chosen = self.init_centers(gradEmbedding, num_query)
        '''
        cluster_learner = KMeans(n_clusters=num_query, init='k-means++')
        cluster_learner.fit(gradEmbedding)
        cluster_idxs = cluster_learner.predict(gradEmbedding)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (gradEmbedding - centers)**2
        dis = dis.sum(axis=1)
        try:
            chosen = np.array([np.arange(gradEmbedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(num_query)])
        except ValueError:
            print(ValueError)
        '''
                    
        return chosen


