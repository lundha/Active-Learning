import numpy as np
from .strategy import Strategy
import torch

class Max_Entropy_Strategy(Strategy):
    def __init__(self, ALD, net, args):
        super().__init__(ALD, net, args)
        self.ALD = ALD 

    def query(self, num_query, n_pool):    
        idx_ulb = self.ALD.index['unlabeled']
        X, Y = self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb]
        probs = self.predict_prob(X,Y)
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        return idx_ulb[U.sort()[1][:num_query]]




