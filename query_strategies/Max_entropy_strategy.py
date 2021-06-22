import numpy as np
from .Strategy import Strategy
import torch

class Max_Entropy_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)

    def query(self, num_query):    
        idx_ulb = self.ALD.index['unlabeled']
        X, Y = self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb]
        probs = self.predict_prob(X,Y)
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        return idx_ulb[U.sort()[1][:num_query]]





