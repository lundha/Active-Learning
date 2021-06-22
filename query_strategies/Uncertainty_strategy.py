import numpy as np
from .Strategy import Strategy
import torch

class Uncertainty_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)

    def query(self, num_query):    
        idx_ulb = self.ALD.index['unlabeled']
        X, Y = self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb]
        P = self.predict_prob(X,Y)
        # .max(1) picks the highest element from each array and puts it into a list 
        U = P.max(1)[0]
        return idx_ulb[U.sort()[1][:num_query]]





