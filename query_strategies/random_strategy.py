import numpy as np
from .strategy import Strategy
import torch


class Random_Strategy(Strategy):
    def __init__(self, ALD, net, args):
        super().__init__(ALD, net, args)
        self.ALD = ALD

    def query(self, num_query):
        idx_ulb = self.ALD.index['unlabeled']
        query = np.random.choice(idx_ulb, num_query, replace=False)
        return query
