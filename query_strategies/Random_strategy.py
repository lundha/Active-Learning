import numpy as np
from .Strategy import Strategy
import torch


class Random_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)

    def query(self, num_query):
        idx_ulb = self.ALD.index['unlabeled']
        query = np.random.choice(idx_ulb, num_query, replace=False)
        return query
