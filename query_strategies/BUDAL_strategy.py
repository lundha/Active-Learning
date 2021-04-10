from kcenter_greedy import KCenterGreedy
import numpy as np
from gurobi_solver import gurobi_solver
from .strategy import Strategy
import pickle
import torch


class BUDAL(Strategy):
    def __init__(self, ALD, net, args, tor=1e-4):
        super().__init__(ALD, net, args)
        self.args = args
        self.ALD = ALD
        self.net = net

    def query(self, num_query, n_pool):

        # Deep Fool on n_pool
        # Return sorted list from most uncertain to least uncertain
        uncertain_samples = DeepFool(n_pool)

        blended_uncertain_list = uncertain_samples[0:int(len(uncertain_samples) * blending_constant)]

        budal_samples = CoreSet(blended_uncertain_list)
        return budal_samples
        # Use core set to find num_query clusters from list * blending constant (0.0 - 1.0)
        # Return samples
        
