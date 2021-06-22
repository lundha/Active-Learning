from kcenter_greedy import KCenterGreedy
import numpy as np
from gurobi_solver import gurobi_solver
from .Strategy import *
import pickle
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb



class Coreset_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)
        self.tor = 1e-4
        self.already_selected = []
        self.n_pool = len(self.ALD.index['unlabeled'])
        self.embedding_dim = net.get_embedding_dim()

    def query(self, num_query):

        idx_ulb = self.ALD.index['unlabeled']
        loader = self.prepare_loader(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'], self.args['tr_args'])
        embedding = self.get_embedding(loader, self.embedding_dim)
        #greedy_idx = self.find_k_means_pp(embedding, num_query)
        print(embedding)
        dist_mat = self.calculate_distance_matrix(embedding)
        greedy_idx, min_dist = self.find_greedy_solution(dist_mat, num_query)
        #opt_idx = self.find_optimal_solution(dist_mat, min_dist, num_query, idx_ulb)
        opt_idx = greedy_idx   
        
        return opt_idx

    # Find greedy solution
    def find_greedy_solution(self, dist_mat, num_query):
        kcg = KCenterGreedy(dist_mat)
        idx = kcg.select_batch(self.already_selected, num_query)
        return idx, kcg.min_distances


    # Optimize clusters
    def find_optimal_solution(self, dist_mat, min_distances, num_query, idx_ulb):
        
        opt = min_distances.min(axis=1).max() 
        bound_l = opt/2.0
        delta = opt
        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]
        subset = [i for i in range(0)]
        self.logger.debug(f"Arguments: {subset}, {float(opt)}, {num_query}, {len(idx_ulb)}")
        #pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), NUM_QUERY, n_pool), open('mip{}.pkl'.format(SEED), 'wb'), 2)
        sols = gurobi_solver(xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), num_query, len(idx_ulb))
        #sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))
        return sols