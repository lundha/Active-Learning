from kcenter_greedy import KCenterGreedy
import numpy as np
from gurobi_solver import gurobi_solver
from strategy import Strategy

class Coreset(Strategy):
    def __init__(self, ALD, net, args, tor=1e-4):
        super(Coreset, self).__init__(ALD, net, args)
        self.tor = tor
        self.args = args
        self.ALD = ALD
        self.already_selected = []
        self.embedding_dim = net.get_embedding_dim()

    def query(self, num_query):

        idx_ulb = self.ALD.index['unlabeled']
        loader = self.prepare_data(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'])
        embedding = self.get_embedding(loader, self.embedding_dim)
        dist_mat = self.calculate_distance_matrix(embedding)
        greedy_idx, min_dist = self.find_greedy_solution(dist_mat, num_query)
        opt_idx = self.find_optimal_solution(min_dist)

        return opt_idx

    # Find greedy solution
    def find_greedy_solution(self, dist_mat, num_query):
        kcg = KCenterGreedy(dist_mat)
        idx = kcg.select_batch(self.already_selected, num_query)
        return idx, kcg.min_distances

    # Optimize clusters
    def find_optimal_solution(self, min_distances):
        
        opt = min_distances.min(axis=1).max() 
        bound_l = opt/2.0
        delta = opt
        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]
        pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), NUM_QUERY, n_pool), open('mip{}.pkl'.format(SEED), 'wb'), 2)
        gurobi_solver()
        sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))
        return sols

 

    