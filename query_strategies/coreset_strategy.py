from kcenter_greedy import KCenterGreedy
import numpy as np
from gurobi_solver import gurobi_solver
from .strategy import *
import pickle
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb



class Coreset(Strategy):
    def __init__(self, ALD, net, args, tor=1e-4):
        super().__init__(ALD, net, args)
        self.tor = tor
        self.args = args
        self.ALD = ALD
        self.already_selected = []
        self.embedding_dim = net.get_embedding_dim()

    def query(self, num_query):

        idx_ulb = self.ALD.index['unlabeled']
        loader = self.prepare_loader(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'], self.args['tr_args'])
        embedding = self.get_embedding(loader, self.embedding_dim)
        #greedy_idx = self.find_k_means_pp(embedding, num_query)
        dist_mat = self.calculate_distance_matrix(embedding)
        greedy_idx, min_dist = self.find_greedy_solution(dist_mat, num_query)
        #opt_idx = self.find_optimal_solution(dist_mat, min_dist, num_query, n_pool)
        opt_idx = greedy_idx   
        
        return opt_idx

    # Find greedy solution
    def find_greedy_solution(self, dist_mat, num_query):
        kcg = KCenterGreedy(dist_mat)
        idx = kcg.select_batch(self.already_selected, num_query)
        return idx, kcg.min_distances


    def find_k_means_pp(self, X, K):
        '''
        X: Embedding
        K: Number of queries
        '''
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

    # Optimize clusters
    def find_optimal_solution(self, dist_mat, min_distances, num_query, n_pool):
        
        opt = min_distances.min(axis=1).max() 
        bound_l = opt/2.0
        delta = opt
        xx, yy = np.where(dist_mat <= opt)
        dd = dist_mat[xx, yy]
        subset = [i for i in range(0)]
        print(f"Arguments: {subset}, {float(opt)}, {num_query}, {n_pool}")
        #pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), NUM_QUERY, n_pool), open('mip{}.pkl'.format(SEED), 'wb'), 2)
        sols = gurobi_solver(xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), num_query, n_pool)
        #sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))
        return sols