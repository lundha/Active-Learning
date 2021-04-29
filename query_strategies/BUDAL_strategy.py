from kcenter_greedy import KCenterGreedy
import numpy as np
from gurobi_solver import gurobi_solver
from .strategy import Strategy
import pickle
import torch

class BUDAL(Strategy):
    def __init__(self, ALD, net, args,tor=1e-4):
        super().__init__(ALD, net, args)
    
        self.args = args
        self.ALD = ALD
        self.blending_constant = 1
        self.embedding_dim = net.get_embedding_dim()
        self.already_selected = []
        self.tor = tor
        self.max_iter = 50
 
    def query(self, num_query):
        '''
        Budal query method
        param num_query: number of samples to be queried for each round
        '''
        
        uncertain_samples, _ = self.uncertain_query()
        
        split = int(len(uncertain_samples) * self.blending_constant)

        blended_uncertain_list = uncertain_samples[0:split]
        budal_samples = self.diverse_query(num_query, blended_uncertain_list)
        self.blending_constant = self.blending_constant*0.9
        
        return budal_samples
        # Use core set to find num_query clusters from list * blending constant (0.0 - 1.0)
        # Return samples

    def query2(self, num_query):
        pass
    
    def diverse_query(self, num_query, samples):
        loader = self.prepare_loader(self.ALD.X[samples], self.ALD.Y[samples], self.args['transform'], self.args['tr_args'])
        embedding = self.get_embedding(loader, self.embedding_dim)
        dist_mat = self.calculate_distance_matrix(embedding)
        greedy_idx, _ = self.find_greedy_solution(dist_mat, num_query)
        #opt_idx = self.find_optimal_solution(dist_mat, min_dist, num_query, n_pool)
        opt_idx = greedy_idx   
        return opt_idx
    
    def k_means_pp(self):
        pass

    def uncertain_query(self):
        
        idx_ulb = self.ALD.index['unlabeled']

        self.classifier.cpu()
        
        self.classifier.eval()
        dis = np.zeros(idx_ulb.shape)

        handler = self.prepare_handler(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'])

        for i in range(len(idx_ulb)):
            if i % 100 == 0:
                print('adv {}/{}'.format(i, len(idx_ulb)))
            x, _, _ = handler[i]
            dis[i] = self.cal_dis(x)

        self.classifier.cuda()

        # argsort() returns the indices that would sort an array. Since the index of the dis() and element i in 
        # idx_ulb have 1-to-1 correspondence, it implicitly returns the idx of the unlabeled sample.
        return idx_ulb[dis.argsort()], dis.argsort()               
        
    # Find greedy solution
    def find_greedy_solution(self, dist_mat, num_query):
        
        kcg = KCenterGreedy(dist_mat)
        idx = kcg.select_batch(self.already_selected, num_query)
        return idx, kcg.min_distances
        
    def cal_dis(self, x):
        
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, _ = self.classifier(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = torch.zeros(nx.shape) # 0 instead of None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())
  
                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi
            try:
                eta += ri.clone()
            except AttributeError as e:
                print(f"AttributeError: {str(e)}")
                continue

            nx.grad.data.zero_()
            out, _ = self.classifier(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1
        
        return (eta*eta).sum()     
