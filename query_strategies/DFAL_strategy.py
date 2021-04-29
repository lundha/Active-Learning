import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy


class DFAL(Strategy):
    def __init__(self, ALD, net, args):
        super().__init__(ALD, net, args)
        self.args = args
        self.ALD = ALD
        self.max_iter = 50
        
    def query(self, num_query):

        idx_ulb = self.ALD.index['unlabeled']


        self.classifier.cpu()
        
        self.classifier.eval()
        print(f"Shape unlabaled samples: {idx_ulb.shape}")
        dis = np.zeros(idx_ulb.shape)

        handler = self.prepare_handler(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'])

        for i in range(len(idx_ulb)):
            if i % 100 == 0:
                print('adv {}/{}'.format(i, len(idx_ulb)))
            x, y, idx = handler[i]
            dis[i] = self.cal_dis(x)

        self.classifier.cuda()

        # argsort() returns the indices that would sort an array. Since the index of the dis() and element i in 
        # idx_ulb have 1-to-1 correspondence, it implicitly returns the idx of the unlabeled sample.
        return idx_ulb[dis.argsort()[:num_query]]        

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
            ri = torch.zeros(nx.shape)

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

            eta += ri.clone()
            nx.grad.data.zero_()
            out, _ = self.classifier(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()



