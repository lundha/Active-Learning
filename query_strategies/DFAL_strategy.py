
import numpy as np
import torch
import torch.nn.functional as F
from .Strategy import Strategy
import PIL
from matplotlib import pyplot as plt
import sys
from random import randint

class DFAL_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)

        self.max_iter = 50
        self.count = 0
        self.prev_count = 0

    def query(self, num_query):

        idx_ulb = self.ALD.index['unlabeled']

        self.classifier.cpu()
        self.classifier.eval()
        self.logger.debug(f"Shape unlabaled samples: {idx_ulb.shape}")
        dis = np.zeros(idx_ulb.shape)

        handler = self.prepare_handler(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'])
        self.count += 1
        for i in range(len(idx_ulb)):
            if i % 100 == 0:
                self.logger.debug(f'adv {i}/{len(idx_ulb)}')
            x, _, _ = handler[i]
            dis[i] = self.cal_dis(x)

        self.classifier.cuda()

        # argsort() returns the indices that would sort an array. Since the index of the dis() and element i in 
        # idx_ulb have 1-to-1 correspondence, it implicitly returns the idx of the unlabeled sample.
        return idx_ulb[dis.argsort()[:num_query]]        

    def cal_dis(self, x):

        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out, _, _ = self.classifier(nx+eta)
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
            out, _, _ = self.classifier(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1
        ### To visualize adversarial attack ###
        visualize = False
        if self.count != self.prev_count and visualize == True:
            import matplotlib.pyplot as plt
            self.prev_count = self.count
            classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            nx = torch.squeeze(nx)
            nx_new = torch.squeeze(nx+eta)
            nx_new = nx_new.permute(1,2,0)
            nx = nx.permute(1,2,0)
            nx = nx.detach().numpy()
            nx_new = nx_new.detach().numpy()

            #mean = np.array([0.4914, 0.4822, 0.4465])
            #std = np.array([0.2470, 0.2435, 0.2616])
            mean = np.array([0.95, 0.95, 0.95])
            std = np.array([0.2, 0.2, 0.2])
            nx = std * nx + mean
            nx_new = std * nx_new + mean
            nx = np.clip(nx, 0, 1)
            nx_new = np.clip(nx_new, 0, 1)
            seed = randint(0,100)
            plt.imshow(nx)
            plt.axis('off')
            plt.title(classes[ny])
            plt.savefig(f'./dfal-imgs/nx_{self.count}_{seed}.png')
            plt.imshow(nx_new)
            plt.axis('off')
            plt.title(classes[py])
            plt.savefig(f'./dfal-imgs/nx_new_{self.count}_{seed}.png')
            print(f"py: {classes[py]}, ny: {classes[ny]}")
        
        return (eta*eta).sum()



