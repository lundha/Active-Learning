from .strategy import Strategy
from .bayesian_utils import *
import numpy as np
import torch
import abc

class Bayesian_Sparse_Set_Strategy(Strategy):

    def __init__(self, ALD, net, args):
        super().__init__(ALD, net, args)
        self.args = args
        self.ALD = ALD
        self.model = resnet18(pretrained=False, resnet_size=84)
        self.kwargs = {'metric': 'Acc', 'feature_extractor': self.model, 'num_features': 256}
        self.cs_kwargs = {'gamma': 0}
        self.num_projections = 100
        self.optim_params = {'num_epochs': 250, 'batch_size': 200, 'initial_lr': 1e-3,
                    'weight_decay': 5e-4, 'weight_decay_theta': 5e-4,
                    'train_transform': self.args['transform'], 'val_transform': self.args['transform']}
        self.nl = self.load_nl()   
        
    def query(self, num_query):
        cs = ProjectedFrankWolfe(self.nl, self.ALD, self.num_projections, transform=self.args['transform'], **self.cs_kwargs)
        print(num_query)
        batch = cs.build(num_query)
        print(len(batch))
        return batch

    def load_nl(self):
        nl = NeuralClassification(self.ALD, self.args, **self.kwargs)
        nl.optimize(self.ALD, **self.optim_params)
        return nl

class CoresetConstruction(metaclass=abc.ABCMeta):
    def __init__(self, acq, data, posterior, **kwargs):
        """
        Base class for constructing active learning batches.
        :param acq: (function) Acquisition function.
        :param data: (ActiveLearningDataset) Dataset.
        :param posterior: (function) Function to compute posterior mean and covariance.
        :param kwargs: (dict) Additional arguments.
        """
        self.acq = acq
        self.posterior = posterior
        self.kwargs = kwargs

        self.save_dir = self.kwargs.pop('save_dir', None)
        train_idx, unlabeled_idx = data.index['train'], data.index['unlabeled']
        self.X_train, self.y_train = data.X[train_idx], data.y[train_idx]
        self.X_unlabeled, self.y_unlabeled = data.X[unlabeled_idx], data.y[unlabeled_idx]
        self.theta_mean, self.theta_cov = self.posterior(self.X_train, self.y_train, **self.kwargs)
        self.scores = np.zeros(len(self.X_unlabeled))

    def build(self, M=1, **kwargs):
        """
        Constructs a batch of points to sample from the unlabeled set.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        :return: (list of ints) Selected data point indices.
        """
        self._init_build(M, **kwargs)
        w = np.zeros([len(self.X_unlabeled), 1])
        for m in range(M):
            w = self._step(m, w, **kwargs)

        # print(w[w.nonzero()[0]])
        return w.nonzero()[0]

    @abc.abstractmethod
    def _init_build(self, M, **kwargs):
        """
        Performs initial computations for constructing the AL batch.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional arguments.
        """
        pass

    @abc.abstractmethod
    def _step(self, m, w, **kwargs):
        """
        Adds the m-th element to the AL batch. This method is also used by non-greedy, batch AL methods
        as it facilitates plotting the selected data points over time.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return:
        """
        return None 


class ProjectedFrankWolfe(object):
    def __init__(self, model, data, J, **kwargs):
        """
        Constructs a batch of points using ACS-FW with random projections. Note the slightly different interface.
        :param model: (nn.module) PyTorch model.
        :param data: (ActiveLearningDataset) Dataset.
        :param J: (int) Number of projections.
        :param kwargs: (dict) Additional arguments.
        """
        self.ELn, self.entropy = model.get_projections(data, J, **kwargs)
        squared_norm = torch.sum(self.ELn * self.ELn, dim=-1)
        self.sigmas = torch.sqrt(squared_norm + 1e-6)
        self.sigma = self.sigmas.sum()
        self.EL = torch.sum(self.ELn, dim=0)

        # for debugging
        self.model = model
        self.data = data

    def _init_build(self, M, **kwargs):
        pass  # unused

    def build(self, M=1, **kwargs):
        """
        Constructs a batch of points to sample from the unlabeled set.
        :param M: (int) Batch size.
        :param kwargs: (dict) Additional parameters.
        :return: (list of ints) Selected data point indices.
        """
        self._init_build(M, **kwargs)
        w = to_gpu(torch.zeros([len(self.ELn), 1]))
        norm = lambda weights: (self.EL - (self.ELn.t() @ weights).squeeze()).norm()
        print(f"M: {M}")
        counter = 0
        for m in range(M):
            counter += 1
            w = self._step(m, w)
        print(f"Counter: {counter}")
        print(f"Len w: {w}")
        # print(w[w.nonzero()[:, 0]].cpu().numpy())
        print('|| L-L(w)  ||: {:.4f}'.format(norm(w)))
        print('|| L-L(w1) ||: {:.4f}'.format(norm((w > 0).float())))
        print('Avg pred entropy (pool): {:.4f}'.format(self.entropy.mean().item()))
        print('Avg pred entropy (batch): {:.4f}'.format(self.entropy[w.flatten() > 0].mean().item()))
        try:
            logdet = torch.slogdet(self.model.linear._compute_posterior()[1])[1].item()
            print('logdet weight cov: {:.4f}'.format(logdet))
        except TypeError:
            pass

        return w.nonzero()[:, 0].cpu().numpy()

    def _step(self, m, w, **kwargs):
        """
        Applies one step of the Frank-Wolfe algorithm to update weight vector w.
        :param m: (int) Batch iteration.
        :param w: (numpy array) Current weight vector.
        :param kwargs: (dict) Additional arguments.
        :return: (numpy array) Weight vector after adding m-th data point to the batch.
        """
        self.ELw = (self.ELn.t() @ w).squeeze()
        scores = (self.ELn / self.sigmas[:, None]) @ (self.EL - self.ELw)
        f = torch.argmax(scores)
        gamma, f1 = self.compute_gamma(f, w)
        # print('f: {}, gamma: {:.4f}, score: {:.4f}'.format(f, gamma.item(), scores[f].item()))
        if np.isnan(gamma.cpu()):
            raise ValueError

        w = (1 - gamma) * w + gamma * (self.sigma / self.sigmas[f]) * f1
        return w

    def compute_gamma(self, f, w):
        """
        Computes line-search parameter gamma.
        :param f: (int) Index of selected data point.
        :param w: (numpy array) Current weight vector.
        :return: (float, numpy array) Line-search parameter gamma and f-th unit vector [0, 0, ..., 1, ..., 0]
        """
        f1 = torch.zeros_like(w)
        f1[f] = 1
        Lf = (self.sigma / self.sigmas[f] * f1.t() @ self.ELn).squeeze()
        Lfw = Lf - self.ELw
        numerator = Lfw @ (self.EL - self.ELw)
        denominator = Lfw @ Lfw
        return numerator / denominator, f1


if __name__ == "__main__":
    
    pass