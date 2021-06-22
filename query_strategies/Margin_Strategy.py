import numpy as np
from .Strategy import Strategy

class MarginSampling_Strategy(Strategy):
	def __init__(self, ALD, net, args, logger, **kwargs):
		super().__init__(ALD, net, args,)

	def query(self, n):
		#idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		idxs_unlabeled = self.ALD.index['unlabeled']
		probs = self.predict_prob(self.ALD.X[idxs_unlabeled], self.ALD.Y[idxs_unlabeled])
		probs_sorted, idxs = probs.sort(descending=True)
		U = probs_sorted[:, 0] - probs_sorted[:,1]
		return idxs_unlabeled[U.sort()[1][:n]]
