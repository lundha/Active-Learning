import numpy as np
from .Strategy import Strategy
from sklearn.cluster import KMeans

class KMeans_Strategy(Strategy):
	def __init__(self, ALD, net, args, logger, **kwargs):
		super().__init__(ALD, net, args, logger)
		self.embedding_dim = net.get_embedding_dim()

	def query(self, n):
		idx_ulb = self.ALD.index['unlabeled']
		loader = self.prepare_loader(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'], self.args['tr_args'])
		embedding = self.get_embedding(loader, self.embedding_dim)
		cluster_learner = KMeans(n_clusters=n, init='k-means++')
		cluster_learner.fit(embedding)
		
		cluster_idxs = cluster_learner.predict(embedding)
		centers = cluster_learner.cluster_centers_[cluster_idxs]
		dis = (embedding - centers)**2
		dis = dis.sum(axis=1)
		q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
		
		#q_idxs = self.init_centers(embedding, n)
		
		return q_idxs
