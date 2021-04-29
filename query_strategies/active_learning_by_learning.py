import numpy as np
from .strategy import Strategy

class ActiveLearningByLearning(Strategy):
	def __init__(self,ALD, net, args, strategy_list, delta = 0.1):
		super(ActiveLearningByLearning, self).__init__(ALD, net, args)
		self.strategy_list = strategy_list
		#self.strategy_list.append(RandomSampling(X, Y, idxs_lb, args))
		self.n_strategy = len(self.strategy_list)
		self.delta = delta
		self.idxs_lb = self.ALD.index['labeled']
		self.w = np.ones((self.n_strategy, ))
		self.pmin = 1.0 / (self.n_strategy * 10.0)
		self.start = True
		self.aw = np.zeros((len(self.ALD.Y)))
		self.aw[self.idxs_lb] = 1.0

	def query(self, n):
		if not self.start:
			#idxs_labeled = self.ALD.index['labeled']
			idxs_labeled = self.idxs_lb #np.arange(self.n_pool)[self.idxs_lb]

			P = self.predict(self.ALD.X[idxs_labeled], self.ALD.Y[idxs_labeled])
			fn = (P.numpy() == self.ALD.Y[idxs_labeled].numpy()).astype(float)
			reward = (fn / self.aw[self.idxs_lb]).mean()

			self.w[self.s_idx] *= np.exp(self.pmin/2.0 * (reward + 1.0 / self.last_p * np.sqrt(np.log(self.n_strategy / self.delta) / self.n_strategy)))

		self.start = False
		W = self.w.sum()
		p = (1.0 - self.n_strategy * self.pmin) * self.w / W + self.pmin

		for i, stgy in enumerate(self.strategy_list):
			print('  {} {}'.format(p[i], type(stgy).__name__))

		self.s_idx = np.random.choice(np.arange(self.n_strategy), p=p)
		print('  select {}'.format(type(self.strategy_list[self.s_idx]).__name__))
		self.strategy_list[self.s_idx].classifier = self.classifier
		q_idxs = self.strategy_list[self.s_idx].query(n)
		self.aw[q_idxs] = p[self.s_idx]
		self.last_p = p[self.s_idx]

		return q_idxs

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb
		for i, stgy in enumerate(self.strategy_list):
			stgy.update(idxs_lb)
