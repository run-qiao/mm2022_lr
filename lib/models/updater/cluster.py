from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from lshash.lshash import LSHash

class KMeansCluster():
	def __init__(self, start_len=10, feature_bank_size=100, score_bank_size=30, max_components=5, start_components=1,
	             main_cluster_num=5, cluster_thres=1.8):

		self.start_components = start_components
		self.features = []
		self.labels = []
		self.ids = []

		self.max_components = max_components
		self.score_bank_size = score_bank_size
		self.feature_bank_size = feature_bank_size
		self.score_bank = deque([], maxlen=self.score_bank_size)
		self.pose_add_thres = 0.6
		self.refresh_thres = 0.7
		self.cluster_thres = cluster_thres
		self.main_cluster_num = main_cluster_num
		self.start_len = start_len
		self.pre_score = None
		self.cluster_class = None
		self.cluster_class_is_init = False
		# PCA
		self.pca = PCA(n_components=50)

	def _create_class(self):
		return KMeans(n_clusters=self.start_components)

	def set_gt(self, gt):
		self.cluster_class_is_init = False
		self.cluster_class = self._create_class()
		self.features = []
		self.labels = []
		self.prob = []
		self.ids = []
		self.score_bank = deque([], maxlen=self.score_bank_size)

		gt = gt.flatten()
		self.features.append(gt)
		self.ids.append(0)

	# 当features满了之后，淘汰feature的方式
	def _replace_bank(self,id=1):
		# FIFO
		self.features.pop(id)
		self.ids.pop(id)

		if self.cluster_class_is_init:
			if not isinstance(self.labels, list):
				self.labels = self.labels.tolist()
			if not isinstance(self.prob, list):
				self.prob = self.prob.tolist()
			if len(self.labels) > 1 and len(self.prob) > 1:
				self.labels.pop(id)
				self.prob.pop(id)

	# return the id of feature
	def _return_id(self):
		if self.cluster_class_is_init:
			if len(self.ids) > self.pca.n_components:
				x = self.pca.transform(self._features_to_tensor())
			else:
				x = self._features_to_tensor()
			dist = self.cluster_class.transform(x)
			self.prob = np.zeros(len(dist))
			self.compute_score(dist)
			convert_inx = (np.array(self.labels) == self.labels[-1])
			cluster_P = self.prob[convert_inx]
			if len(cluster_P < self.cluster_thres) > self.main_cluster_num:
				return np.array(self.ids)[convert_inx][cluster_P < self.cluster_thres]
		return None

	def compute_score(self, dist):
		for i in range(10):
			index = np.where(self.labels == i)
			if len(index[0]) == 0:
				continue
			temp = dist[index, i]
			f = np.min(dist[index, i])
			if f != 0:
				self.prob[index] = temp / f
			else:
				self.prob[index] = 1

	def update(self, id, feature, is_update=False):

		feature = feature.flatten()
		self.ids.append(id)
		self.features.append(feature)
		return_id = None

		if is_update:
			self._refresh_best()
			return_id = self._return_id()

			# 等后面调整box之后再加进来
			self._replace_bank(-1)

		# features length > feature_bank_size
		if len(self.features) > self.feature_bank_size:
			self._replace_bank()

		return return_id

	def _refresh_class(self, x=None, visual_model=False):
		if visual_model:
			if len(self.ids) > self.start_components:
				if len(self.ids) > self.pca.n_components:
					self.labels = self.cluster_class.fit_predict(self.pca.fit_transform(self._features_to_tensor()))
				else:
					self.labels = self.cluster_class.fit_predict(self._features_to_tensor())
			else:
				self.labels=np.ones_like(self.ids)
				self.labels[0]=0
		elif x is not None:
			self.labels = self.cluster_class.fit_predict(x)

	def _refresh_best(self):

		self.cluster_class.n_clusters = self.start_components
		if len(self.ids) > self.pca.n_components:
			x = self.pca.fit_transform(self._features_to_tensor())
		else:
			x = self._features_to_tensor()
		if len(self.ids)>self.start_components:
			self._refresh_class(x)
			self.cluster_class_is_init =True
		else:
			self.cluster_class_is_init=False
			return
		if len(self.ids)>self.max_components+1:
			score = calinski_harabasz_score(x, self.labels)
			cursor = self.start_components
			for i in range(self.start_components + 1, self.max_components + 1):
				self.cluster_class.n_clusters = i
				self._refresh_class(x)
				temp_score = calinski_harabasz_score(x, self.labels)
				if temp_score > score:
					score = temp_score
					cursor = i
			if cursor == self.max_components:
				return
			else:
				self.cluster_class.n_clusters = cursor
				self._refresh_class(x)
				return

	def get_labels(self):
		if len(self.ids) == len(self.labels):
			if len(self.ids)==len(self.prob):
				return self.ids, self.labels, self.prob
			return self.ids, self.labels, np.ones_like(self.labels)
		else:
			self._refresh_class(visual_model=True)
			return self.ids, self.labels, np.ones_like(self.labels)

	def _features_to_tensor(self):
		return np.array(self.features)

	def get_mixture_feature(self, X):
		return X.mean(0)


class LSHashCluster():
	def __init__(self, bit_size=15, start_len=15, feature_bank_size=100):

		self.features = []
		self.labels = []
		self.ids = []

		self.bit_size = bit_size
		self.feature_size = None

		self.cluster_class = None
		self.cluster_class_is_init = False

		self.start_len = start_len
		self.feature_bank_size = feature_bank_size

	def _create_class(self):
		return LSHash(self.bit_size, self.feature_size)

	def _features_to_tensor(self):
		return np.array(self.features)

	def set_gt(self, gt):

		self.features = []
		self.labels = []
		self.prob = []
		self.ids = []

		gt = gt.flatten()
		self.feature_size = gt.shape[-1]
		self.features.append(gt)
		self.ids.append(0)
		self.cluster_class = self._create_class()
		self.cluster_class.index(self.features[0], self.ids[0])

	def get_labels(self):
		return self.ids, self.labels, self.prob

	def update(self, id, feature, is_update=False):

		feature = feature.flatten()
		self.ids.append(id)
		self.features.append(feature)
		self.cluster_class.index(feature, id)
		return_id = -1

		if is_update:
			# first fit, start_len == features length
			if len(self.features) >= self.start_len and not self.cluster_class_is_init:
				self._refresh_class()
				self.cluster_class_is_init = True
			# start_len < features length
			elif len(self.features) > self.start_len:
				self._refresh_class()
				return_id = self._return_id()

		# features length > feature_bank_size
		if len(self.features) > self.feature_bank_size:
			self._replace_bank()
		return return_id

	def _refresh_class(self):
		self.labels = np.zeros(len(self.ids))
		self.prob = np.zeros(len(self.ids))
		id_set = set()
		cursor = 0
		for id, feature in zip(self.ids, self.features):
			if id not in id_set:
				id_set.add(id)
				result = self.cluster_class.query(feature)
				temp_id = self.ids.index(id)
				self.labels[temp_id] = cursor
				self.prob[temp_id] = 1
				for i in result[1:]:
					idx = i[0][1]
					if idx in self.ids:
						label_idx = self.ids.index(idx)
						self.labels[label_idx] = cursor
						id_set.add(idx)
						self.prob[label_idx] = 1
				cursor += 1

	# 当features满了之后，淘汰feature的方式
	def _replace_bank(self):
		# FIFO
		self.features.pop(1)
		self.ids.pop(1)
		if self.cluster_class_is_init:

			if not isinstance(self.labels, list):
				self.labels = self.labels.tolist()
			self.labels.pop(1)
			if not isinstance(self.prob, list):
				self.prob = self.prob.tolist()
			self.prob.pop(1)

	# return the id of feature
	def _return_id(self):

		result = self.cluster_class.query(self.features[-1])

		if len(result) > 1:
			return result[1][0][1]
		return -1


def build_cluster(cfg):
	return KMeansCluster()


def exemplify():
	gmm = build_cluster()
	feature_dim = 768
	gt_feat = torch.Tensor(np.random.randn(feature_dim).flatten())
	gmm.set_gt(gt_feat)
	stop = True
	component = 8
	num = 300
	state = np.zeros(component, dtype=int)
	x, y = make_blobs(n_samples=[num] * component, n_features=feature_dim, shuffle=False)
	labels = []
	id = 1
	while stop:
		rng_idx = np.random.randint(low=0, high=component)
		state[rng_idx] += 1

		if state[rng_idx] > num:
			state[rng_idx] = num

		X = x[rng_idx * num + state[rng_idx] - 1:rng_idx * num + state[rng_idx], :]
		X = torch.Tensor(X.flatten())
		gmm.update(id, X)
		id += 1

		labels.append(gmm.pre_score)
		if sum(state) == component * num:
			stop = False

	dots = range(len(labels))
	plt.figure()
	plt.plot(dots, labels)
	plt.show()


if __name__ == '__main__':
	exemplify()
