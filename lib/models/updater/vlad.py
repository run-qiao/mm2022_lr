# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : vlad.py
# Copyright (c) Skye-Song. All Rights Reserved
import numpy as np
from sklearn.cluster import KMeans


class VLAD():
	def __init__(self, k=10):
		self.k = k

	def forward(self, X):
		k = self.k
		# X = X.squeeze(0).detach().cpu().numpy()
		visualDictionary = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=0).fit(X)
		predictedLabels = visualDictionary.labels_
		centers = visualDictionary.cluster_centers_
		# k = visualDictionary.n_clusters

		m, d = X.shape
		V = np.zeros([k, d])
		# for all the clusters (visual words)
		for i in range(k):
			# if there is at least one descriptor in that cluster
			if np.sum(predictedLabels == i) > 0:
				# add the diferences
				V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

		V = V.flatten()
		# power normalization, also called square-rooting normalization
		V = np.sign(V) * np.sqrt(np.abs(V))

		# L2 normalization
		V = V / np.sqrt(np.dot(V, V))

		return V


def build_vlad(cfg):
	return VLAD(k=10)


# a = np.random.randn(196, 768)
# get_VLAD(a)
