# -*- coding=utf-8 -*-
# @Time :2022/3/30 14:39
# @Author :Run Luo
# @Site : 
# @File : test.py
# @Software : PyCharm


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score
import numpy as np

from lshash.lshash import LSHash
if __name__ == '__main__':

	lsh = LSHash(6, 8)
	lsh.index([1, 0.9, 3.1, 4, 5, 6, 7, 8],1)
	lsh.index([2, 3, 4, 5, 6, 7, 8, 10],2)
	lsh.index([10, 12, 99, 1, 5, 31, 2, 3],3)
	lsh.index([1, 2, 3, 4, 5, 6, 7, 8],4)
	a = lsh.query([1, 2, 3, 4, 5, 6, 7, 8],4)
	print(a)