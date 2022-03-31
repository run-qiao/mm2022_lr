# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : position_encoding.py
# Copyright (c) Skye-Song. All Rights Reserved

import numpy as np
import torch

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
	''' Sinusoid position encoding table '''

	# TODO: make it with torch instead of numpy
	def get_position_angle_vec(position):
		return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

	sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

	return torch.FloatTensor(sinusoid_table).unsqueeze(0)