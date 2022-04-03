# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : register.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from lib.utils.image import *
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xywh_to_xyxy


class Register():
	def __init__(self, patch_size=16, feature_size=224):
		self.patch_size = patch_size
		self.feature_size = feature_size
		self.d_size = [0.95, 0.98, 1.0, 1.02, 1.05]
		self.d_xy = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20]
		print()

	def _add_mask(self, features, masks):
		# return features
		# features=[t, p, n], p = h * w
		# masks = [t, feature_size, feautre_size]
		if len(features.shape) == 2:
			features = features.unsqueeze(0)
		if len(masks.shape) == 2:
			masks = masks.unsqueeze(0)

		[t, p, n] = features.shape
		if p != self.feature_size * self.feature_size:
			masks = F.interpolate(masks[None].float(), size=self.feature_size // self.patch_size,
			                      mode='bilinear').squeeze(0)

		return features * (1 - masks.flatten(1).unsqueeze(-1))

	def _cal_score(self, features, match_feature):
		# features=[t, p, n], match_feaure = [1, p, n];
		dims = features.shape[-1]
		attn_score = (match_feature @ features.transpose(-2, -1))
		attn_score = attn_score / dims
		return attn_score.mean(-1)

	# return attn_score.softmax(dim=-1) @ match_feature.squeeze(0)

	def _get_score(self, features, masks, match_feature):
		# features=[t, p, n], match_feaure = [p, n];
		# masks = [t, feature_size, feautre_size]
		# n = patch_size * patch_size; p = number of patch
		features = torch.Tensor(features).cuda()
		masks = torch.stack(masks, dim=0).cuda()

		mfeature = torch.Tensor(match_feature).unsqueeze(0).cuda()
		# draw_feat(rearrange(features[4], '(h w) c -> h w c', h=14))
		# patch score
		masked_features = self._add_mask(features, masks)
		patch_score = self._cal_score(masked_features, mfeature)
		patch_score = rearrange(patch_score, 't (h w) -> t h w', h=self.feature_size // self.patch_size)
		patch_score = F.interpolate(patch_score.unsqueeze(0), size=self.feature_size, mode="bicubic").squeeze(0)
		return patch_score

	# pixel score
	# 可以采用其他的插值方法
	# features = rearrange(features, 't (h w) p -> t p h w', h=self.feature_size // self.patch_size)
	# features = F.interpolate(features,size=self.feature_size, mode="bicubic").mean(1).flatten(1)
	# mfeature = rearrange(mfeature, 't (h w) p -> t p h w', h=self.feature_size // self.patch_size)
	# mfeature = F.interpolate(mfeature, size=self.feature_size, mode="bicubic").mean(1).flatten(1)
	# masked_features = self._add_mask(features.flatten(1).unsqueeze(-1), masks)
	# pixel_score = self._cal_score(masked_features, mfeature.flatten(1).unsqueeze(-1))
	# pixel_score = rearrange(pixel_score, 't (h w) -> t h w', h=self.feature_size)
	#
	# return patch_score + 0.5 * pixel_score

	def _generate_mask(self, mask, d_xy, d_size):

		gen_mask = torch.ones_like(mask)

		y_index = torch.where(mask.float().mean(1) < 1)[0]
		y1 = y_index[0].float().item()
		y2 = y_index[-1].float().item()

		x_index = torch.where(mask.float().mean(0) < 1)[0]
		x1 = x_index[0].float().item()
		x2 = x_index[-1].float().item()

		x_c, y_c = [i + j for i, j in zip(d_xy, [(x1 + x2) / 2, (y1 + y2) / 2])]
		w, h = [i * j for i, j in zip(d_size, [x2 - x1, y2 - y1])]

		x1 = max(round(x_c - w / 2), 0)
		x2 = min(round(x1 + w), self.feature_size - 1)

		y1 = max(round(y_c - h / 2), 0)
		y2 = min(round(y1 + h), self.feature_size - 1)

		gen_mask[y1:y2 + 1, x1:x2 + 1] = 0

		return gen_mask

	def _refine_mask(self, score_map, mask):
		# score = [t, h, w]
		# mask = [h, w]
		max_score = float("inf")
		max_d_xy = None
		max_d_size = None
		for d_x in self.d_xy:
			for d_y in self.d_xy:
				for d_w in self.d_size:
					for d_h in self.d_size:
						d_xy = [d_x, d_y]
						d_size = [d_w, d_h]
						gen_mask = self._generate_mask(mask, d_xy, d_size)
						gen_mask = 1 - gen_mask.float()
						score = score_map * gen_mask
						score = score.sum() / gen_mask.sum()
						if score < max_score:
							max_score = score
							max_d_xy = d_xy
							max_d_size = d_size

		return max_d_xy, max_d_size

	def _refine_box(self, box, d_xy, d_size):
		x, y, w, h = box
		resize_factor = math.ceil(math.sqrt(w * h) * 2.0) / self.feature_size

		new_x = round(x + resize_factor * d_xy[0])
		new_y = round(y + resize_factor * d_xy[1])
		new_w = round(w * d_size[0])
		new_h = round(h * d_size[1])

		return [new_x, new_y, new_w, new_h]

	def refine(self, features, masks, box):
		pre_features = features[:-1]
		pre_masks = masks[:-1]
		feature = features[-1]
		mask = masks[-1]

		score_map = self._get_score(pre_features, pre_masks, feature)  # [t, feat_size, feat_size]

		d_xy, d_size = self._refine_mask(score_map, mask)

		if d_xy is None or d_size is None:
			return None

		return_box = self._refine_box(box, d_xy, d_size)

		return return_box


class Register_ST(nn.Module):
	def __init__(self, patch_size=16, feature_size=224):
		super().__init__()
		self.patch_size = patch_size
		self.feature_size = feature_size
		self.box_head = build_box_head()
		self._reset_parameters()

	def _reset_parameters(self):
		for p in self.box_head.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def _add_mask(self, features, masks):

		# return features
		# features=[b, t, p, n], p = h * w
		# masks = [b, t, feature_size, feautre_size]
		if len(features.shape) == 2:
			features = features.unsqueeze(0)
		if len(masks.shape) == 2:
			masks = masks.unsqueeze(0)

		[b, t, p, n] = features.shape
		if p != self.feature_size * self.feature_size:
			masks = F.interpolate(masks.float(), size=self.feature_size // self.patch_size,
			                      mode='bilinear')

		return features * (1 - masks.flatten(2).unsqueeze(-1))

	def _score_map(self, features, masks, match_feature):
		# features=[b, t, p, n], match_feaure = [b, p, n];
		# masks = [b, t, feature_size, feautre_size]
		# n = patch_size * patch_size; p = number of patch

		# score
		b, t, p, n = features.shape
		masked_features = self._add_mask(features, masks)  # b, t, p, n
		mfeature = match_feature.transpose(-2, -1)  # b * n * p

		masked_features = rearrange(masked_features, 'b t p n -> b (t p) n')  # b * tp * n
		score = masked_features @ mfeature  # b * tp * p
		score = score / math.sqrt(n)
		score = torch.softmax(score, dim=1)

		out_feat1 = masked_features.transpose(-2, -1) @ score  # b * n * p
		out_feat = torch.cat([out_feat1, mfeature], dim=1)
		out_feat = rearrange(out_feat, 'b n (h w) -> b n h w', h=self.feature_size // self.patch_size)
		return out_feat

	def map_box_back(self, pred_box: list, resize_factor: float):
		cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
		cx, cy, w, h = pred_box
		half_side = 0.5 * self.params.search_size / resize_factor
		cx_real = cx + (cx_prev - half_side)
		cy_real = cy + (cy_prev - half_side)
		return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

	# for test
	def _refine_box(self, prev_box, boxs):
		x, y, w, h = prev_box
		cx_prev = x + 0.5 * w
		cy_prev = y + 0.5 * h

		boxs = boxs.view(-1, 4)  # [x1,y1,x2,y2]

		# get the absolute coordinate
		crop_size = math.ceil(math.sqrt(w * h) * 2.0)
		boxs = (boxs * crop_size).tolist()  # [x1,y1,x2,y2]

		# map to real coordinate
		r_x1, r_y1, r_x2, r_y2 = boxs[0]
		r_x = r_x1 + (cx_prev - 0.5 * crop_size)
		r_y = r_y1 + (cy_prev - 0.5 * crop_size)

		return [r_x, r_y, r_x2 - r_x1, r_y2 - r_y1]

	# outputs_coord = box_xyxy_to_xywh(box)

	def forward(self, features, masks, box=None, map_box=True):
		pre_features = features[:-1]
		pre_masks = masks[:-1]
		feature = features[-1]
		mask = masks[-1]

		if map_box:
			pre_features = torch.Tensor(pre_features).unsqueeze(0).cuda()
			pre_masks = torch.stack(pre_masks, dim=0).unsqueeze(0).cuda()
			feature = torch.Tensor(feature).unsqueeze(0).cuda()
		else:
			pre_features = torch.stack(pre_features, dim=1).cuda()
			pre_masks = torch.stack(pre_masks, dim=1).cuda()
			feature = feature

		# draw_feat(rearrange(features[4], '(h w) c -> h w c', h=14))

		feat = self._score_map(pre_features, pre_masks, feature)  # b * n * h * w

		return_box = self.box_head(feat)  # b * n * 4

		if map_box:
			return_box = self._refine_box(box, return_box)

		return return_box


def build_register(cfg=None):
	return Register_ST()
