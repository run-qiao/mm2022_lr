# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : update_model.py
# Copyright (c) Skye-Song. All Rights Reserved

import torch
import importlib

from .vit.vit import build_vit
from .updater.cluster import build_cluster
from .updater.vlad import build_vlad
from .updater.register import build_register

from lib.utils.load_pretrain import _get_prefix_dic
from lib.train.data.util.processing_utils import sample_target
import lib.train.data.util.transforms as tfm
from lib.utils.image import *


class UPDATOR():
	def __init__(self, vit, vlad, cluster, register, process, score_thres_add=0.9, score_thres_update=0.9):
		self.backbone = vit
		self.vlad = vlad
		self.cluster = cluster
		self.register = register
		self.gt_template = None
		self.gt_mask = None
		self.pre_template = None
		self.pre_mask = None
		self.pre_image = None
		self.pre_box = None
		self.image_bank = []
		self.box_bank = []
		self.encode_bank = []
		self.attmask_bank = []
		self.score_thres_update = score_thres_update
		self.score_thres_add = score_thres_add
		self.process = process
		self.count = 0

		self.start_len = 20
		self.start_interval = 2
		self.update_interval = 10

		self.backbone.cuda()
		self.register.cuda()

	def _deal_image(self, image, box):
		image, resize_factor, att_mask = sample_target(image, box, search_area_factor=2.0, output_sz=224)
		image, att_mask = self.process(image=[image], att=[att_mask], joint=False)
		return image[0].cuda(), att_mask[0].cuda()

	def _reduce_dim(self, x):

		x = x.squeeze(0).detach().cpu().numpy()

		# vlad
		# x = self.vlad.forward(x)

		# PCA
		# pca = PCA(n_components=10)
		# x = pca.fit_transform(x)

		# LLE
		# lle = LocallyLinearEmbedding(n_components=10,n_neighbors=100, method='hessian')
		# x = lle.fit_transform(x)

		# LSH
		# lsh = LSHash(20,10)
		# lsh.query()

		return x

	def _encoder(self, img, att_mask):

		triple_images = torch.stack([self.gt_template, self.pre_template, img], dim=0).unsqueeze(0)
		triple_masks = torch.stack([self.gt_mask, self.pre_mask, att_mask], dim=0).unsqueeze(0)
		score, encode = self.backbone(triple_images, triple_masks)
		encode = self._reduce_dim(encode)

		return score, encode

	def set_gt(self, image, box):
		self.count = 1

		self.backbone.eval()
		self.pre_image = image
		self.pre_box = box

		# crop the image and transform image to tensor
		gt_template, att_mask = self._deal_image(image, box)

		# record the image
		self.gt_mask = att_mask
		self.gt_template = gt_template
		self.pre_mask = att_mask
		self.pre_template = gt_template
		_, gt_encode = self._encoder(self.gt_template, self.gt_mask)

		# cluster init
		self.cluster.set_gt(gt_encode)

		# add to bank, put in last for _encoder_vlad
		self.image_bank.append(self.pre_image)
		self.box_bank.append(self.pre_box)
		self.encode_bank.append(gt_encode)
		self.attmask_bank.append(self.pre_mask)

	def update(self, image, box):
		self.count += 1

		is_update = False
		# if self.count >= 100 and self.count%80==0:
		# 	self.visualize()

		# less than start_len and in the start_interval
		if self.count < self.start_len and self.count % self.start_interval != 0:
			return None, None
		# more than start_len and in the update_interval
		elif self.count >= self.start_len and self.count % self.update_interval != 0:
			return None, None

		self.backbone.eval()
		template_img, att_mask = self._deal_image(image, box)
		score, template_encode = self._encoder(template_img, att_mask)

		if score >= self.score_thres_update and self.count >= self.start_len:
			is_update = True

		if score >= self.score_thres_add:
			image_id = len(self.image_bank)

			self.image_bank.append(image)
			self.box_bank.append(box)
			self.encode_bank.append(template_encode)
			self.attmask_bank.append(att_mask)

			update_id_list = self.cluster.update(image_id, template_encode, is_update)

			# 调整 box
			if is_update and update_id_list is not None:
				encode_list = [self.encode_bank[i] for i in update_id_list[:-1]] + [template_encode]
				att_mask_list = [self.attmask_bank[i] for i in update_id_list[:-1]] + [att_mask]
				refine_box = self.register(encode_list, att_mask_list, box)

				# 调整box之后重新计算
				refine_template, refine_att_mask = self._deal_image(image, refine_box)
				refine_score, refine_template_encode = self._encoder(refine_template, refine_att_mask)

				self.pre_template = refine_template
				self.pre_mask = refine_att_mask
				self.pre_image = image
				self.pre_box = refine_box

				self.box_bank[image_id] = refine_box
				self.encode_bank[image_id] = refine_template_encode
				self.attmask_bank[image_id] = refine_att_mask

				# 重新放入cluster的bank里面
				is_update = False
				self.cluster.update(image_id, refine_template_encode, is_update)

				return self.pre_image, self.pre_box

		# return None if template not change
		return None, None

	def visualize(self):
		ids, labels, prob = self.cluster.get_labels()

		ids = np.array(ids)
		labels = np.array(labels)
		prob = np.array(prob)
		# print(len(ids), len(prob), len(labels))
		if len(labels) == 0 and len(prob) == 0:
			return
		for i in range(12):
			index = np.where(labels == i)
			if len(index[0]) == 0:
				continue
			cluster_ids = ids[index]
			cluster_probs = prob[index]
			print('the cluster {} is {}:'.format(i, cluster_ids))
			cluster_imgs = np.array(self.image_bank)[cluster_ids[cluster_probs < 1.6]] / 255.
			cluster_boxes = np.array(self.box_bank)[cluster_ids[cluster_probs < 1.6]]
			draw_seq_image(imgs=cluster_imgs, crop_boxes=cluster_boxes)


def build_update_model(params):
	config_module = importlib.import_module("lib.config.vit.config")
	cfg = config_module.cfg
	process = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
	cfg = cfg.MODEL
	vit = build_vit(cfg)
	vlad = build_vlad(cfg)
	cluster = build_cluster(cfg)
	register = build_register(cfg)

	# load checkpoints
	checkpoint_path = params.updater_checkpoint_pth
	check_model = torch.load(checkpoint_path, map_location='cpu')['net']
	vit.load_state_dict(_get_prefix_dic(check_model,"backbone."), strict = True)
	register.load_state_dict(_get_prefix_dic(check_model, "register."), strict=True)
	print("loaded model: " + checkpoint_path)
	model = UPDATOR(
		vit=vit,
		vlad=vlad,
		cluster=cluster,
		process=process,
		register=register
	)
	return model

torch.save()
