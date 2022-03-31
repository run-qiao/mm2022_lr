# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : update_model.py
# Copyright (c) Skye-Song. All Rights Reserved

import torch
import importlib
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from lshash.lshash import LSHash

from .vit.vit import build_vit
from .updater.cluster import build_cluster
from .updater.vlad import build_vlad

from lib.train.data.util.processing_utils import sample_target
import lib.train.data.util.transforms as tfm
from lib.utils.image import *


class UPDATOR():
	def __init__(self, vit, vlad, cluster, process, update_thres=0.9):
		self.vit = vit
		self.vlad = vlad
		self.cluster = cluster
		self.gt_template = None
		self.gt_mask = None
		self.pre_template = None
		self.pre_mask = None
		self.pre_image = None
		self.pre_box = None
		self.image_bank = []
		self.box_bank = []
		self.update_thres = update_thres
		self.process = process
		self.count = 0

		self.start_len = 20
		self.start_interval = 2
		self.update_interval = 20

		self.vit.cuda()

	def _deal_image(self, image, box):
		image, resize_factor, att_mask = sample_target(image, box, search_area_factor=2.0, output_sz=224)
		image, att_mask = self.process(image=[image], att=[att_mask], joint=False)
		return image[0].cuda(), att_mask[0].cuda()

	def set_gt(self, image, box):
		self.count = 1

		self.vit.eval()
		self.pre_image = image
		self.pre_box = box

		# crop the image and transform image to tensor
		gt_template, att_mask = self._deal_image(image, box)

		# record the image
		self.gt_template = gt_template
		self.gt_mask = att_mask
		self.pre_template = gt_template
		self.pre_mask = att_mask
		_, gt_encode = self._encoder(self.gt_template, self.gt_mask)

		# cluster init
		self.cluster.set_gt(gt_encode)

		# add to bank, put in last for _encoder_vlad
		self.image_bank.append(self.pre_image)
		self.box_bank.append(self.pre_box)

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
		score, encode = self.vit(triple_images, triple_masks)
		encode = self._reduce_dim(encode)

		return score, encode

	def update(self, image, box):
		self.count += 1
		is_update=True

		# if self.count >= 100 and self.count%30==0:
		# 	self.visualize()



		# less than start_len and in the start_interval
		if self.count < self.start_len:
			return None, None
		if self.count % self.start_interval != 0:
			return None, None
		# more than start_len and in the update_interval
		elif self.count % self.update_interval != 0:
			is_update=False


		self.vit.eval()
		template_img, att_mask = self._deal_image(image, box)
		score, template_encode = self._encoder(template_img, att_mask)

		if score >= self.update_thres:
			image_id = len(self.image_bank)

			self.image_bank.append(image)
			self.box_bank.append(box)

			update_id = self.cluster.update(image_id, template_encode,is_update)

			if update_id > 0:
				self.pre_template = template_img
				self.pre_mask = att_mask
				self.pre_image = self.image_bank[update_id]
				self.pre_box = self.box_bank[update_id]
				return self.pre_image, self.pre_box

		# return None if template not change
		return None, None

	def visualize(self):
		ids,labels,prob = self.cluster.get_labels()

		ids = np.array(ids)
		labels = np.array(labels)
		prob=np.array(prob)
		# print(len(ids), len(prob), len(labels))
		if len(labels)==0 and len(prob)==0:
			return
		for i in range(12):
			index = np.where(labels == i)
			if len(index[0]) == 0:
				continue
			cluster_ids = ids[index]
			cluster_probs=prob[index]
			print('the cluster {} is {}:'.format(i,cluster_ids))
			# cluster_imgs = np.array(self.image_bank)[cluster_ids[cluster_probs<1.6]]/255.
			# cluster_boxes = np.array(self.box_bank)[cluster_ids[cluster_probs<1.6]]
			# draw_seq_image(imgs=cluster_imgs, crop_boxes=cluster_boxes)




def build_update_model(params):
	config_module = importlib.import_module("lib.config.vit.config")
	cfg = config_module.cfg
	process = tfm.Transform(tfm.ToTensor(), tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
	cfg = cfg.MODEL
	vit = build_vit(cfg)

	checkpoint_path = params.vit_checkpoint_pth
	vit.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'], strict=True)
	print("load vit model: " + checkpoint_path)

	vlad = build_vlad(cfg)
	cluster = build_cluster(cfg)
	model = UPDATOR(
		vit=vit,
		vlad=vlad,
		cluster=cluster,
		process=process
	)
	return model





