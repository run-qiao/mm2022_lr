# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : mae.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn

from .mae_3d.mae_3d_encoder import build_encoder
from .mae_3d.mae_3d_decoder import build_decoder
from .mae.position_encoding import get_sinusoid_encoding_table
from .mae.module import trunc_normal_

from lib.utils.image import *

class MAE3D(nn.Module):
	def __init__(self,
	             encoder,
	             decoder,
	             encoder_embed_dim=768,
	             decoder_embed_dim=512,
	             ):
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
		self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
		self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)
		trunc_normal_(self.mask_token, std=.02)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def get_num_layers(self):
		return len(self.blocks)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed', 'cls_token', 'mask_token'}

	def forward(self, x, mask):
		# [B, T, C, H, W]

		x_vis = self.encoder(x, mask)
		x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

		B, N, C = x_vis.shape
		# we don't unshuffle the correct visible token order,
		# but shuffle the pos embedding accorddingly.
		pos_emd_vis = self.pos_embed.repeat(B, self.encoder.seq_len, 1).type_as(x).to(x.device).clone().detach()
		pos_emd_mask = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
		x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

		x = self.decoder(x_full, pos_emd_mask.shape[1])  # [B, token, 3 * 16 * 16]

		return x


def _cfg(url='', **kwargs):
	return {
		'url': url,
		'num_classes': 1000,
		'input_size': (3, 224, 224),
		'pool_size': None,
		'crop_pct': .9,
		'interpolation': 'bicubic',
		'mean': (0.5, 0.5, 0.5),
		'std': (0.5, 0.5, 0.5),
		**kwargs
	}

def build_mae_3d(cfg):
	encoder = build_encoder(cfg)
	decoder = build_decoder(cfg)
	model = MAE3D(
		encoder=encoder,
		decoder=decoder,
		encoder_embed_dim=cfg.ENCODER.EMBED_DIM,
		decoder_embed_dim=cfg.DECODER.EMBED_DIM)
	model.default_cfg = _cfg()
	return model