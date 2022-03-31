# -*- coding:utf-8 -*-
# author  : Skye Song
# file    : mae_decoder.py
# Copyright (c) Skye-Song. All Rights Reserved
import torch
import torch.nn as nn
from .module import PatchEmbed, Block, trunc_normal_, get_norm_layer
from .position_encoding import get_sinusoid_encoding_table
from einops import rearrange

class VIT(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
	             num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
	             drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
	             use_learnable_pos_emb=False):
		super().__init__()
		self.num_classes = num_classes
		self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

		self.patch_embed = PatchEmbed(
			img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
		num_patches = self.patch_embed.num_patches

		if use_learnable_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * 3, embed_dim))
		else:
			# sine-cosine positional embeddings
			self.pos_embed = get_sinusoid_encoding_table(num_patches * 3, embed_dim)
		self.pos_drop = nn.Dropout(p=drop_rate)

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
		self.blocks = nn.ModuleList([
			Block(
				dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
				drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
				init_values=init_values)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim // 3)
		self.head_for_code = nn.Linear(embed_dim, embed_dim//3)
		self.head = nn.Linear(embed_dim // 3, num_classes) if num_classes > 0 else nn.Identity()

		if use_learnable_pos_emb:
			trunc_normal_(self.pos_embed, std=.02)

		# trunc_normal_(self.cls_token, std=.02)
		self.apply(self._init_weights)

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
		return {'pos_embed', 'blocks', 'patch_embed'}

	def forward_features(self, x, mask=None):
		batch_size = x.shape[0]

		x = rearrange(x, 'b t c h w -> (b t) c h w')
		if mask is not None:
			mask = rearrange(mask, 'b t h w -> (b t) h w')
			x, mask = self.patch_embed(x, mask)
			mask = rearrange(mask, '(b t) p-> b (t p)', b=batch_size)
		else:
			x= self.patch_embed(x)
		x = rearrange(x, '(b t) p n -> b (t p) n', b=batch_size)

		x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
		x = self.pos_drop(x)

		for blk in self.blocks:
			x = blk(x, mask)
		return x

	def forward(self, x, mask = None):
		x_feat = self.forward_features(x, mask)
		x_feat = self.head_for_code(x_feat)
		x = self.norm(x_feat.mean(1))
		x = self.head(x)
		return x, x_feat[:,-x_feat.shape[1]//3:,:]

def build_vit(cfg):
	return VIT(
		img_size=cfg.IMAGE_SIZE,
		patch_size=cfg.PATCH_SIZE,
		in_chans=cfg.VIT.IN_CHANNELS,
		num_classes=cfg.VIT.NUM_CLASSES,
		embed_dim=cfg.VIT.EMBED_DIM,
		depth=cfg.VIT.DEPTH,
		num_heads=cfg.VIT.NUM_HEADS,
		mlp_ratio=cfg.VIT.MLP_RATIO,
		qkv_bias=cfg.VIT.QKV_BIAS,
		# qk_scale=cfg.ENCODER.QK_SCALE,
		qk_scale=None,
		drop_rate=cfg.VIT.DROP_RATE,
		attn_drop_rate=cfg.VIT.ATTN_DROP_RATE,
		drop_path_rate=cfg.VIT.DROP_PATH_RATE,
		norm_layer=get_norm_layer(cfg.VIT.NORM_LAYER),
		init_values=cfg.VIT.INIT_VALUES,
		use_learnable_pos_emb=cfg.VIT.USE_LEARNABLE_POS_EMB)
