import torch.nn as nn
import torch


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
		          padding=padding, dilation=dilation, bias=True),
		nn.BatchNorm2d(out_planes),
		nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
	""" Corner Predictor module"""
	def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16):
		super(Corner_Predictor, self).__init__()
		self.feat_sz = feat_sz
		self.stride = stride
		self.img_sz = self.feat_sz * self.stride
		'''top-left corner'''
		self.conv1_tl = conv(inplanes, channel)
		self.conv2_tl = conv(channel, channel // 2)
		self.conv3_tl = conv(channel // 2, channel // 4)
		self.conv4_tl = conv(channel // 4, channel // 8)
		self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

		'''bottom-right corner'''
		self.conv1_br = conv(inplanes, channel)
		self.conv2_br = conv(channel, channel // 2)
		self.conv3_br = conv(channel // 2, channel // 4)
		self.conv4_br = conv(channel // 4, channel // 8)
		self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

		'''about coordinates and indexs'''
		with torch.no_grad():
			self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
			# generate mesh-grid
			self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
				.view((self.feat_sz * self.feat_sz,)).float().cuda()
			self.coord_y = self.indice.repeat((1, self.feat_sz)) \
				.view((self.feat_sz * self.feat_sz,)).float().cuda()

	def forward(self, x):
		""" Forward pass with input x. """
		score_map_tl, score_map_br = self.get_score_map(x)
		coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
		coorx_br, coory_br = self.soft_argmax(score_map_br)
		return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

	def get_score_map(self, x):
		# top-left branch
		x_tl1 = self.conv1_tl(x)
		x_tl2 = self.conv2_tl(x_tl1)
		x_tl3 = self.conv3_tl(x_tl2)
		x_tl4 = self.conv4_tl(x_tl3)
		score_map_tl = self.conv5_tl(x_tl4)

		# bottom-right branch
		x_br1 = self.conv1_br(x)
		x_br2 = self.conv2_br(x_br1)
		x_br3 = self.conv3_br(x_br2)
		x_br4 = self.conv4_br(x_br3)
		score_map_br = self.conv5_br(x_br4)
		return score_map_tl, score_map_br

	def soft_argmax(self, score_map):
		""" get soft-argmax coordinate for a given heatmap """
		prob_vec = nn.functional.softmax(
			score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
		exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
		exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
		return exp_x, exp_y

def build_box_head(cfg=None):
	corner_head = Corner_Predictor(inplanes=512, channel=256,
	                               feat_sz=14, stride=16)
	return corner_head
