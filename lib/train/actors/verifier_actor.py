from . import BaseActor
from einops import rearrange
from lib.utils.image import *

class VERIFIERACTOR(BaseActor):
	def __init__(self, net, objective, loss_weight, settings):
		super().__init__(net, objective)
		self.loss_weight = loss_weight
		self.settings = settings
		self.patchsize = self.settings.batch_size

	def __call__(self, data):
		pos_pair_images = torch.stack([data['gt_images'], data['gt_new_images'], data['pos_sample_images']], dim=1)
		pos_pair_masks = torch.stack([data['gt_masks'], data['gt_new_masks'], data['pos_sample_masks']], dim=1)

		neg_pair_images = torch.stack([data['gt_images'], data['gt_new_images'], data['neg_sample_images']], dim=1)
		neg_pair_masks = torch.stack([data['gt_masks'], data['gt_new_masks'], data['neg_sample_masks']], dim=1)

		# forward pass
		pos_result, pos_feat = self.net(pos_pair_images, pos_pair_masks)

		neg_result, neg_feat = self.net(neg_pair_images, neg_pair_masks)

		# compute losses
		loss, status = self.compute_losses(pos_result, neg_result)

		return loss, status

	def compute_losses(self, pos_result, neg_result, return_status=True):
		# compute loss
		try:
			pred = torch.cat([pos_result, neg_result], dim=1)
			label = torch.cat([torch.ones(pos_result.shape), torch.zeros(neg_result.shape)], dim=1).cuda()
			mse_loss = self.objective['mse'](input=pred, target=label)
		except:
			mse_loss = torch.tensor(0.0).cuda()

		# weighted sum
		loss = self.loss_weight['mse'] * mse_loss

		if return_status:
			# status for log
			status = {"Loss/total": loss.item(),
			          "Loss/mse": mse_loss.item()}
			return loss, status
		else:
			return loss
