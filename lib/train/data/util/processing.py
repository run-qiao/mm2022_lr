import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.util.processing_utils as prutils
import torch.nn.functional as F
from lib.utils.image import *
from lib.utils.box_ops import box_xywh_to_xyxy, box_iou


def stack_tensors(x):
	if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
		if len(x) > 1:
			return torch.stack(x)
		else:
			return x[0]
	return x


class BaseProcessing:
	""" Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
	 through the network. For example, it can be used to crop a search region around the object, apply various data
	 augmentations, etc."""

	def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None,
	             joint_transform=None):
		"""
		args:
			transform       - The set of transformations to be applied on the images. Used only if template_transform or
								search_transform is None.
			template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
								argument is used instead.
			search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
								argument is used instead.
			joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
								example, it can be used to convert both template and search images to grayscale.
		"""
		self.transform = {'template': transform if template_transform is None else template_transform,
		                  'search': transform if search_transform is None else search_transform,
		                  'joint': joint_transform}

	def __call__(self, data: TensorDict):
		raise NotImplementedError


class MaskProcessing():

	def __init__(self, output_sz, center_jitter_factor, scale_jitter_factor,
	             mode='image', settings=None, transform=transforms.ToTensor()):
		self.output_sz = output_sz
		self.center_jitter_factor = center_jitter_factor
		self.scale_jitter_factor = scale_jitter_factor
		self.mode = mode
		self.settings = settings
		self.transform = transform

	def _get_jittered_box(self, box):
		jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor)
		max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor).float())
		jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

		return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

	def __call__(self, data: TensorDict):
		# Add a uniform noise to the center pos
		jittered_anno = [self._get_jittered_box(a) for a in data['bboxes']]

		# 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
		w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

		crop_sz = torch.ceil(torch.sqrt(w * h) * 2.0)
		if (crop_sz < 1).any():
			data['valid'] = False
			# print("Too small box is found. Replace it with new data.")
			return data

		# Crop image region centered at jittered_anno box and get the attention mask
		crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data['images'], jittered_anno,
		                                                                  data['bboxes'],
		                                                                  2.0,
		                                                                  self.output_sz, masks=data['masks'])
		# Apply transforms
		data['images'], data['bboxes'], data['att'], data['masks'] = self.transform(image=crops, bbox=boxes,
		                                                                            att=att_mask, mask=mask_crops,
		                                                                            joint=False)

		data['valid'] = True
		data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
		return data


class SequenceProcessing():
	def __init__(self, output_sz, center_jitter_factor, scale_jitter_factor,
	             mode='image', settings=None, transform=transforms.ToTensor()):
		self.output_sz = output_sz
		self.center_jitter_factor = center_jitter_factor
		self.scale_jitter_factor = scale_jitter_factor
		self.mode = mode
		self.settings = settings
		self.transform = transform

	def _get_jittered_box(self, box):

		jittered_size = box[2:4] * torch.exp(torch.randn(2).clamp(-1., 1.) * self.scale_jitter_factor)
		max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor).float())
		jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

		return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

	def __call__(self, data: TensorDict):

		for s in ['seq_', 'gt_', 'label_']:

			# Add a uniform noise to the center pos
			if s == 'seq_':
				jittered_anno = [self._get_jittered_box(a) for a in data[s + 'bboxes']]
			else:
				jittered_anno = data[s + 'bboxes']

			# Check whether data is valid. Avoid too small bounding boxes
			w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

			crop_sz = torch.ceil(torch.sqrt(w * h) * 2.0)
			if (crop_sz < 1).any():
				data['valid'] = False
				# print("Too small box is found. Replace it with new data.")
				return data

			# Crop image region centered at jittered_anno box and get the attention mask
			crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + 'images'], jittered_anno,
			                                                                  data[s + 'bboxes'],
			                                                                  2.0,
			                                                                  self.output_sz, masks=data[s + 'masks'])
			# Apply transforms
			data[s + 'images'], data[s + 'bboxes'], data[s + 'att'], data[s + 'masks'] = self.transform(image=crops,
			                                                                                            bbox=boxes,
			                                                                                            att=att_mask,
			                                                                                            mask=mask_crops,
			                                                                                            joint=False)

			# Check whether elements in data[s + '_att'] is all 1
			# Note that type of data['att'] is tuple, type of ele is torch.tensor
			for ele in data[s + 'att']:
				if (ele == 1).all():
					data['valid'] = False
					# print("Values of original attention mask are all one. Replace it with new data.")
					return data
			# # more strict conditions: require the donwsampled masks not to be all 1
			for ele in data[s + 'att']:
				feat_size = self.output_sz // 16  # 16 is the backbone stride
				mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
				if (mask_down == 1).all():
					data['valid'] = False
					# print("Values of down-sampled attention mask are all one. "
					#       "Replace it with new data.")
					return data

		data['valid'] = True
		data = data.apply(stack_tensors)
		return data


class PairProcessing():
	def __init__(self, output_sz, pos_center_jitter_factor, pos_scale_jitter_factor, neg_center_jitter_factor,
	             neg_scale_jitter_factor, settings=None, transform=transforms.ToTensor()):
		self.output_sz = output_sz
		self.pos_center_jitter_factor = pos_center_jitter_factor
		self.pos_scale_jitter_factor = pos_scale_jitter_factor
		self.neg_center_jitter_factor = neg_center_jitter_factor
		self.neg_scale_jitter_factor = neg_scale_jitter_factor
		self.settings = settings
		self.transform = transform

	def _get_jittered_box(self, box, is_positive):

		# box [x_left, y_left, w, h]
		if is_positive:
			center_jitter_factor = self.pos_center_jitter_factor
			scale_jitter_factor = self.pos_scale_jitter_factor
		else:
			center_jitter_factor = self.neg_center_jitter_factor
			scale_jitter_factor = self.neg_scale_jitter_factor

		valid = False
		while not valid:
			jittered_size = box[2:4] * torch.exp(torch.randn(2).clamp(-1., 1.) * scale_jitter_factor)
			max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor).float())
			jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

			jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

			iou, _ = box_iou(box_xywh_to_xyxy(jittered_box)[np.newaxis, :], box_xywh_to_xyxy(box)[np.newaxis, :])

			if is_positive and iou[0] > 0.5:
				valid = True
			elif not is_positive and iou[0] < 0.3:
				valid = True

		return jittered_box

	def __call__(self, data: TensorDict, pos: bool, neg: bool):

		prev_s = ['gt_', 'gt_new_']
		if pos:
			prev_s.append('pos_sample_')
		if neg:
			prev_s.append('neg_sample_')

		for s in prev_s:
			if s.endswith('sample_'):
				data[s + 'valid'] = True

			if s == 'gt_':
				jittered_anno = data[s + 'bboxes']
			elif s == 'gt_new_' or s == 'pos_sample_':
				jittered_anno = [self._get_jittered_box(a, True) for a in data[s + 'bboxes']]
			elif s == 'neg_sample_':
				jittered_anno = [self._get_jittered_box(a, False) for a in data[s + 'bboxes']]

			# Check whether data is valid. Avoid too small bounding boxes
			w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
			crop_sz = torch.ceil(torch.sqrt(w * h) * 2.0)
			if (crop_sz < 1).any():
				data['valid'] = False
				print("Too small box is found. Replace it with new data.")
				return data

			# Crop image region centered at jittered_anno box and get the attention mask
			crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + 'images'], jittered_anno,
			                                                                  data[s + 'bboxes'],
			                                                                  2.0,
			                                                                  self.output_sz, masks=data[s + 'masks'])
			# Apply transforms
			data[s + 'images'], data[s + 'bboxes'], data[s + 'att'], data[s + 'masks'] = self.transform(image=crops,
			                                                                                            bbox=boxes,
			                                                                                            att=att_mask,
			                                                                                            mask=mask_crops,
			                                                                                            joint=False)

			# Check whether elements in data[s + '_att'] is all 1
			# Note that type of data['att'] is tuple, type of ele is torch.tensor

			# if s == 'neg_sample_':
			# 	continue

			for ele in data[s + 'att']:
				if (ele == 1).all():
					data['valid'] = False
					# print("Values of original attention mask are all one. Replace it with new data.")
					return data
			# # more strict conditions: require the donwsampled masks not to be all 1
			for ele in data[s + 'att']:
				feat_size = self.output_sz // 16  # 16 is the backbone stride
				mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
				if (mask_down == 1).all():
					data['valid'] = False
					# print("Values of down-sampled attention mask are all one. "
					#       "Replace it with new data.")
					return data

		data['valid'] = True
		data = data.apply(stack_tensors)
		return data


class TrackingProcessing(BaseProcessing):
	""" The processing class used for training LittleBoy. The images are processed in the following way.
	First, the target bounding box is jittered by adding some noise.
	Next, a square region (called search region ) centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
	cropped from the image.
	The reason for jittering the target box is to avoid learning the bias that the target is always at the center of the search region.
	The search region is then resized to a fixed size given by the argument output_sz.
	"""

	def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
	             mode='pair', settings=None, *args, **kwargs):
		"""
		args:
			search_area_factor - The size of the search region  relative to the target size.
			output_sz - An integer, denoting the size to which the search region is resized. The search region is always
						square.
			center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
									extracting the search region. See _get_jittered_box for how the jittering is done.
			scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
									extracting the search region. See _get_jittered_box for how the jittering is done.
			mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
		"""
		super().__init__(*args, **kwargs)
		self.search_area_factor = search_area_factor
		self.output_sz = output_sz
		self.center_jitter_factor = center_jitter_factor
		self.scale_jitter_factor = scale_jitter_factor
		self.mode = mode
		self.settings = settings

	def _get_jittered_box(self, box, mode):
		""" Jitter the input box
		args:
			box - input bounding box
			mode - string 'template' or 'search' indicating template or search data

		returns:
			torch.Tensor - jittered box
		"""

		jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
		max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
		jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

		# fix_rand = torch.tensor([0.2665, -0.6954], dtype=torch.float32)
		# jittered_size = box[2:4] * torch.exp(fix_rand * self.scale_jitter_factor[mode])
		# max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
		# jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (fix_rand - 0.5)

		return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

	def __call__(self, data: TensorDict):
		"""
		args:
			data - The input data, should contain the following fields:
				'template_images', search_images', 'template_anno', 'search_anno'
		returns:
			TensorDict - output data block with following fields:
				'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
		"""
		# Apply joint transforms
		# draw_image(data['search_images'][0], data['search_anno'][0].cpu().numpy())
		if self.transform['joint'] is not None:
			data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
				image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
			data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
				image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

		for s in ['template', 'search']:
			assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
				"In pair mode, num train/test frames must be 1"

			# Add a uniform noise to the center pos
			jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

			# 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
			w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

			crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
			if (crop_sz < 1).any():
				data['valid'] = False
				# print("Too small box is found. Replace it with new data.")
				return data

			# Crop image region centered at jittered_anno box and get the attention mask
			crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
			                                                                  data[s + '_anno'],
			                                                                  self.search_area_factor[s],
			                                                                  self.output_sz[s],
			                                                                  masks=data[s + '_masks'])
			# Apply transforms
			data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
				image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

			# 2021.1.9 Check whether elements in data[s + '_att'] is all 1
			# Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
			for ele in data[s + '_att']:
				if (ele == 1).all():
					data['valid'] = False
					print("Values of original attention mask are all one. Replace it with new data.")
					return data
			# 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
			for ele in data[s + '_att']:
				feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
				# (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
				mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
				if (mask_down == 1).all():
					data['valid'] = False
					print("Values of down-sampled attention mask are all one. "
					      "Replace it with new data.")
					return data

		data['valid'] = True
		# if we use copy-and-paste augmentation
		if data["template_masks"] is None or data["search_masks"] is None:
			data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
			data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
		# Prepare output
		if self.mode == 'sequence':
			data = data.apply(stack_tensors)
		else:
			data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

		return data
