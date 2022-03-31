import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
from lib.utils.image import *


def no_processing(data):
	return data


class MaskDataset(torch.utils.data.Dataset):
	def __init__(self, datasets, p_datasets, samples_per_epoch, processing=no_processing):
		self.datasets = datasets

		if p_datasets is None:
			p_datasets = [len(d) for d in self.datasets]
		# Normalize
		p_total = sum(p_datasets)
		self.p_datasets = [x / p_total for x in p_datasets]

		self.samples_per_epoch = samples_per_epoch
		self.processing = processing

	def __len__(self):
		return self.samples_per_epoch

	def __getitem__(self, index):
		"""
		returns:
			TensorDict - dict containing all the data blocks
		"""
		valid = False
		while not valid:
			# Select a dataset
			dataset = random.choices(self.datasets, self.p_datasets)[0]
			is_video_dataset = dataset.is_video_sequence()

			# sample a sequence
			seq_id = random.randint(0, dataset.get_num_sequences() - 1)
			seq_info_dict = dataset.get_sequence_info(seq_id)
			visible = seq_info_dict['visible']

			# sample a visible frame
			valid_ids = [i for i in range(0, len(visible)) if visible[i]]

			if len(valid_ids) == 0:
				continue

			frame_ids = random.choices(valid_ids)

			try:
				frames, anno, meta_obj = dataset.get_frames(seq_id, frame_ids, seq_info_dict)
				H, W, _ = frames[0].shape
				masks = anno['mask'] if 'mask' in anno else [torch.zeros((H, W))]
				data = TensorDict({'images': frames,
				                   'bboxes': anno['bbox'],
				                   'masks': masks,
				                   'dataset': dataset.get_name(),
				                   'test_class': meta_obj.get('object_class_name')})
				# make data augmentation
				data = self.processing(data)
				# check whether data is valid
				valid = data['valid']
			except:
				valid = False

		return data


class PairDataset(torch.utils.data.Dataset):
	def __init__(self, datasets, p_datasets, samples_per_epoch, same_anchor=False, neg_to_pos=5,
	             processing=no_processing):
		self.datasets = datasets

		if p_datasets is None:
			p_datasets = [len(d) for d in self.datasets]
		# Normalize
		p_total = sum(p_datasets)
		self.p_datasets = [x / p_total for x in p_datasets]

		self.samples_per_epoch = samples_per_epoch
		self.same_anchor = same_anchor
		self.neg_to_pos = neg_to_pos
		self.processing = processing

	def __len__(self):
		return self.samples_per_epoch

	def _sample_frame_ids(self, visible, avoid_id=None, min_id=None, max_id=None):
		""" Samples num_ids frames between min_id and max_id for which target is visible

		returns:
			list - List of sampled frame numbers. None if not sufficient visible frames could be found.
		"""
		if min_id is None or min_id < 0:
			min_id = 0
		if max_id is None or max_id > len(visible):
			max_id = len(visible)
		# get valid ids
		valid_ids = [i for i in range(min_id, max_id) if visible[i]]

		if avoid_id is not None:
			avoid_len = min(100, len(valid_ids) // 3)
			avoid_ids = [i for i in range(max(0, avoid_id - avoid_len), min(len(valid_ids), avoid_id + avoid_len))]
			valid_ids = list(filter(lambda x: x not in avoid_ids, valid_ids))

		# No visible ids
		if len(valid_ids) < 3:
			return None

		ids = random.sample(valid_ids, k=3)
		return sorted(ids)

	def _sample_seq_id(self, dataset, is_video_dataset):
		enough_visible_frames = False
		while not enough_visible_frames:
			seq_id = random.randint(0, dataset.get_num_sequences() - 1)
			seq_info_dict = dataset.get_sequence_info(seq_id)
			visible = seq_info_dict['visible']

			enough_visible_frames = visible.type(torch.int64).sum().item() > 5 and len(visible) >= 20
			enough_visible_frames = enough_visible_frames or not is_video_dataset

		return seq_id, visible, seq_info_dict

	def __getitem__(self, index):
		"""
		returns:
			TensorDict - dict containing all the data blocks
		"""
		valid = False
		while not valid:
			# Select a dataset
			dataset = random.choices(self.datasets, self.p_datasets)[0]
			is_video_dataset = dataset.is_video_sequence()

			# sample a sequence
			seq_id, visible, seq_info_dict = self._sample_seq_id(dataset, is_video_dataset)

			if is_video_dataset:
				# sample 3 visible frame, [template_gt, template_new, track_result]
				frame_ids = self._sample_frame_ids(visible)
			else:
				frame_ids = [1] * 3

			if frame_ids is None:
				continue

			try:
				frames, anno, meta_obj = dataset.get_frames(seq_id, frame_ids, seq_info_dict)
				H, W, _ = frames[0].shape
				masks = anno['mask'] if 'mask' in anno else [torch.zeros((H, W))] * 3

				# sample 4 frame, [gt, gt_new, sample]
				data = TensorDict({'gt_images': [frames[0]],
				                   'gt_bboxes': [anno['bbox'][0]],
				                   'gt_masks': [masks[0]],
				                   'gt_new_images': [frames[1]],
				                   'gt_new_bboxes': [anno['bbox'][1]],
				                   'gt_new_masks': [masks[1]],
				                   'pos_sample_images': [frames[2]],
				                   'pos_sample_bboxes': [anno['bbox'][2]],
				                   'pos_sample_masks': [masks[2]],
				                   'neg_sample_images': [frames[2]],
				                   'neg_sample_bboxes': [anno['bbox'][2]],
				                   'neg_sample_masks': [masks[2]],
				                   'dataset': dataset.get_name(),
				                   'test_class': meta_obj.get('object_class_name')})
				
				pos = neg = False
				if self.same_anchor:
					pos = True
					neg = True
				elif random.randint(0, self.neg_to_pos) == 0:
					pos = True
				else:
					neg = True

				# make data augmentation
				data = self.processing(data, pos, neg)
				# check whether data is valid
				valid = data['valid']
			except:
				valid = False

		return data


class SequenceDataset(torch.utils.data.Dataset):
	def __init__(self, datasets, p_datasets, samples_per_epoch, seq_len=10, processing=no_processing):
		self.datasets = datasets

		if p_datasets is None:
			p_datasets = [len(d) for d in self.datasets]
		# Normalize
		p_total = sum(p_datasets)
		self.p_datasets = [x / p_total for x in p_datasets]

		self.samples_per_epoch = samples_per_epoch
		self.seq_len = seq_len
		self.processing = processing

	def __len__(self):
		return self.samples_per_epoch

	def _sample_frame_ids(self, visible, num_ids=1, interval_ids=10, allow_invisible=False, avoid_id=None, min_id=None,
	                      max_id=None):
		""" Samples num_ids frames between min_id and max_id for which target is visible

		args:
			visible - 1d Tensor indicating whether target is visible for each frame
			num_ids - number of frames to be samples
			min_id - Minimum allowed frame number
			max_id - Maximum allowed frame number

		returns:
			list - List of sampled frame numbers. None if not sufficient visible frames could be found.
		"""
		if num_ids == 0:
			return []
		if min_id is None or min_id < 0:
			min_id = 0
		if max_id is None or max_id > len(visible):
			max_id = len(visible)
		# get valid ids

		if allow_invisible:
			valid_ids = [i for i in range(min_id, max_id)]
		else:
			valid_ids = [i for i in range(min_id, max_id) if visible[i]]

		if avoid_id is not None:
			avoid_len = min(100, len(valid_ids) // 3)
			avoid_ids = [i for i in range(max(0, avoid_id - avoid_len), min(len(valid_ids), avoid_id + avoid_len))]
			valid_ids = list(filter(lambda x: x not in avoid_ids, valid_ids))

		# No visible ids
		if len(valid_ids) < num_ids:
			return None

		ids = random.sample(valid_ids, k=num_ids)

		return sorted(ids)

	def __getitem__(self, index):
		"""
		returns:
			TensorDict - dict containing all the data blocks
		"""
		valid = False
		while not valid:
			# Select a dataset
			dataset = random.choices(self.datasets, self.p_datasets)[0]
			is_video_dataset = dataset.is_video_sequence()

			if not is_video_dataset:
				continue

			# sample a sequence
			seq_id = random.randint(0, dataset.get_num_sequences() - 1)
			seq_info_dict = dataset.get_sequence_info(seq_id)
			visible = seq_info_dict['visible']

			# sample a visible frame
			frame_ids = self._sample_frame_ids(visible, num_ids=self.seq_len, interval_ids=10, allow_invisible=False)

			if frame_ids is None or len(frame_ids) != self.seq_len:
				continue

			try:
				frames, anno, meta_obj = dataset.get_frames(seq_id, frame_ids, seq_info_dict)
				H, W, _ = frames[0].shape
				masks = anno['mask'] if 'mask' in anno else [torch.zeros((H, W))] * self.seq_len
				data = TensorDict({'seq_images': frames,
				                   'seq_bboxes': anno['bbox'],
				                   'seq_masks': masks,
				                   'gt_images': [frames[0]],
				                   'gt_bboxes': [anno['bbox'][0]],
				                   'gt_masks': [masks[0]],
				                   'label_images': [frames[-1]],
				                   'label_bboxes': [anno['bbox'][-1]],
				                   'label_masks': [masks[-1]],
				                   'dataset': dataset.get_name(),
				                   'test_class': meta_obj.get('object_class_name')})
				# make data augmentation
				data = self.processing(data)
				# check whether data is valid
				valid = data['valid']
			except:
				valid = False

		return data
