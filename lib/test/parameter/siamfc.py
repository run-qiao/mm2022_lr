from lib.test.utils import TrackerParams
import os

def parameters(yaml_name=None):
	params = TrackerParams()

	params.save_all_boxes = False
	params.checkpoint_pth = os.path.abspath(
		os.path.join(os.getcwd(), "..")) + '/checkpoints/siamfc/siamfc_alexnet_e50.pth'

	params.vit_checkpoint_pth = os.path.abspath(
		os.path.join(os.getcwd(), "..")) + '/output/checkpoints/train/vit/base_224/VIT_ep0070.pth.tar'

	return params
