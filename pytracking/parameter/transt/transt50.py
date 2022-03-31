from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()
    params.vit_checkpoint_pth = r'D:\Learning\GitRepository\Clone\pytracking\pytracking\networks\VIT_ep0080.pth.tar'
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.net = NetWithBackbone(net_path='transt50.pth',
                                 use_gpu=params.use_gpu)
    return params
