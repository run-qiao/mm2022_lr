from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()
    params.updater_checkpoint_pth = r'D:\Learning\GitRepository\Clone\mm2022_lr\pytracking\networks\BOXTRAINER_ep0020.pth.tar'
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.net = NetWithBackbone(net_path='transt50.pth',
                                 use_gpu=params.use_gpu)
    return params
