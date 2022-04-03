from pytracking.tracker.base import BaseTracker
import torch
import numpy as np
import time
from pytracking.tracker.siamrpn_updatenet.net_upd import SiamRPNBIG
from pytracking.tracker.siamrpn_updatenet.updatenet import UpdateResNet
from pytracking.tracker.siamrpn_updatenet.run_SiamRPN_upd import SiamRPN_init, SiamRPN_track_upd
from pytracking.tracker.siamrpn_updatenet.utils import cxy_wh_2_rect
import cv2

class SiamRPN_UPDATENET(BaseTracker):

    multiobj_mode = 'parallel'
    def __init__(self,params):
        super(SiamRPN_UPDATENET,self).__init__(params)
        self.features_initialized = False

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net=SiamRPNBIG()
            self.net.load_state_dict(torch.load(self.params.siam_path))
            self.net.eval().to(self.params.device)
            self.updatenet = UpdateResNet()
            self.updatenet.load_state_dict(torch.load(self.params.update_model_path)['state_dict'])
            self.updatenet.eval().to(self.params.device)
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        if not self.features_initialized:
            # Initialize network
            self.initialize_features()
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


        # Time initialization
        tic = time.time()

        # Get target position and size
        state = info['init_bbox']
        self.pos = np.array([state[0] + (state[2] - 1)/2,state[1] + (state[3] - 1)/2])
        self.target_sz = np.array([state[2],state[3]])

        self.state=SiamRPN_init(image, self.pos, self.target_sz, self.net)

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert image
        self.state = SiamRPN_track_upd(self.state, image, self.updatenet)
        res = cxy_wh_2_rect(self.state['target_pos'], self.state['target_sz'])
        out = {'target_bbox': [res[0], res[1], res[2], res[3]]}
        return out
