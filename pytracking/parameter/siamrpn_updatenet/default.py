# -*- coding=utf-8 -*-
# @Time :2022/4/2 12:51
# @Author :Run Luo
# @Site : 
# @File : default.py
# @Software : PyCharm


from pytracking.utils import TrackerParams


def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = True
    params.use_gpu = True
    params.siam_path=r'D:\Learning\GitRepository\Clone\pytracking\pytracking\networks\SiamRPNBIG.model'
    params.update_model_path=r'D:\Learning\GitRepository\Clone\pytracking\pytracking\networks\vot2018.pth.tar'
    return params