# -*- coding=utf-8 -*-
# @Time :2022/4/2 12:47
# @Author :Run Luo
# @Site : 
# @File : __init__.py.py
# @Software : PyCharm

from .siamrpn_updatenet import SiamRPN_UPDATENET

def get_tracker_class():
    return SiamRPN_UPDATENET