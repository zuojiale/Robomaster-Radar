# -*- coding: UTF-8 -*- 

import numpy as np
from numpy import *
import matplotlib.pyplot as plt



ZED_Transform_Matrix = {
    # R，旋转矩阵
	"R": [[1,0,0],
			[0,0,1],
			[0,-1,0]],
    # T' = T *R，平移向量,此处的T为相机对应的世界坐标系下的（相反数！！！！）
    "T": [-3.62 , 0.85 , 0.15]
}

camera_intrinsic = ZED_Transform_Matrix
   

	
def convert_cc_to_wc(joint_cam):
    """
	相机坐标系 -> 世界坐标系: inv(R) * pt +T 
	joint_wd = np.dot(inv(R), joint_ca.T)+T
	:return:
	"""
    joint_cam = np.asarray(joint_cam)
    R = np.asarray(camera_intrinsic["R"])
    T = np.asarray(camera_intrinsic["T"])
    joint_num = len(joint_cam)
    joint_world = np.dot(R, (joint_cam - T).T).T  # R * (pt - T)
    return joint_world	
    
def convert_wc_to_sc(joint_world):
	"""
	世界坐标（X_w,Y_w,Z_w） -> 客户端显示屏幕坐标(X_s,Y_s) 
	:return:
	"""
	joint_world = np.asarray(joint_world)
	joint_screen = [47.25*joint_world[1], 47.4*joint_world[0]]
	return joint_screen
	
    
    
    
