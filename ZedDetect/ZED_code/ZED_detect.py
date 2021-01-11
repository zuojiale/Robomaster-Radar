'''
##作者:左家乐
##日期：2020-07-30
##功能：本文件实现利用ZED相机实时的对赛车进行识别检测，并用socket传送每一辆赛车的世界坐标至map程序进行刷新
##说明：zed提取的是相机坐标系下的，需要转换到世界坐标系
'''

# Deep learning group's  hpp
import cv2
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils import array_tool as at
from PIL import Image
from utils.vis_tool import visdom_bbox
import numpy as np

# ZED's  hpp
import pyzed.sl as sl
import sys

#socket' hpp
import paramiko
import socket

#zuobiao's hpp
from ZuoBiao_Trans import camera2world as c2w
import random

#ignore warning
import warnings 


kind_list = ['aeroplane',  'bicycle' ,'bird' ,'boat ','bottle' ,'bus' ,'car' ,'cat' ,'chair' ,'cow' ]


def socket_init(IP,Port):
	try:
		global server
		server = socket.socket() # 有一些默认参数，即可使用ipv4，这一句是声明socket类型和返回socket连接对象
		server.connect((IP ,Port))
	except:
		print("1.The client isn't ready!")
		return -1
	else:
		print("1.Socket is ready!")
	
	
def zed_init():
	'''
	##作者:左家乐
	##日期：2020-08-01
	##功能：Init the ZED
	##IN-para : no
	##return ： err()
	'''
	camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
	str_camera_settings = "BRIGHTNESS"
	step_camera_settings = 1
	print("3.Detected the ZED...")
	global cam 
	cam = sl.Camera()
	init_params = sl.InitParameters()
	init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
	init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
	init_params.camera_resolution = sl.RESOLUTION.HD720
	err = cam.open(init_params)  #if failed to open zed ,return 0
	global runtime_parameters
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
	runtime_parameters.confidence_threshold = 100
	runtime_parameters.textureness_confidence_threshold = 100   
	global mat
	mat = sl.Mat()
	global depth
	depth = sl.Mat()
	global point_cloud
	point_cloud = sl.Mat()
	mirror_ref = sl.Transform()
	mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
	tr_np = mirror_ref.m
	return err
		
		
def faster_rcnn_init():
	'''
	##作者:左家乐
	##日期：2020-08-01
	##功能：load faster-rcnn network
	##IN-para : no
	##return ： err
	'''
	try:
		opt.caffe_pretrain = True
		faster_rcnn = FasterRCNNVGG16()
		global trainer
		trainer = FasterRCNNTrainer(faster_rcnn).cuda()
		trainer.load('/home/zjl/python_zed/4network/copy_simple_fast/checkpoints/fasterrcnn_08042317_0.9090909090909093')  
	except:
		print('2.Failed to load Network model!please CHECK!')
		return -1
	else:
		print('2.Network loads succeed...')
		return 0
	

def img_preprocess(f,dtype=np.float32, color=True):
	'''
	##作者:左家乐
	##日期：2020-07-30
	##功能：该函数在神经网络识别之前，对原始图片进行预处理
	##IN-para : 一帧ZED数据
	##return ： 预处理后的img
	'''
	try:
		if color:
		    img = f.convert('RGB')
		else:
		    img = f.convert('P')
		img = np.asarray(img, dtype=dtype)
	finally:
		if hasattr(f, 'close'):
		    f.close()
	if img.ndim == 2:
		return img[np.newaxis]
	else:
		return img.transpose((2, 0, 1))



def Recognize(img):
	'''
	##作者:左家乐
	##日期：2020-07-30
	##功能：该函数用来检测目标赛车roboot
	##IN-para : 一帧ZED数据
	##return ： 车的目标框坐标
	'''
	img = t.from_numpy(img)[None]
	opt.caffe_pretrain=False       # this model was trained from caffe-pretrained model
	_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
	bboxes = at.tonumpy(_bboxes[0])
	labels = at.tonumpy(_labels[0].reshape(-1))
	return bboxes,labels


def stable_object_cam_zb(bbox):#传进来框的左上角和右下角两个点的像素坐标（x,y）
	#取的9个点的像素坐标
	Z_center_x = (bbox[1]+bbox[3])/2
	Z_center_y = (bbox[0]+bbox[2])/2
	#算出新矩形两个顶点坐标
	Rect_x1 = (bbox[1]+Z_center_x)/2
	Rect_y1 = (bbox[0]+Z_center_y)/2
	Rect_x2 = (Z_center_x+bbox[3])/2
	Rect_y2 = (Z_center_y+bbox[2])/2
	
	CAM = []
	lenth = 0
	while lenth<5:
		i = Rect_x1+ (Rect_x2-Rect_x1)*random.random()
		j = Rect_y1+ (Rect_y2-Rect_y1)*random.random()
		err,cam = point_cloud.get_value(round(i),round(j))
		if not np.isnan(cam[0]) and not np.isinf(cam[0]):
			CAM.append([cam[0],cam[1],cam[2]])
			lenth = lenth+1

	CAM = np.array(CAM)	
	print(CAM)
	#筛选Z坐标
	Z_array = []
	for cam in CAM:
		Z_array.append(cam[2])
	Z_array = np.array(Z_array)
	Z_array.sort()
	print(Z_array)
	
	#取排序最中间的三个点的坐标
	zuobiao1 = []
	zuobiao2 =[]
	zuobiao3 = []
	for i in range(5):
		if Z_array[1]==CAM[i][2]:
			zuobiao1 = CAM[i]
			break
	for i in range(5):
		if Z_array[2]==CAM[i][2]:
			zuobiao2 = CAM[i]
			break
	for i in range(5):
		if Z_array[3]==CAM[i][2]:
			zuobiao3 = CAM[i]
			break
	#取排序9个坐标中最中间3个点坐标，取其x、y、z平均值为最终返回的坐标（x、y、z）
	X = (zuobiao1[0]+zuobiao2[0]+zuobiao3[0])/3
	Y = (zuobiao1[1]+zuobiao2[1]+zuobiao3[1])/3
	Z = (zuobiao1[2]+zuobiao2[2]+zuobiao3[2])/3
	stable_cam_zb = [X,Y,Z]
	return stable_cam_zb



if __name__ == '__main__':
	'''
	##作者:左家乐
	##日期：2020-07-30
	##功能：主函数，实现对zed相机初始化、调用，识别赛车，转换坐标系，socket传送数据
	'''
	warnings.filterwarnings('ignore')
	
	#1.Timer count
	Timer_count = 0

	#2.socket ready
	
	sc_err  = socket_init("222.195.67.69",6969)
	if sc_err == -1:
		exit(1)
	
	
	#3.加载神经网络
	#opt.env = 'test'
	err = faster_rcnn_init()
	if err == -1:
		exit(1)


	# 4.Init the camera ZED
	err = zed_init()
	if err != sl.ERROR_CODE.SUCCESS:
		exit(1)  #if failed, then quit the code
	else:
		print("4.Opening ZED Camera...")


	#5. while loop
	key = ' '
	while key != 113:         #for Q key to quit
		err = cam.grab(runtime_parameters)
		if err == sl.ERROR_CODE.SUCCESS:
		    cam.retrieve_image(mat, sl.VIEW.LEFT)
		    cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
		    cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
		    key = cv2.waitKey(5)
		    Timer_count = Timer_count + 1
		else :
		    key = cv2.waitKey(5)
		    Timer_count =  Timer_count + 1
		
		##6.定时处理一帧数据，目标识别，坐标转换，传输数据
		if Timer_count == 40:  #Timer 
			image = Image.fromarray(cv2.cvtColor(mat.get_data(), cv2.COLOR_BGR2RGB))  #cv->pil format tranform
			image = img_preprocess(image)  
			bboxes,labels  = Recognize(image)
			img=mat.get_data()
			
			screen_zb_list = []
			for  i in range(bboxes.shape[0]):
				img = cv2.rectangle(img , (bboxes[i,1],bboxes[i,0]) , (bboxes[i,3],bboxes[i,2]), (0,0,255), 2)
				cam_zb = stable_object_cam_zb(bboxes[i,:])
				world_zb = c2w.convert_cc_to_wc(cam_zb)
				screen_zb = c2w.convert_wc_to_sc(world_zb)
				screen_zb_list_tmp = str(i+1)+","+ str(screen_zb[0]) +"," +str(screen_zb[1]) 
				screen_zb_list.append(screen_zb_list_tmp)
			
			screen_zb_list  =  "/".join(screen_zb_list)
			server.send(screen_zb_list.encode(encoding='utf-8'))
			
			cv2.imshow("Recognize", img)
			Timer_count = 0

	#server.close()
	cv2.destroyAllWindows()
	cam.close()
			 
		      
		    
			

	
	
	

