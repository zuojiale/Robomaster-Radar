'''
Code：数据接收端程序
Author: 江培玲、左家乐
Function：从服务器接收数据，并显示地图
Date：2020-08-01
'''

import pygame
from pygame.locals import *
import numpy as np
import socket
server = socket.socket()

class Player:
	'''
	定义玩家类，包括玩家名、位置及照片
	'''
	playerpos = []			#位置数组
	def sucai(self,num):
		self.player = pygame.image.load('素材/'+str(num)+'.png').convert_alpha()		#照片

def Map_refresh():
	'''		
	位置实时刷新
	'''
	screen.fill(0)
	screen.blit(black,(0,0))
	global playerlist		#全局变量，加上global关键字便可访问全局变量playerlist

	for Player in playerlist:	
		screen.blit(Player.player,Player.playerpos)
		pygame.display.flip()

    
def Str2Mat(st):
	'''		
	功能：将传来的字符串转换成数字并存放到矩阵中。
	传入：字符串 
	Return：转换后的矩阵
	指定的通讯协议为：zed发送端发送过来的数据	格式为：1,111,127/3,267,435/3,678,365诸如此类的字符串st。
	第一位数据为玩家编号（1-8为红方，9-16为蓝方），第二、三位为X、Y坐标	
	'''
	st_list = st.split('/')	#将上述字符串按'/'分割，变成三段字符串的列表：‘1,111,127’，‘3,267,435’，‘3,678,365’
	print(st_list)				#为一个list
	row = len(st_list)			#列表有几段就代表有几个玩家，上述列表有三段，这也是矩阵的行数
	#print(row)					#统计列表里元素个数
	mat = [[0]*3 for _ in range(row)]		#row*3列的矩阵并初始化为0
	mat =np.array(mat)
	#print(mat)
	j = 0
	for i in st_list:
		num_list = i.split(',')
		print(num_list)
		mat[j,0] = float(num_list[0])
		mat[j,1] = float(num_list[1])
		mat[j,2] = float(num_list[2])
		j = j+1
	return mat	
 
def Data_Recv():
	'''		
	功能：监听服务器端传来的数据
	传入：空
	结束：输出将字符串转换成的矩阵
	说明：用socket监听服务器，将接收到的data1解码，并将解码后的字符串转换成所需的位置矩阵data
	'''
	global data
	while True:
		data1 = conn.recv(1024)
		if not data1:
		    print('The server is end,exit!\n ')
		    break
		data1 = data1.decode()				#将传来的数据解码，避免乱码
		print('data:',data1)
		data = Str2Mat(data1)
		break
		
    
if __name__ == "__main__":
	#1、socket初始化
	server.bind(("222.195.67.69",6969)) 	#绑定要监听的端口port
	server.listen(1) 						#监听，这里表示最多有1个客户端连接服务器，python2不好使
	print('等待发送程序上线.....')
	
	while True:
   		conn,addr = server.accept() 		#等待连接
	   	print('数据发送程序已上线！等待接受数据....')
	   	break
	
	#2、数据初始化以及socket接收   	
	data = []
	Data_Recv()
		
	#3、地图初始化
	pygame.init()																	
	screen=pygame.display.set_mode([1323,711])
	black=pygame.image.load('素材/map1.png')
	
	#4、刷新地图显示数据
	while 1:
		robot_num = data.shape[0]			#传入的数据多少行就表示多少个玩家
		playerlist = []						#定义 存放data中出现的玩家
		for i in range(robot_num):
			for j in range(17):	 			#种类+1,0-16	
				if(data[i,0] == j):			#一旦检索到编号，进入下面
					#创建一个新的player对象，并以player(j)命名
					locals()['player' + str(j)] = Player()		
					#对该玩家对象的位置及图片赋值
					locals()['player' + str(j)].playerpos = [data[i,1],data[i,2]]
					locals()['player' + str(j)].sucai(j)
					#该对象出现，记录在playerlist里面
					playerlist.append(locals()['player' + str(j)])
					break	
		
		#5、刷新地图		
		Map_refresh()
		data = []
		Data_Recv()
		
	
	
		
		
			
			
			
		
		
	
