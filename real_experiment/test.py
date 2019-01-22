import numpy as np
import pcl
import cv2 as cv

#import matlab
#import matlab.engine

import sensortest as ss
#import urx

#set
height = 424
width = 512
tran = [[ 6.66014493e-01 ,3.34576756e-02 , 1.35094322e-01 ,-1.26152930e-01],
 [ 7.33478322e-02, -6.47372228e-01 ,-7.38042072e-02 ,-5.49179663e-01],
 [-1.79801804e-01 ,-5.25346497e-02 ,-2.57603813e-01 , 4.17893897e-01],
 [-1.69626218e-15 , 5.58283578e-16  ,8.53285696e-15  ,1.00000000e+00]]

#main
cmd = 1;
#engine = matlab.engine.start_matlab()
import os
os.system("th ./infer.lua -imgColorPath demo/0001_color.jpg -imgDepthPath demo/0001_depth.png -resultPath demo/0000_results.h5")

'''
#while(cmd != 0):
	location = engine.visualize()
	print "suc pos:"
	print location

	sucx = location[0][0]
	sucy = location[0][1]
	cloud = pcl.load("/home/lukai/vision_data/0000_cloud.pcd")
	sucx_fix = cloud[(height-sucy)*width + sucx][0]
	sucy_fix = cloud[(height-sucy)*width + sucx][1]
	sucz_fix = cloud[(height-sucy)*width + sucx][2]

	#color_img = cv.imread("/home/lukai/vision_data/0000_color.png")


	print('eye pos')
	neweye = [[sucx_fix], [sucy_fix], [sucz_fix],[1]]
	hand_calcu = np.dot(tran,neweye)

	print('the calculator answer is')
	print(hand_calcu)

	hand_calcu[2][0] += 0.2
	print('the hand calculator answer fix is')
	print(hand_calcu) 

	cmd = input("go?")

	posin = [hand_calcu[0][0],hand_calcu[1][0],hand_calcu[2][0],0,-3.139,0]
	ss.mymain(posin)
        print('quit?')
'''



