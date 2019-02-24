# import sensortest as ss
#import urx [-1.69626218e-15 , 5.58283578e-16  ,8.53285696e-15  ,1.00000000e+00]]

import numpy as np
import math
import PIL.Image as Image
import array
import h5py
import cv2 as cv
from scipy.ndimage.filters import uniform_filter
import pcl
import os
import threading
import urx
import serial
import time

class SerialData(threading.Thread):  # 创建threading.Thread的子类SerialData
    def __init__(self):
        threading.Thread.__init__(self)     # 初始化线程

    def open_com(self, port, baud):         # 打开串口
        self.ser = serial.Serial(port, baud, timeout=0.5)
        return self.ser

    def com_isopen(self):  # 判断串口是否打开
        return self.ser.isOpen()

    def send_data(self, data):  # 发送数据
        self.ser.write((data+'\r\n').encode())

    def next(self):  # 接收的数据组
        all_data = ''
        all_data = self.ser.readline().decode()  # 读一行数据
        trash = self.ser.readline().decode()
        trash = self.ser.readline().decode()
        print(all_data)
        print('into next')
        return all_data[0] #???? cut \r\n

    def close_listen_com(self):  # 关闭串口
        return self.ser.close()

class Hand():
    def __init__(self, rec_data):
        self.rec_data = rec_data  # 初始化线程
    def grasp(self):
        rec_data.send_data('g')

    def place(self):
        rec_data.send_data('p')

    def suck(self):
        rec_data.send_data('s')

    def loose(self):
        rec_data.send_data('l')

    def handin(self):
        rec_data.send_data('h')

def opening_Hand(rec_data):
    # 搜索匹配字符 ‘/dev/ttyACM0’的设备  connect to arduino
    port = '/dev/ttyACM0'
    baud = 9600
    openflag = rec_data.open_com(port, baud)  # 打开串口

def hdf2affimg(filename):
    """
        Convert hdf5 file(output of affordance map network in lua) to img
        # Notice according to the output of the model we need to resize
    """
    h = h5py.File(filename,'r')

    res = h['results']
    res = np.array(res)
    res = res[0,1,:,:]
    resresize = cv.resize(res, (512, 424), interpolation=cv.INTER_CUBIC)
    resresize[np.where(resresize>1)] = 0.9999
    resresize[np.where(resresize<0)] = 0
    return resresize

def _get_patch(location_2d, cur_color, cur_depth, post_afford, size=(128,128)):
    """
        input the postprocessed affordance map 
        return the 4*128*128 patch (cascade the depth and rgb)
    """
    y = location_2d[0]
    x = location_2d[1]
    r = int(size[0]/2)
    patch_color = cur_color[y-r:y+r,x-r:x+r,:]
    patch_depth = cur_depth[y-r:y+r,x-r:x+r]
    patch_afford = post_afford[y-r:y+r,x-r:x+r]
    patch_rgbd = np.zeros((size[0], size[1], 4))
    patch_rgbd[:,:,0:3] = patch_color
    patch_rgbd[:,:,3] = patch_depth
    return np.transpose(patch_rgbd, (2,0,1)), patch_afford, location_2d

def _postproc_affimg(cur_color, cur_depth, cur_afford):
    """
        # postprocess the affordance map
        # convert from the afford_model from matlab to python
    """
    cur_color = cur_color / 255.0
    cur_depth = cur_depth / 10000
    #print(bg_depth)
    temp = (np.abs(cur_color - bg_color) < 0.1)
    foregroundMaskColor = np.sum(temp, axis = 2) != 3
    backgroundDepth_mask = np.zeros(bg_depth.shape, dtype = bool)
    backgroundDepth_mask[bg_depth!=0] = True
    foregroundMaskDepth = backgroundDepth_mask & (np.abs(cur_depth-bg_depth) > 0.02)
    foregroundMask = (foregroundMaskColor | foregroundMaskDepth)
    #print(np.abs(cur_depth-bg_depth) > 0.5)
    #print(sum(foregroundMask== True))

    x = np.linspace(0,Img_WIDTH-1,Img_WIDTH)
    y = np.linspace(0,Img_HEIGHT-1,Img_HEIGHT)
    pixX,pixY = np.meshgrid(x,y)
    camX = (pixX-intri[0,2])*cur_depth/intri[0,0]
    camY = (pixY-intri[1,2])*cur_depth/intri[1,1]
    camZ = cur_depth
    validDepth = foregroundMask & (camZ != 0) # only points with valid depth and within foreground mask
    inputPoints = [camX[validDepth],camY[validDepth],camZ[validDepth]]
    inputPoints = np.asarray(inputPoints,dtype=np.float32)
    foregroundPointcloud = pcl.PointCloud(np.transpose(inputPoints))
    foregroundNormals = _surface_normals(foregroundPointcloud)
    tmp = np.zeros((foregroundNormals.size, 4), dtype=np.float16)
    for i in range(foregroundNormals.size):
        tmp[i] = foregroundNormals[i]
    foregroundNormals = np.nan_to_num(tmp[:,0:3])
    pixX = np.rint(np.dot(inputPoints[0,:],intri[0,0])/inputPoints[2,:]+intri[0,2])
    pixY = np.rint(np.dot(inputPoints[1,:],intri[1,1])/inputPoints[2,:]+intri[1,2])

    surfaceNormalsMap = np.zeros(cur_color.shape)
    arraySize = surfaceNormalsMap.shape
    pixX = pixX.astype(np.int)
    pixY = pixY.astype(np.int)
    tmp = np.ones(pixY.shape)
    tmp = tmp.astype(np.int)

    surfaceNormalsMap.ravel()[np.ravel_multi_index((pixY,pixX,tmp-1), dims=arraySize, order='C')] = foregroundNormals[:,0]
    surfaceNormalsMap.ravel()[np.ravel_multi_index((pixY,pixX,2*tmp-1), dims=arraySize, order='C')] = foregroundNormals[:,1]
    surfaceNormalsMap.ravel()[np.ravel_multi_index((pixY,pixX,3*tmp-1), dims=arraySize, order='C')] = foregroundNormals[:,2]

    # filter the affordance map
    tmp = _window_stdev(surfaceNormalsMap,25)
    # print(tmp[100:105,100:105,2])
    # accelarate the filter
    meanStdNormals = np.mean(tmp,axis = 2)
    normalBasedSuctionScores = 1 - meanStdNormals / np.max(meanStdNormals)
    cur_afford[np.where(normalBasedSuctionScores < 0.05)] = 0 
    cur_afford[np.where(foregroundMask == False)] = 0 
    post_afford = cv.GaussianBlur(cur_afford,(7,7),7)
    #cv.imshow('aff',post_afford)
    #cv.waitKey(0)
    affordanceMap = post_afford*255

    aff_center = post_afford[border_pos[0]:border_pos[1],border_pos[2]:border_pos[3]]
    aff_max = np.max(aff_center)
    print(' -- Candidate Maximum location {}' .format(np.transpose(np.where(aff_center == aff_max))))
    location = np.transpose(np.where(aff_center == aff_max))[0] + \
        np.asarray([border_pos[0],border_pos[2]])
    tmp = aff_max+np.zeros(post_afford.shape)
    tmp[border_pos[0]:border_pos[1],border_pos[2]:border_pos[3]] = aff_center     
    post_afford = tmp
    # cv.imshow('aff',post_afford)
    #cv.waitKey(0)

    #affordanceMap = post_afford*255
    affordanceMap = np.array(affordanceMap,np.uint8)

    affordanceMap = cv.applyColorMap(affordanceMap,cv.COLORMAP_JET)/255

    cv.imshow('affordanceMap',0.5*cur_color+0.5*affordanceMap)
    # cv.imwrite(affordanceMapPath,255*(0.5*inputColor+0.5*affordanceMap))
    cv.waitKey(500)
    return post_afford,location
    
def _window_stdev( X, window_size):
    """
        std filt in np
    """
    rw = window_size
    k = ((rw*rw)/(rw*rw-1))**(0.5)
    r,c,d = X.shape
    new = np.zeros(X.shape)
    for index in range(0,d):
        XX=X[:,:,index]
        XX+=np.random.rand(r,c)*1e-6
        c1 = uniform_filter(XX, window_size, mode='reflect')
        c2 = uniform_filter(XX*XX, window_size, mode='reflect')
        new[:,:,index] = np.sqrt(c2 - c1*c1)
    return k*new

def _sub2ind( arraySize, dim0Sub, dim1Sub, dim2Sub):
    """
        sub 2 ind
    """
    return np.ravel_multi_index((dim0Sub,dim1Sub,dim2Sub), dims=arraySize, order='C')

def _surface_normals( cloud):
    """
        # calculate the surface normals from the cloud
    """
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.001)  #this parameter influence the speed
    cloud_normals = ne.compute()
    return cloud_normals

def reward_metric(afford_map):
    """
        -- calculate the metric on a single affordance map
        -- for now we weight three values() 
        1. distance between two peaks in the local patch
        2. flatten level of the center peaks
        3. the value of the center peaks
    """
    screen_height = 128
    half_screen = screen_height // 2
    peaklocation = []
    rr = 2
    peakjudge = 0.5

    # define local peak as the maximum num in area with radius = rr
    peaknum = 0
    tmp = np.zeros((2*rr+1, 2*rr+1))
    for i in range(rr,screen_height-rr-1):
        for j in range(rr,screen_height-rr-1):
            tmp[:,:] = afford_map[i-rr:i+rr+1,j-rr:j+rr+1]
            local_value = tmp[rr, rr]
            tmp[rr, rr] = 0
            if local_value > np.max(tmp) and local_value > peakjudge:
                peaknum += 1
                peaklocation.append((i,j))
    peak_dis= 0.
    for i in range(0,peaknum):
        peak_dis += ((peaklocation[i][0]-half_screen) **2 + (peaklocation[i][1]-half_screen) **2) **0.5
    if peaknum == 0:
        reg_peak_dis = 0
    else:
        avg_peak_dis = peak_dis / peaknum
        reg_peak_dis = avg_peak_dis * 1.414 / screen_height

    center_value = afford_map[half_screen][half_screen]
    affmax = center_value + np.zeros(afford_map.shape)
    flatten = np.sum((affmax - afford_map) **2)/(afford_map.size - 1)

    metric = 0.4*(1-reg_peak_dis) + 0.4*flatten + 0.2*center_value
    print(' -- Metric for current frame: %f' %(metric))
    return metric

border_pos = [0,423, 0, 512]
bg_color = cv.imread('./demo/Bg_color.png')/255.0
bg_depth = cv.imread('./demo/Bg_depth.png', -1)/10000
Img_WIDTH = 512
Img_HEIGHT = 424
intri = np.array([[363.253, 0, 254.157], 
                  [0, 363.304, 206.881 ],
                  [0, 0, 1 ]])

robot = urx.Robot("192.168.1.111") 
print('robot connect successfully')
robot_vel = 0.5    #accerlation
robot_acc = 0.3 
hand_height = 0.2 # mile
tran =  [[-0.999746, -0.0196759 ,-0.0109776, -0.0182557],
         [-0.0199747,   0.999414 , 0.0278074,   -0.52257],
         [  0.010424 , 0.0280196,  -0.999553 ,  0.967819 + 0.025],
         [        0  ,        0,          0   ,       1]]

pos_ori = [-0.1,0.06,0.64,0,-3.139,0]
arduino_port = '/dev/ttyACM0*'
rec_data = SerialData()
opening_Hand(rec_data)
hand = Hand(rec_data)

if __name__ == "__main__":

    os.system("th ./infer.lua -imgColorPath demo/0000_color.png -imgDepthPath demo/0000_depth.png -resultPath demo/0000_results.h5")
    cur_afford = hdf2affimg('demo/0000_results.h5')
    cur_color = cv.imread('./demo/0000_color.png')
    cur_depth = cv.imread('./demo/0000_depth.png', -1)
    post_afford, location_2d = _postproc_affimg(cur_color, cur_depth, cur_afford)
    print(' -- Maximum Location at {}' .format(location_2d))
    # according to the affordance map -> get the local patch with size of 4*128*128

    cmd = 1;

    while(cmd != 'n'):
        print ("suc eye pos_2d:")
        print (location_2d)

        sucx = location_2d[1]
        sucy = location_2d[0]
        cur_depth = cur_depth[sucy][sucx]/1000
        x = cur_depth * (sucx - intri[0][2]) / intri[0][0]
        y = cur_depth * (sucy - intri[1][2]) / intri[1][1]
        location_direct = [x, y, cur_depth]
        print(location_direct)

        location_3d = location_direct
        # calculate the hand position: need to fix the tran metrix
        eyepos = [[location_3d[0]], [location_3d[1]], [location_3d[2]],[1]]
        hand_calcu = np.dot(np.linalg.inv(tran),eyepos)

        print('the calculate hand pos_3d use depth is')
        print(hand_calcu)

        cmd = input("go? [y/n]")
        if cmd == 'y':
            pos_in = [hand_calcu[0][0],hand_calcu[1][0],hand_calcu[2][0]+hand_height,0,-3.139,0] # make the hand vertical down and adjust z of xyz
            robot.movel(pos_in, acc = robot_acc,vel = robot_vel)            
            hand.grasp()
            print(rec_data.next())
            time.sleep(1000)
            robot.movel(pos_ori, acc = robot_acc,vel = robot_vel)
            time.sleep(1000)
            hand.place()
            print(rec_data.next())

        elif cmd == 'n':
            break


        '''
        cloud = pcl.load("./demo/0000_cloud.pcd")

        location_3d = [0, 0, 0]
        location_3d[0] = cloud[(Img_HEIGHT-sucx)*Img_WIDTH + sucy][0]
        location_3d[1] = cloud[(Img_HEIGHT-sucx)*Img_WIDTH + sucy][1]
        location_3d[2] = cloud[(Img_HEIGHT-sucx)*Img_WIDTH + sucy][2]

        # bad point check : if the point in the point cloud is a bad point ,it should be [0.00022,-0.00006,0.00005].than try its neighbor
        bp_limit = 1e-4
        while(abs(location_3d[0]) < bp_limit and abs(location_3d[1]) < bp_limit and location_3d[2] < bp_limit):
            print('warning:a bad point in pcl, find its neighbor ...')
            sucx += 1
            sucy += 1
            location_3d[0] = cloud[(Img_HEIGHT-sucx)*Img_WIDTH + sucy][0]
            location_3d[1] = cloud[(Img_HEIGHT-sucx)*Img_WIDTH + sucy][1]
            location_3d[2] = cloud[(Img_HEIGHT-sucx)*Img_WIDTH + sucy][2]
        print('suc eye pos_3d:')
        # 0.189 0.586 0.212
        print(location_3d)

        eyepos = [[location_3d[0]], [location_3d[1]], [location_3d[2]],[1]]
        hand_calcu = np.dot(np.linalg.inv(tran),eyepos)

        print('the calculate hand pos_3d use cloud is')
        print(hand_calcu)
        '''
 



