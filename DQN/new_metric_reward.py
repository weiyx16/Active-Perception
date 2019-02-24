#coding=utf-8 
import cv2 as cv
import numpy as np 
from skimage import measure
import copy
import PIL.Image as img
import math
from scipy import optimize
import matplotlib.pyplot as plt 

def peak_dis(self,ilabel,iaff):
    maxnum = np.max(ilabel)
    center = [] #record centers of connected area except (64,64)
    peaknum = 0
    peak_dis = 100000
    half_screen = 64
    screen_height = 128
    screen_width = 128
    peakjudge = 0.5

    #using connected area:
    for i in range(1,maxnum):
        if(np.sum(ilabel == i) > 50 and i != ilabel[64,64]):
            point_arr = np.transpose(np.where(ilabel == i))
            cnt = cv.minAreaRect(point_arr)
            if iaff[int(cnt[0][0]),int(cnt[0][1])] > 0.5: ## the limit
                peaknum += 1
                center.append(cnt[0])

    point_arr = np.transpose(np.where(ilabel == ilabel[64,64]))
    cnt = cv.minAreaRect(point_arr)
    distance_refer = (cnt[1][0]+cnt[1][1]) * 0.5 #refer = (h+w)/2 ;h,w based on the rect of the center area

    
    #using peak of 5*5 window
    rr = 2
    tmp = np.zeros((2*rr+1, 2*rr+1))
    for i in range(rr,screen_height-rr-1):
        for j in range(rr,screen_width-rr-1):
            tmp[:,:] = iaff[i-rr:i+rr+1,j-rr:j+rr+1]
            local_value = tmp[rr, rr]
            tmp[rr, rr] = 0
            if local_value > np.max(tmp) and i!= 64 and j !=64 and local_value > peakjudge:
                peaknum += 1
                center.append((i,j))
    #cal the min peak distance
    for i in range(0, peaknum):
        tmp_dis = ((center[i][0]-half_screen) **2 + (center[i][1]-half_screen) **2) **0.5
        if peak_dis > tmp_dis and tmp_dis > 1:
            peak_dis = tmp_dis

    if peaknum < 1:
        reg_peak_dis = 1.
    else:
        # avg_peak_dis = peak_dis / peaknum
        reg_peak_dis = peak_dis / distance_refer
        if reg_peak_dis >1:
            reg_peak_dis = 1
    return reg_peak_dis

def gauss(x,y,ex,ey,vx,vy):
    return np.exp((-(x-ex)**2)/(2*vx**2)+(-(y-ey)**2)/(2*vy**2))

def flatness(self,iaff,icenter,cnt):
    iaff_tmp = copy.deepcopy(iaff)
    iaff_tmp[np.where(icenter == 0)] = 0

    igauss = np.zeros((128,128))
    angle=cnt[2]/180*math.pi

    x = 64
    y = 64
    xn = x*math.cos(angle) - y*math.sin(angle)
    yn = x*math.sin(angle) + y*math.cos(angle)
    center = [yn,xn]

    para = 1.5*2 # based on gauss(1.5) = 32.46%
    for x in range(0,128):
        for y in range(0,128):
            xn = x*math.cos(angle) - y*math.sin(angle)
            yn = x*math.sin(angle) + y*math.cos(angle)
            igauss[y,x] = gauss(xn,yn,center[1],center[0],cnt[1][1]/para,cnt[1][0]/para)
    
    #igauss = (igauss*((np.max(iaff_tmp)-np.min(iaff_tmp)))+np.min(iaff_tmp))
    score = (np.sum(np.abs((iaff_tmp - igauss)**2)/1)/(iaff_tmp.size - 1))**0.5
    return math.exp(1-score)/math.exp(1)

def reward_metric(self,afford_map)
        """
            -- calculate the metric on a single affordance map
            -- for now we weight three values() 
            1. distance between two peaks in the local patch
            2. flatten level of the center peaks
            3. the value of the center peaks
            # TODO:
        """
    limit = 0.4 #32.5*255/100

    iaff_gray_ori = afford_map # 0~1 aff

    iaff_gray_bm = copy.deepcopy(iaff_gray_ori)
    iaff_gray_bm[np.where(iaff_gray <limit)] = 0
    iaff_gray_bm[np.where(iaff_gray >limit)] = 1

    ilabel = measure.label(iaff_gray_bm,connectivity = 1) #8-connect,original file use aff(255), but here using aff(1.0)
    
    icenter = copy.deepcopy(iaff_gray_bm) 
    icenter[np.where(ilabel == ilabel[64,64])] = 1
    icenter[np.where(ilabel != ilabel[64,64])] = 0

    point_arr = np.transpose(np.where(icenter == 1))
    cnt = cv.minAreaRect(point_arr)

    ipeak_dis = self.peak_dis(ilabel,iaff_gray_ori)
    iflatness = self.flatness(iaff_gray_ori,icenter,cnt)
    icenter_value = np.max(iaff)

    if ipeak_dis == 1:
        reward_metrix = 0.9*iflatness+ 0.1*icenter_value
    else:
        reward_metrix = 0.75*iflatness+ 0.15 * ipeak_dis +0.1*icenter_value

    print(' -- Metric for current frame: %f \n   With peak_dis: %f, flatten: %f, max_value: %f' %(reward_metrix, ipeak_dis, iflatness, icenter_value))

