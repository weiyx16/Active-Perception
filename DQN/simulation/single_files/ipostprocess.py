import math
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

import time
import numpy.random as random
import numpy as np
import math
import PIL.Image as Image
import array
import h5py
import cv2 as cv
#import dlm

import pcl

pi = 3.1415926
iwait = 5000  #ms for keeping imshow
backgroundColorImage = '../vision_data/back_color.png';   # 24-bit RGB PNG
backgroundDepthImage = '../vision_data/back_depth.png';   # 16-bit PNG depth in deci-millimeters
inputColorImage = '../vision_data/0004_color.jpg';             # 24-bit RGB PNG
inputDepthImage = '../vision_data/0004_depth.png';             # 16-bit PNG depth in deci-millimeters
cameraIntrinsicsFile = '../suction-based-grasping/convnet/demo/test-camera-intrinsics.txt';  # 3x3 camera intrinsics matrix
resultsFile = './0004_results.h5'; # HDF5 ConvNet output file from running i
affordanceMapPath = '../aff_nor/0004_iresults.png'
normalsPath = '../aff_nor/0004_inormals.png'
localaffPath = '../local_patch/0004_ilocalaff.png'
localrgbPath = '../local_patch/0004_ilocalrgb.png'

def hdf2affimg(filename):
    h = h5py.File(filename,'r')
    res = h['results']
    res = np.array(res)
    res = res[0,1,:,:]    
    resresize = cv.resize(res, (512, 424), interpolation=cv.INTER_CUBIC)
    resresize[np.where(resresize>1)] = 0.9999
    resresize[np.where(resresize<0)] = 0
    return resresize

def Surface_normals(cloud):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.001)  #this parameter influence the speed
    print(ne)
    cloud_normals = ne.compute()
    return cloud_normals

def sub2ind(arraySize, dim0Sub, dim1Sub, dim2Sub):
    #return np.ravel_multi_index((1, 0, 1), dims=(3, 4, 2), order='C')
    return np.ravel_multi_index((dim0Sub,dim1Sub,dim2Sub), dims=arraySize, order='C')

def stdfilt(map,filter):
    h,w,d = map.shape
    half_filter = int(filter.shape[0]/2)
    tmp = np.ones((h,half_filter,d))
    newmap = np.hstack((tmp,map,tmp))
    tmp = np.ones((half_filter,w+2*half_filter,d))
    newmap = np.vstack((tmp,newmap,tmp))
    ians = np.zeros(map.shape)
    for i in range(0,h-1):
        for j in range(0,w-1):
            tmpmap = newmap[i:i+2*half_filter+1,j:j+2*half_filter+1,:]
            ians[i,j,:] = np.std(tmpmap)
    return ians

def visualize(resultsFile,inputColorImage,inputDepthImage,backgroundColorImage,backgroundDepthImage,cameraIntrinsicsFile):
    affordanceMap = hdf2affimg(resultsFile)
    inputColor = cv.imread(inputColorImage)/255.
    inputDepth = cv.imread(inputDepthImage,0)/10000.*256.0
    backgroundColor = cv.imread(backgroundColorImage)/255.
    backgroundDepth = cv.imread(backgroundDepthImage,0)/10000.*256.0
    #cameraIntrinsics = dlm.dlmread(cameraIntrinsicsFile)
    cameraIntrinsics = np.array([[616.521545,0.,311.354492],
                                 [0.,616.521606,231.087402],
                                 [0.,0., 1.]])

    a_abs = (np.abs(inputColor-backgroundColor) < 0.3)
    a_abs_sum = np.sum(a_abs, axis = 2)
    foregroundMaskColor = np.logical_not(a_abs_sum == 3)

    backgroundDepth_mask = np.zeros(backgroundDepth.shape, dtype = bool)
    backgroundDepth_mask[backgroundDepth!=0] = True
    foregroundMaskDepth = backgroundDepth_mask & (np.abs(inputDepth-backgroundDepth) > 0.02)
    foregroundMask = (foregroundMaskColor | foregroundMaskDepth) # correct

    x = np.linspace(0,511,512)
    y = np.linspace(0,423,424)
    pixX,pixY = np.meshgrid(x,y)

    camX = (pixX-cameraIntrinsics[0,2])*inputDepth/cameraIntrinsics[0,0]
    camY = (pixY-cameraIntrinsics[1,2])*inputDepth/cameraIntrinsics[1,1]
    camZ = inputDepth
    print("foregroundMask.shape is %s"%str(foregroundMask.shape))
    validDepth = foregroundMask & (camZ != 0); # only points with valid depth and within foreground mask
    inputPoints = [camX[validDepth],camY[validDepth],camZ[validDepth]]
    inputPoints = np.asarray(inputPoints,dtype=np.float32) ## have some difference with the matlab data
    foregroundPointcloud = pcl.PointCloud(np.transpose(inputPoints))
    print('normals calculating ...')
    foregroundNormals = Surface_normals(foregroundPointcloud)
    print('done')

    tmp = np.zeros((foregroundNormals.size,4))
    for i in range(0,foregroundNormals.size-1):
        tmp[i] = foregroundNormals[i]
    foregroundNormals = np.nan_to_num(tmp[:,0:3])

    sensorCenter = np.array([0,0,0])
    for k in range(0,inputPoints.shape[1]-1):
        p1 = sensorCenter - np.array([inputPoints[0,k],inputPoints[1,k],inputPoints[2,k]])
        p2 = np.array([foregroundNormals[k,0],foregroundNormals[k,1],foregroundNormals[k,2]])
        angle = np.arctan2(np.linalg.norm(np.cross(p1,p2)),np.transpose(np.dot(p1,p2)))
        if angle > pi/2 or angle < -pi/2:
            0
        else:
            foregroundNormals[k,:] = -foregroundNormals[k,:]

    pixX = np.rint(np.dot(inputPoints[0,:],cameraIntrinsics[0,0])/inputPoints[2,:]+cameraIntrinsics[0,2])
    pixY = np.rint(np.dot(inputPoints[1,:],cameraIntrinsics[1,1])/inputPoints[2,:]+cameraIntrinsics[1,2])

    surfaceNormalsMap = np.zeros(inputColor.shape)

    pixX = pixX.astype(np.int)
    pixY = pixY.astype(np.int)
    tmp = np.ones(pixY.shape)
    tmp = tmp.astype(np.int)

    surfaceNormalsMap.ravel()[sub2ind(surfaceNormalsMap.shape,pixY,pixX,tmp-1)] = foregroundNormals[:,0]
    surfaceNormalsMap.ravel()[sub2ind(surfaceNormalsMap.shape,pixY,pixX,2*tmp-1)] = foregroundNormals[:,1]
    surfaceNormalsMap.ravel()[sub2ind(surfaceNormalsMap.shape,pixY,pixX,3*tmp-1)] = foregroundNormals[:,2]

    print('stdfilt calculating ...')
    meanStdNormals = np.mean(stdfilt(surfaceNormalsMap,np.ones((25,25))),axis = 2)
    print('done')
    normalBasedSuctionScores = np.ones(meanStdNormals.shape) - meanStdNormals / np.nanmax(meanStdNormals)

    affordanceMap[np.where(normalBasedSuctionScores < 0.05)] = 0 
    affordanceMap[np.where(foregroundMask == False)] = 0 
    affordanceMap = cv.GaussianBlur(affordanceMap,(7,7),7)
    print('affordanceMap calculating ... done')
    
    afmap = affordanceMap
    afmax = np.nanmax(afmap)
    location = np.where(afmap == afmax)
    location_2d = np.transpose(np.array(location))[0]

    print('max location is %s'%str(location_2d))
    print('max affordances is %s'%str(np.max(affordanceMap)))
   
    affordanceMap = affordanceMap*255
    affordanceMap = np.array(affordanceMap,np.uint8)

    affordanceMap = cv.applyColorMap(affordanceMap,cv.COLORMAP_JET)/255
    cv.imshow('affordanceMap',0.5*inputColor+0.5*affordanceMap)
    cv.imshow('normals',surfaceNormalsMap)
    cv.imwrite(affordanceMapPath,0.5*inputColor+0.5*affordanceMap)
    cv.imwrite(normalsPath,surfaceNormalsMap)
    cv.waitKey(iwait)
    print('save ... done')
    return affordanceMap,location_2d,inputColor,inputDepth,afmap #!!max is not accurate ,find an area will be better

def local_patch(size = (128,128),num_of_patch = 1,inputColor = 1,inputDepth = 1,location_2d = 1,affordanceMap = 1,afmap = 1):
    for i in range(0,num_of_patch):
        print(i)
        y = location_2d[0]
        x = location_2d[1]
        r = int(size[0]/2)
        tmpcolor = inputColor[y-r:y+r,x-r:x+r,:]
        tmpdepth = inputDepth[y-r:y+r,x-r:x+r]
        localrgb = tmpcolor
        localrgbd = np.zeros((2*r,2*r,4))
        localrgbd[:,:,0:3] = tmpcolor
        localrgbd[:,:,3] = tmpdepth
        localaff_rgb = affordanceMap[y-r:y+r,x-r:x+r,:]
        localaff_gray = afmap[y-r:y+r,x-r:x+r]

        print('local_patching ... done')
        cv.imshow('localaff',0.5*localaff_rgb+0.5*localrgb)
        cv.waitKey(iwait)
        cv.imwrite(localaffPath,0.5*localaff_rgb+0.5*localrgb)
        cv.imwrite(localrgbPath,localrgb)
        print(localrgbd[0:10,0:10,3])
        print(tmpdepth[0:10,0:10])

        if i != num_of_patch-1:
            afmap[y-r:y+r,x-r:x+r] = 0
            afmax = np.nanmax(afmap)
            location = np.where(afmap == afmax)
            location_2d = np.transpose(np.array(location))[0]

        return localrgbd,localaff_rgb,localaff_gray
    
affordanceMap,location_2d,inputColor,inputDepth,afmap = visualize(resultsFile,inputColorImage,inputDepthImage,backgroundColorImage,backgroundDepthImage,cameraIntrinsicsFile)
local_patch((128,128),1,inputColor,inputDepth,location_2d,affordanceMap,afmap)