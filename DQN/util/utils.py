import PIL.Image as Image
import array
import h5py
import cv2 as cv
from scipy.ndimage.filters import uniform_filter
import pcl
import numpy as np
import time
import _pickle as cPickle
import copy
import math

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

def get_patch(location_2d, cur_color, cur_depth, post_afford, size=(128,128)):
    """
        input the postprocessed affordance map 
        return the 4*128*128 patch (cascade the depth and rgb)
    """
    # Normalization of input
    cur_color = cur_color / 255.
    cur_depth = cur_depth / 65535.
    y = location_2d[0]
    x = location_2d[1]
    r = int(size[0]/2)
    patch_color = cur_color[y-r:y+r,x-r:x+r,:]
    patch_depth = cur_depth[y-r:y+r,x-r:x+r]
    patch_afford = post_afford[y-r:y+r,x-r:x+r]
    patch_rgbd = np.zeros((size[0], size[1], 4))
    patch_rgbd[:,:,0:3] = patch_color
    patch_rgbd[:,:,3] = patch_depth
    # Resize the patch for a smaller action space
    patch_size = int(size[0]/4)
    patch_rgbd =  cv.resize(patch_rgbd, dsize=(patch_size, patch_size), interpolation=cv.INTER_CUBIC)

    return np.transpose(patch_rgbd, (2,0,1)), patch_afford, location_2d

def postproc_affimg(cur_color, cur_depth, cur_afford, bg_color, bg_depth, intri, border_pos):
    """
        # postprocess the affordance map
        # convert from the afford_model from matlab to python
    """
    cur_color = cur_color / 255.0
    cur_depth = cur_depth / 10000
    temp = (np.abs(cur_color -bg_color) < 0.3)
    foregroundMaskColor = np.sum(temp, axis = 2) != 3
    backgroundDepth_mask = np.zeros(bg_depth.shape, dtype = bool)
    backgroundDepth_mask[bg_depth!=0] = True
    foregroundMaskDepth = backgroundDepth_mask & (np.abs(cur_depth-bg_depth) > 0.005)
    foregroundMask = (foregroundMaskColor | foregroundMaskDepth)

    x = np.linspace(0,512-1,512)
    y = np.linspace(0,424-1,424)
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
    # accelarate the filter
    meanStdNormals = np.mean(tmp,axis = 2)
    normalBasedSuctionScores = 1 - meanStdNormals / np.max(meanStdNormals)
    cur_afford[np.where(normalBasedSuctionScores < 0.1)] = 0 
    cur_afford[np.where(foregroundMask == False)] = 0 
    post_afford = cv.GaussianBlur(cur_afford,(7,7),7)

    aff_center = post_afford[border_pos[0]:border_pos[1],border_pos[2]:border_pos[3]]
    aff_max = np.max(aff_center)
    # print(' -- Candidate Maximum location {}' .format(np.transpose(np.where(aff_center == aff_max))))
    location = np.transpose(np.where(aff_center == aff_max))[0] + \
        np.asarray([border_pos[0],border_pos[2]])
    tmp = aff_max+np.zeros(post_afford.shape)
    tmp[border_pos[0]:border_pos[1],border_pos[2]:border_pos[3]] = aff_center     
    post_afford = tmp
    return post_afford,location
    
def _window_stdev(X, window_size):
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

def _sub2ind(arraySize, dim0Sub, dim1Sub, dim2Sub):
    """
        sub 2 ind
    """
    return np.ravel_multi_index((dim0Sub,dim1Sub,dim2Sub), dims=arraySize, order='C')

def _surface_normals(cloud):
    """
        # calculate the surface normals from the cloud
    """
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.001)  #this parameter influence the speed
    cloud_normals = ne.compute()
    return cloud_normals

def peak_dis(ilabel,iaff):
    """
        Used in affordance metric
    """
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
    """
        Used in affordance metric
    """
    return np.exp((-(x-ex)**2)/(2*vx**2)+(-(y-ey)**2)/(2*vy**2))

def flatness(iaff,icenter,cnt):
    """
        Used in affordance metric
    """
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

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print(" [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed

def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print(" [*] save %s" % path)

@timeit
def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
        print(" [*] load %s" % path)
        return obj

@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print(" [*] save %s" % path)

@timeit
def load_npy(path):
    obj = np.load(path)
    print(" [*] load %s" % path)
    return obj