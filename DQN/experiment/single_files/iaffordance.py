import numpy as np
import h5py
import cv2 as cv
from scipy.ndimage.filters import uniform_filter
import pcl

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
    cur_afford[np.where(normalBasedSuctionScores < 0.1)] = 0 
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
