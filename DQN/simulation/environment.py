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
import pcl

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

# TODO: what's the range of color/depth for each network (like affordance map network/ post process/ DQN?)

class Camera(object):
    """
        # kinect camera in simulation
    """
    def __init__(self, clientID, Lua_PATH):
        """
            Initialize the Camera in simulation
        """
        self.RAD2EDG = 180 / math.pi
        self.EDG2RAD = math.pi / 180
        self.Save_IMG = True
        self.Save_PATH_COLOR = r'./color/'
        self.Save_PATH_DEPTH = r'./depth/'
        self.Save_PATH_RES = r'./afford_h5/'
        self.Lua_PATH = Lua_PATH
        self.Dis_FAR = 10 #TODO:
        self.INT16 = 65535
        self.Img_WIDTH = 512
        self.Img_HEIGHT = 424
        self.theta = 70
        self.Camera_NAME = r'kinect'
        self.Camera_RGB_NAME = r'kinect_rgb'
        self.Camera_DEPTH_NAME = r'kinect_depth'
        self.clientID = clientID
        self._setup_sim_camera()
        self.bg_color = np.empty((3, self.Img_HEIGHT, self.Img_WIDTH), dtype = np.float16)
        self.bg_depth = np.empty((1, self.Img_HEIGHT, self.Img_WIDTH), dtype = np.float16)

    def _euler2rotm(self,theta):
        """
            -- Get rotation matrix from euler angles
        """
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])         
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])            
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def _setup_sim_camera(self):
        """
            -- Get some param and handles from the simulation scene
            and set necessary parameter for camera
        """
        # Get handle to camera
        _, self.cam_handle = vrep.simxGetObjectHandle(self.clientID, self.Camera_NAME, vrep.simx_opmode_oneshot_wait)
        _, self.kinectRGB_handle = vrep.simxGetObjectHandle(self.clientID, self.Camera_RGB_NAME, vrep.simx_opmode_oneshot_wait)
        _, self.kinectDepth_handle = vrep.simxGetObjectHandle(self.clientID, self.Camera_DEPTH_NAME, vrep.simx_opmode_oneshot_wait)
        # Get camera pose and intrinsics in simulation
        _, self.cam_position = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        _, cam_orientation = vrep.simxGetObjectOrientation(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)

        self.cam_trans = np.eye(4,4)
        self.cam_trans[0:3,3] = np.asarray(self.cam_position)
        self.cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        self.cam_rotm = np.eye(4,4)
        self.cam_rotm[0:3,0:3] = np.linalg.inv(self._euler2rotm(cam_orientation))
        self.cam_pose = np.dot(self.cam_trans, self.cam_rotm) # Compute rigid transformation representating camera pose
        self._intri_camera()

    def _intri_camera(self):
        """
            Calculate the intrinstic parameters of camera
        """
        fx = -self.Img_WIDTH/(2.0*self.Dis_FAR*math.tan(self.theta * self.EDG2RAD))
        fy = fx
        u0 = self.Img_WIDTH / 2
        v0 = self.Img_HEIGHT / 2
        self.intri = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0, 0, 1]])
    
    def _bg_init(self):
        """
            -- use this function to save background RGB and Depth in the beginning
            -- it is used for the post process of affordance map
        """
        self.bg_depth, self.bg_color = self.get_camera_data()

    def get_camera_data(self):
        """
            -- Read images data from vrep and convert into np array
        """
        # Get color image from simulation
        res, resolution, raw_image = vrep.simxGetVisionSensorImage(self.clientID, self.kinectRGB_handle, 0, vrep.simx_opmode_oneshot_wait)
        self._error_catch(res)
        color_img = np.array(raw_image, dtype=np.uint8)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        res, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.kinectDepth_handle, vrep.simx_opmode_oneshot_wait)
        self._error_catch(res)
        depth_img = np.array(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        # zNear = 0.01
        # zFar = 10
        # depth_img = depth_img * (zFar - zNear) + zNear
        # TODO: ? What's the range?
        depth_img[depth_img < 0] = 0
        depth_img[depth_img > 1] = 1
        depth_img = depth_img * self.INT16
        return depth_img, color_img

    def save_image(self, cur_depth, cur_color, img_idx):
        """
            -- Save Color&Depth images
        """
        img = Image.fromarray(cur_color.astype('uint8')).convert('RGB')
        img_path = self.Save_PATH_COLOR + str(img_idx) + '_Rgb.png'
        img.save(img_path)
        depth_img = Image.fromarray(cur_depth.astype(np.uint32),mode='I')
        depth_path = self.Save_PATH_DEPTH + str(img_idx) + '_Depth.png'
        depth_img.save(depth_path)

        return depth_path, img_path

    def _error_catch(self, res):
        """
            -- Deal with error unexcepted
        """
        if res == vrep.simx_return_ok:
            print ("--- Image Exist!!!")
        elif res == vrep.simx_return_novalue_flag:
            print ("--- No image yet")
        else:
            print ("--- Error Raise")

    def hdf2affimg(self, filename):
        """
            Convert hdf5 file(output of affordance map network in lua) to img
            # Notice according to the output of the model we need to resize
        """
        h = h5py.File(filename,'r')

        res = h['results']
        res = np.array(res)
        res = res[0,1,:,:]
        resresize = 255.0 * cv.resize(res, (self.Img_WIDTH, self.Img_HEIGHT), interpolation=cv.INTER_CUBIC)
        resresize[np.where(resresize>1)] = 0.9999
        resresize[np.where(resresize<0)] = 0
        return resresize

    def local_patch(self, img_idx, patch_size = (128, 128)):
        """
            # according to the affordance map output
            # postprocess it and get the maximum local patch (4*128*128)
            # cascade the RGBD 4 channels and input to the agent
        """
        # first get the current img data first
        cur_depth, cur_color = self.get_camera_data()
        self.cur_depth = cur_depth
        cur_depth_path, cur_img_path = self.save_image(cur_depth, cur_color, img_idx)
        
        # feed this image into affordance map network and get the h5 file
        cur_res_path = self.Save_PATH_RES + str(img_idx) + '_results.h5'
        affordance_cmd = 'th ' + self.Lua_PATH + ' -imgColorPath ' + cur_img_path + \
                        ' -imgDepthPath ' + cur_depth_path + ' -resultPath ' + cur_res_path
        try:
            os.system(affordance_cmd)
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!  Error occurred during creating affordance map')
            exit()
        
        # get the initial affordance map from h5 file
        cur_afford = self.hdf2affimg(cur_res_path)

        # postprocess the affordance map
        post_afford = self._postproc_affimg(cur_color, cur_depth, cur_afford)
        location = np.where(post_afford == np.nanmax(post_afford))
        location_2d = np.transpose(np.array(location))[0]

        # according to the affordance map -> get the local patch with size of 4*128*128
        return self._get_patch(location_2d, cur_color, cur_depth, patch_size)

    def _get_patch(self, location_2d, cur_color, cur_depth, post_afford, size=(128,128)):
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

    def _postproc_affimg(self, cur_color, cur_depth, cur_afford):
        """
            # postprocess the affordance map
            # convert from the afford_model from matlab to python
        """
        cur_color = cur_color / 255.0
        rgb_sim = np.sum((np.abs(cur_color - self.bg_color) < 0.3), axis = 2)
        foregroundMaskColor = np.logical_not(rgb_sim == 3)

        backgroundDepth_mask = np.zeros(self.bg_depth.shape, dtype = bool)
        backgroundDepth_mask[self.bg_depth!=0] = True
        foregroundMaskDepth = backgroundDepth_mask & (np.abs(cur_depth-self.bg_depth) > 0.02)
        # TODO: what's the true depth range???????
        foregroundMask = (foregroundMaskColor | foregroundMaskDepth)

        x = np.linspace(0,self.Img_WIDTH-1,self.Img_WIDTH)
        y = np.linspace(0,self.Img_HEIGHT-1,self.Img_HEIGHT)
        pixX,pixY = np.meshgrid(x,y)
        camX = (pixX-self.intri[0,2])*cur_depth/self.intri[0,0]
        camY = (pixY-self.intri[1,2])*cur_depth/self.intri[1,1]
        camZ = cur_depth
        validDepth = foregroundMask & (camZ != 0) # only points with valid depth and within foreground mask
        inputPoints = [camX[validDepth],camY[validDepth],camZ[validDepth]]
        inputPoints = np.asarray(inputPoints,dtype=np.float32)
        foregroundPointcloud = pcl.PointCloud(np.transpose(inputPoints))
        foregroundNormals = self._surface_normals(foregroundPointcloud)

        tmp = np.zeros((foregroundNormals.size, 4), dtype=np.float16)
        for i in range(foregroundNormals.size):
            tmp[i] = foregroundNormals[i]
        foregroundNormals = np.nan_to_num(tmp[:,0:3])

        sensorCenter = np.array([0,0,0])
        for k in range(0,inputPoints.shape[1]):
            p1 = sensorCenter - np.array([inputPoints[0,k],inputPoints[1,k],inputPoints[2,k]])
            p2 = np.array([foregroundNormals[k,0],foregroundNormals[k,1],foregroundNormals[k,2]])
            angle = np.arctan2(np.linalg.norm(np.cross(p1,p2)),np.transpose(np.dot(p1,p2)))
            if angle > math.pi/2 or angle < -math.pi/2:
                pass
            else:
                foregroundNormals[k,:] = -foregroundNormals[k,:]

        pixX = np.rint(np.dot(inputPoints[0,:],self.intri[0,0])/inputPoints[2,:]+self.intri[0,2])
        pixY = np.rint(np.dot(inputPoints[1,:],self.intri[1,1])/inputPoints[2,:]+self.intri[1,2])

        surfaceNormalsMap = np.zeros(cur_color.shape)

        pixX = pixX.astype(np.int)
        pixY = pixY.astype(np.int)
        tmp = np.ones(pixY.shape)
        tmp = tmp.astype(np.int)

        surfaceNormalsMap.ravel()[self._sub2ind(surfaceNormalsMap.shape,pixY,pixX,tmp-1)] = foregroundNormals[:,0]
        surfaceNormalsMap.ravel()[self._sub2ind(surfaceNormalsMap.shape,pixY,pixX,2*tmp-1)] = foregroundNormals[:,1]
        surfaceNormalsMap.ravel()[self._sub2ind(surfaceNormalsMap.shape,pixY,pixX,3*tmp-1)] = foregroundNormals[:,2]
        
        # filter the affordance map
        meanStdNormals = np.mean(self._stdfilt(surfaceNormalsMap, np.ones((25,25))),axis = 2)
        normalBasedSuctionScores = np.ones(meanStdNormals.shape) - meanStdNormals / np.nanmax(meanStdNormals)
        cur_afford[np.where(normalBasedSuctionScores < 0.05)] = 0 
        cur_afford[np.where(foregroundMask == False)] = 0 
        post_afford = cv.GaussianBlur(cur_afford,(7,7),7)

        return post_afford

    def _stdfilt(self, src_img, filter):
        """
            std filt in np
            # TODO: convert to conv2d function for faster?
        """
        h,w,d = src_img.shape
        half_filter = int(filter.shape[0]/2)
        tmp = np.ones((h,half_filter,d))
        newmap = np.hstack((tmp,src_img,tmp))
        tmp = np.ones((half_filter,w+2*half_filter,d))
        newmap = np.vstack((tmp,newmap,tmp))
        ians = np.zeros(src_img.shape)
        for i in range(0, h):
            for j in range(0, w):
                tmpmap = newmap[i:i+2*half_filter+1,j:j+2*half_filter+1,:]
                ians[i,j,:] = np.std(tmpmap)
        return ians

    def _sub2ind(self, arraySize, dim0Sub, dim1Sub, dim2Sub):
        """
            sub 2 ind
        """
        return np.ravel_multi_index((dim0Sub,dim1Sub,dim2Sub), dims=arraySize, order='C')

    def _surface_normals(self, cloud):
        """
            # calculate the surface normals from the cloud
        """
        ne = cloud.make_NormalEstimation()
        tree = cloud.make_kdtree()
        ne.set_SearchMethod(tree)
        ne.set_RadiusSearch(0.001)  #this parameter influence the speed
        cloud_normals = ne.compute()
        return cloud_normals

    def pixel2ur5(self, u, v, ur5_position, push_depth):
        """
            from pixel u,v and correspondent depth z -> coor in ur5 coordinate (x,y,z)
        """
        depth = self.cur_depth[u][v] / self.INT16 * self.Dis_FAR
        x = depth*(u - self.intri[0][2]) / self.intri[0][0]
        y = depth*(v - self.intri[1][2]) / self.intri[1][1]
        camera_coor = np.array([x, y, depth - push_depth])

        """
            from camera coor to ur5 coor
            Notice the camera faces the plain directly and we needn't convert the depth to real z
        """
        camera_coor[2] = - camera_coor[2]
        location = camera_coor + self.cam_position - np.asarray(ur5_position)
        return location
    

class UR5(object):
    """
        # ur5 arm in simulation
        # including the initial of the scene and the initial of the simulation
    """
    def __init__(self):
        self.RAD2DEG = 180 / math.pi   # 常数，弧度转度数
        self.tstep = 0.005             # 定义仿真步长
        self.targetPosition=np.zeros(3,dtype=np.float)#目标位置
        self.targetQuaternion=np.array([0.707, 0, 0.707, 0])
        # 配置关节信息
        self.jointNum = 6
        self.baseName = 'UR5'         #机器人名字
        self.ikName = 'UR5_ikTarget'
        self.jointName = 'UR5_joint'
        self.jointHandle = np.zeros((self.jointNum,), dtype=np.int) # 各关节handle
        self.jointangel=[-111.5,-22.36,88.33,28.08,-90,-21.52]
        # 配置方块信息
        self.cubename= 'imported_part_'
        self.filename= 'test-10-obj-'
        self.scenepath = './scenes'
        self.cubenum = 11
        self.cubeHandle = np.zeros((self.cubenum,), dtype=np.int) # 各cubehandle
        self.obj_colors=[]
        self.obj_positions=[]
        self.obj_orientations=[]
        self.obj_order=[]

    def connect(self):  
        """
            # connect to v-rep
        """
        print('Simulation started') # 关闭潜在的连接 
        vrep.simxFinish(-1) # 每隔0.2s检测一次，直到连接上V-rep 
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
            if self.clientID > -1: 
                break 
            else: 
                time.sleep(0.2) 
                print("Failed connecting to remote API server!") 
        print("Connection success!")
        for i in range(self.cubenum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.cubename + str(i), vrep.simx_opmode_blocking) 
            self.cubeHandle[i] = returnHandle 
            print(returnHandle)
        vrep.simxSetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, self.tstep, vrep.simx_opmode_oneshot) # 保持API端与V-rep端相同步长
        vrep.simxSynchronous(self.clientID, True) # 然后打开同步模式 
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot) 

    def ankleinit(self):
        """
            # initial the ankle angle for ur5
        """
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.jointName + str(i+1), vrep.simx_opmode_blocking) 
            self.jointHandle[i] = returnHandle
        vrep.simxSynchronousTrigger(self.clientID) # 让仿真走一步 
        for i in range(self.jointNum):
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetJointTargetPosition(self.clientID, self.jointHandle[i],self.jointangel[i]/self.RAD2DEG, vrep.simx_opmode_oneshot)  #设置关节角
            vrep.simxPauseCommunication(self.clientID, False)
            vrep.simxSynchronousTrigger(self.clientID) # 进行下一步 
            vrep.simxGetPingTime(self.clientID) # 使得该仿真步走完
        _, self.position = vrep.simxGetObjectPosition(self.clientID, self.get_handle(), -1, vrep.simx_opmode_oneshot_wait)

    def cubeinit(self, filenum):
        """
            initial the scene (the arrangement of the blocks)
        """
        fileadd=os.path.join(self.scenepath, self.filename+str('%02d' % filenum)+'.txt')
        fs = open(fileadd, 'r')
        file_content = fs.readlines()
        for object_idx in range(self.cubenum):
            file_content_curr_object = file_content[object_idx].split()
            self.obj_order.append(file_content_curr_object[0])
            self.obj_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
            self.obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
            self.obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
        fs.close()
        for j in range(self.cubenum):
            i=int(self.obj_order[j])
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetObjectOrientation(self.clientID,self.cubeHandle[i],-1,self.obj_orientations[j],vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.clientID, False)
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetObjectPosition(self.clientID,self.cubeHandle[i],-1,self.obj_positions[j],vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.clientID, False)

    def disconnect(self):
        """
            # disconnect from v-rep
            # and stop simulation
        """
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot)
        vrep.simxFinish(self.clientID)
        print ('Simulation ended!')
    
    def get_clientID(self):
        return self.clientID

    def get_handle(self):
        _, self.ur5_handle = vrep.simxGetObjectHandle(self.clientID, self.baseName, vrep.simx_opmode_oneshot_wait)
        return self.ur5_handle

    def get_position(self):
        return self.position

    def ur5moveto(self, move_to_location):
        """
            Push the ur5 hand to the location of move_to_location
        """
        vrep.simxSynchronousTrigger(self.clientID) # 让仿真走一步 
        self.targetPosition = move_to_location
        vrep.simxPauseCommunication(self.clientID, True)    #开启仿真
        vrep.simxSetIntegerSignal(self.clientID, 'ICECUBE_0', 21, vrep.simx_opmode_oneshot)
        for i in range(3):
            vrep.simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+1),self.targetPosition[i],vrep.simx_opmode_oneshot)
        for i in range(4):
            vrep.simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+4),self.targetQuaternion[i], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)
        vrep.simxSynchronousTrigger(self.clientID) # 进行下一步 
        vrep.simxGetPingTime(self.clientID) # 使得该仿真步走完

    def grasp(self):
        vrep.simxSetIntegerSignal(self.clientID,'RG2CMD',1,vrep.simx_opmode_blocking)

    def lose(self):
        vrep.simxSetIntegerSignal(self.clientID,'RG2CMD',0,vrep.simx_opmode_blocking)

    def getcubepos(self):
        for j in range(self.cubenum):
            i=int(self.obj_order[j])
            _,self.obj_positions=vrep.simxGetObjectPosition(self.clientID,self.cubeHandle[i],-1,vrep.simx_opmode_blocking)
            _,self.obj_orientations=vrep.simxGetObjectOrientation(self.clientID,self.cubeHandle[i],-1,vrep.simx_opmode_blocking)
            print(self.obj_positions+self.obj_orientations)

class DQNEnvironment(object):
    """
        # environment for training of DQN
    """
    def __init__(self, config):
        self.Lua_PATH = config.Lua_PATH
        self.EDG2RAD = math.pi / 180
        # initial the ur5 arm in simulation
        self.ur5 = UR5()
        self.ur5.connect()
        self.ur5.ankleinit()
        # initial the camera in simulation
        self.clientID = self.ur5.get_clientID()
        self.camera = Camera(self.clientID, self.Lua_PATH)
        self.camera._bg_init()

        self.terminal = False
        self.reward = 0
        self.action = -1
        self.inChannel = config.inChannel
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width
        self.location_2d = [self.screen_height//2, self.screen_width//2]
        self.screen = np.empty((self.inChannel, self.screen_height, self.screen_width))
        self.index = 0
        self.save_size = 1000 # use this params is for over-storage of early img in disk -> which used for the affordance input

    def new_scene(self):
        """
            Random initial the scene
            # scene index is from 0~9
        """
        scene_num = random.randint(0, 10)
        self.ur5.cubeinit(scene_num)
        # return the camera_data
        self.index = (self.index + 1)% self.save_size
        # location_2d stores the maximum affordance value coor know in the scene
        self.screen, self.local_afford_past, self.location_2d = self.camera.local_patch(self.index, (self.screen_height, self.screen_width))
        self.local_afford_new = self.local_afford_past
        self.terminal = self.ifterminal()
        return self.screen, 0, -1, self.terminal

    def close(self):
        """
            End the simulation
        """
        self.ur5.disconnect()

    def act(self, action, is_training=True):
        """
            first convert the action to (x,y,depth)
            then convert x,y,depth to pixel coor(according to the location_2d)
            then convert to coor in ur5
            then act it
            then take camera data and apply the local_patch again -> get new local_afford
            use this two affordance map -> reward
            use new affordance map -> terminal
        """
        # act on the scene
        move_to_location = self.action2ur5(action)
        self.ur5.ur5moveto(move_to_location)

        # get the new camera_data
        self.index = (self.index + 1)% self.save_size
        # location_2d stores the maximum affordance value coor know in the scene
        self.local_afford_past = self.local_afford_new
        self.screen, self.local_afford_new, self.location_2d = self.camera.local_patch(self.index, (self.screen_height, self.screen_width))
        self.reward = self.calcreward()
        self.terminal = self.ifterminal()
        return self.screen, self.reward, self.terminal

    def calcreward(self):
        """
            Use two affordance map to calculate the reward
        """
        pass

    def ifterminal(self):
        """
            Use the self.local_afford_new to judge if terminal
        """
        pass

    def action2ur5(self, action):
        """
            first convert the action to (x,y,depth)
            then convert x,y,depth to pixel coor(according to the location_2d)
            then convert to coor in ur5
        """
        # 96, 96 is the output of the u-net
        idx = np.unravel_index(action, (96, 96, 16))
        relate_local = list(idx[0:2])
        ori_depth_idx = np.unravel_index(int(idx[2]), (8,2))
        ori = ori_depth_idx[0] * 360. / 8.
        push_depth = ori_depth_idx[1] * 0.02 # choose in two depth 0 or 0.02 (deeper than the pixel depth)
        push_dis = self.screen_height // 4 # fix the push distance in 32 pixel 

        # seems the output of the u-net is the same size of input so we need to resize the output idx
        relate_local = (np.asarray(relate_local) + 1.0) * self.screen_height / 96 - 1.0
        relate_local = np.round(relate_local)
        real_local = relate_local
        real_local[0] = self.location_2d[0] + relate_local[0] - self.screen_height // 2
        real_local[1] = self.location_2d[1] + relate_local[1] - self.screen_width // 2

        real_dest = real_local
        real_dest[0] = real_local[0] + push_dis * math.cos(ori*self.EDG2RAD)
        real_dest[1] = real_local[1] + push_dis * math.sin(ori*self.EDG2RAD)
        real_dest = np.round(real_dest)

        # from pixel coor to real ur5 coor
        return self.camera.pixel2ur5(real_dest[0], real_dest[1], self.ur5.get_position, push_depth)