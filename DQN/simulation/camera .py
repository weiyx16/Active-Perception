import math
import numpy as np
import PIL.Image as Image
import array
"""
    -- Start connect to the VREP first using port 19999
"""
import time

class CAMERA():

    def __init__(self, clientID):
        """
            Initialize the Camera in simulation
        """
        # self.RAD2EDG = 180 / math.pi
        self.Save_IMG = True
        self.Save_PATH_COLOR = r'./color/'
        self.Save_PATH_DEPTH = r'./depth/'
        self.Dis_NEAR = 0.01
        self.Dis_FAR = 10
        self.INT16 = 65535
        self.Img_WIDTH = 512
        self.Img_HEIGHT = 424
        self.Camera_NAME = r'kinect'
        self.Camera_RGB_NAME = r'kinect_rgb'
        self.Camera_DEPTH_NAME = r'kinect_depth'
        self.clientID = clientID
        self._setup_sim_camera()

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
        _, cam_position = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        _, cam_orientation = vrep.simxGetObjectOrientation(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)

        self.cam_trans = np.eye(4,4)
        self.cam_trans[0:3,3] = np.asarray(cam_position)
        self.cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        self.cam_rotm = np.eye(4,4)
        self.cam_rotm[0:3,0:3] = np.linalg.inv(self._euler2rotm(cam_orientation))
        self.cam_pose = np.dot(self.cam_trans, self.cam_rotm) # Compute rigid transformation representating camera pose

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
        depth_img = (depth_img - self.Dis_NEAR)/self.Dis_FAR * self.INT16
        return depth_img, color_img

    def save_image(self, cur_depth, cur_color, currCmdTime, Save_IMG = True):
        """
            -- Save Color&Depth images
        """
        if Save_IMG:
            img = Image.fromarray(cur_color.astype('uint8')).convert('RGB')
            img_path = self.Save_PATH_COLOR + str(currCmdTime) + '_Rgb.png'
            img.save(img_path)
            depth_img = Image.fromarray(cur_depth.astype(np.uint32),mode='I')
            depth_path = self.Save_PATH_DEPTH + str(currCmdTime) + '_Depth.png'
            depth_img.save(depth_path)
            print("Succeed to save current images")
            return depth_path, img_path
        else:
            print("Skip saving images this time")

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