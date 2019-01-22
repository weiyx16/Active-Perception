# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.

"""
    -- Some const params
"""
import math
import numpy as np
import PIL.Image as Image
import array
Clinet_PORT = 19999
RAD2EDG = 180 / math.pi
Get_RGB = 0
Save_IMG = True
Save_PATH_COLOR = r'./color/'
Save_PATH_DEPTH = r'./depth/'
Simu_STEP = 10
Dis_NEAR = 0.01
Dis_FAR = 10
INT16 = 65535
Img_WIDTH = 512
Img_HEIGHT = 424
Camera_NAME = r'kinect'
Camera_RGB_NAME = r'kinect_rgb'
Camera_DEPTH_NAME = r'kinect_depth'

def setup_sim_camera():
    """
        -- Get some param and handles from the simulation scene
        and set necessary parameter for camera
    """
    # Get handle to camera
    _, cam_handle = vrep.simxGetObjectHandle(clientID, Camera_NAME, vrep.simx_opmode_oneshot_wait)
    _, kinectRGB_handle = vrep.simxGetObjectHandle(clientID, Camera_RGB_NAME, vrep.simx_opmode_oneshot_wait)
    _, kinectDepth_handle = vrep.simxGetObjectHandle(clientID, Camera_DEPTH_NAME, vrep.simx_opmode_oneshot_wait)
    # Get camera pose and intrinsics in simulation
    _, cam_position = vrep.simxGetObjectPosition(clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)
    _, cam_orientation = vrep.simxGetObjectOrientation(clientID, cam_handle, -1, vrep.simx_opmode_oneshot_wait)

    # TODO: Further saved in the camera class`
    cam_trans = np.eye(4,4)
    cam_trans[0:3,3] = np.asarray(cam_position)
    cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
    cam_rotm = np.eye(4,4)
    cam_rotm[0:3,0:3] = np.linalg.inv(euler2rotm(cam_orientation))
    cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose

    return kinectRGB_handle, kinectDepth_handle


def euler2rotm(theta):
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

def get_camera_data(kinectRGB_handle, kinectDepth_handle):
    """
        -- Read images data from vrep and convert into np array
    """
    # Get color image from simulation
    res, resolution, raw_image = vrep.simxGetVisionSensorImage(clientID, kinectRGB_handle, Get_RGB, vrep.simx_opmode_oneshot_wait)
    error_catch(res)
    color_img = np.array(raw_image, dtype=np.uint8)
    color_img.shape = (resolution[1], resolution[0], 3)
    color_img = color_img.astype(np.float)/255
    color_img[color_img < 0] += 1
    color_img *= 255
    color_img = np.fliplr(color_img)
    color_img = color_img.astype(np.uint8)

    # Get depth image from simulation
    res, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(clientID, kinectDepth_handle, vrep.simx_opmode_oneshot_wait)
    error_catch(res)
    depth_img = np.array(depth_buffer)
    depth_img.shape = (resolution[1], resolution[0])
    depth_img = np.fliplr(depth_img)
    # zNear = 0.01
    # zFar = 10
    # depth_img = depth_img * (zFar - zNear) + zNear
    depth_img = (depth_img - Dis_NEAR)/Dis_FAR * INT16
    return depth_img, color_img

def save_image(cur_depth, cur_color, currCmdTime, Save_IMG = True):
    """
        -- Save Color&Depth images
    """
    if Save_IMG:
        img = Image.fromarray(cur_color.astype('uint8')).convert('RGB')
        img_path = Save_PATH_COLOR + str(currCmdTime) + '_Rgb.png'
        img.save(img_path)
        depth_img = Image.fromarray(cur_depth.astype(np.uint32),mode='I')
        depth_path = Save_PATH_DEPTH + str(currCmdTime) + '_Depth.png'
        depth_img.save(depth_path)
        print("Succeed to save current images")
    else:
        print("Skip saving images this time")

def error_catch(res):
    """
        -- Deal with error unexcepted
    """
    if res == vrep.simx_return_ok:
        print ("--- Image Exist!!!")
    elif res == vrep.simx_return_novalue_flag:
        print ("--- No image yet")
    else:
        print ("--- Error Raise")

if __name__ == "__main__":
    """
        -- Start connect to the VREP first using port 19999
    """
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

    import time

    print ('Try to connect to the VREP')
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID = vrep.simxStart('127.0.0.1',Clinet_PORT, True, True, 5000,5) # Connect to V-REP
    if clientID != -1:
        print ('Connected to remote API server')
        vrep.simxAddStatusbarMessage(clientID,'Hello V-REP!',vrep.simx_opmode_oneshot)
    else:
        print ('Failed connecting to remote API server')
    print ('Program started')

    """
        -- Basic simulation setting
    """
    tstep = 5
    vrep.simxSetFloatingParameter(clientID,vrep.sim_floatparam_simulation_time_step,tstep,vrep.simx_opmode_oneshot)
    vrep.simxSynchronous(clientID, True)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

    """
        -- Start simulation
    """
    lastCmdTime = vrep.simxGetLastCmdTime(clientID)  # 记录当前时间
    vrep.simxSynchronousTrigger(clientID)  # 让仿真走一步
    count = 0
    while vrep.simxGetConnectionId(clientID) != -1 and count < Simu_STEP:
        currCmdTime=vrep.simxGetLastCmdTime(clientID)
        dt = currCmdTime - lastCmdTime # 记录时间间隔

        kinectRGB_handle, kinectDepth_handle = setup_sim_camera()
        cur_depth, cur_color = get_camera_data(kinectRGB_handle, kinectDepth_handle)
        save_image(cur_depth, cur_color, currCmdTime, Save_IMG)
        count += 1
        lastCmdTime=currCmdTime
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)    # 使得该仿真步走完

    """
        -- Finish the simulation
    """
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
    vrep.simxFinish(clientID)
    print ('Program ended!')