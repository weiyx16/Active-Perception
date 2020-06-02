import sys
import os
import time
import numpy.random as random
import numpy as np
import math
import PIL.Image as Image
import array
import threading
import urx
import serial
from skimage import measure
import copy
import cv2 as cv
from util.utils import *
"""
    # connect to the real kinect
"""

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel
# TODO: Need to update to master branch

try: 
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

"""
    Set Kinect:
"""
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print(" [!] No kinect device connected!")
    sys.exit(1)

serial_kinect = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial_kinect, pipeline=pipeline)

# listener = SyncMultiFrameListener( FrameType.Color | FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener( FrameType.Color | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

device.start()

# NOTE: must be called after device.start()
registration = Registration(device.getIrCameraParams(),
                            device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

class Camera(object):
    """
        # kinect camera in simulation
    """
    def __init__(self, Lua_PATH):
        """
            Initialize the Camera in simulation
        """
        self.RAD2EDG = 180 / math.pi
        self.EDG2RAD = math.pi / 180
        self.Save_PATH = r'./experiment/img'
        self.Save_init_PATH = r'./experiment/scene'
        self.intri = np.array([[363.253, 0, 254.157], 
                                [0, 363.304, 206.881 ],
                                [0, 0, 1 ]])
        self.Lua_PATH = Lua_PATH
        self.depth_scale = 1000
        self.Img_WIDTH = 512
        self.Img_HEIGHT = 424
        self.border_pos = [80,320,95,370] #[0, self.Img_HEIGHT - 1, 0, self.Img_WIDTH - 1] # TODO:
        self.bg_color = np.empty((self.Img_HEIGHT, self.Img_WIDTH ,3))
        self.bg_depth = np.empty((self.Img_HEIGHT, self.Img_WIDTH, 1))

        self._mkdir_save(self.Save_PATH)
        self._mkdir_save(self.Save_init_PATH)
        self.tran =  [[-0.999746, -0.0196759 ,-0.0109776, -0.0182557],
                        [-0.0199747,   0.999414 , 0.0278074,   -0.52257],
                        [  0.010424 , 0.0280196,  -0.999553 ,  0.967819], # TODO: need 0.025?
                        [        0  ,        0,          0   ,       1]]
        self.rotate_tran = [[-0.999746, -0.0196759 ,-0.0109776],
                            [-0.0199747,   0.999414 , 0.0278074],
                            [  0.010424 , 0.0280196,  -0.999553]] 
        # self._set_real_camera()
        #self._bg_init()

    def _mkdir_save(self, path_name):
        if not os.path.isdir(path_name):         
            os.mkdir(path_name)

    def _set_real_camera(self):
        pass

    def _bg_init(self):
        '''
            Init background rgb and depth data
        '''
        #input(' >> Initial the background now \n >> If you have done, use "Enter" ')
        bg_depth_path, bg_color_path = self.get_camera_data(is_bg=True)
        self.bg_depth = cv.imread('./experiment/img/Bg_Depth.png', -1)/10000.0
        self.bg_color = cv.imread('./experiment/img/Bg_Rgb.png')/255.0
        #exit()

    def get_camera_data(self, is_bg=False, is_init=False):
        '''
            return kinect data and save it in the same time
        '''
        
        frames = listener.waitForNewFrame()

        color = frames["color"]
        depth_now = frames["depth"]

        registration.apply(color, depth_now, undistorted, registered,
                        bigdepth=None,
                        color_depth_map=None)

        listener.release(frames)
        registered_lr = np.fliplr(registered.asarray(np.uint8))
        registered_lr = registered_lr[:,:,0:3] # Remove the alpha channel

        depth_now = np.fliplr(depth_now.asarray(np.float32))
        depth_copy = copy.deepcopy(depth_now)

        # depth_now image always crushed with ir
        while np.max(depth_copy) > 2000:
            time.sleep(0.05) 
            frames = listener.waitForNewFrame()

            color = frames["color"]
            depth_now = frames["depth"]

            registration.apply(color, depth_now, undistorted, registered,
                            bigdepth=None,
                            color_depth_map=None)  
            listener.release(frames)
            depth_now = np.fliplr(depth_now.asarray(np.float32))
            depth_copy = copy.deepcopy(depth_now)

        device.stop()
        device.close()

        """
            -- Save Color&Depth images
        """
        
        #img = Image.fromarray(registered_lr.astype(np.uint8)).convert('RGB')
        if is_bg:
            img_path = os.path.join(self.Save_PATH, 'Bg_Rgb.png')
            cv.imwrite(img_path,registered_lr)

        elif is_init:
            index = len([lists for lists in os.listdir(self.Save_init_PATH) if os.path.isfile(os.path.join(self.Save_init_PATH, lists))])
            img_path = os.path.join(self.Save_init_PATH, str(index) + '_init_color.png')
            cv.imwrite(img_path,registered_lr)
            img_path = os.path.join(self.Save_PATH, 'Cur_Rgb.png')
            cv.imwrite(img_path,registered_lr)
        else:
            img_path = os.path.join(self.Save_PATH, 'Cur_Rgb.png')
            cv.imwrite(img_path,registered_lr)
        
        depth_img = Image.fromarray(depth_copy.astype(np.uint32),mode='I')
        if is_bg:
            depth_path = os.path.join(self.Save_PATH, 'Bg_Depth.png')
        else:
            depth_path = os.path.join(self.Save_PATH, 'Cur_Depth.png')
        depth_img.save(depth_path)
        '''
        input(' >>  wait for camera!')
        if is_bg:
            img_path = os.path.join(self.Save_PATH, 'Bg_Rgb.png')

        else:
            img_path = os.path.join(self.Save_PATH, 'Cur_Rgb.png')

        if is_bg:
            depth_path = os.path.join(self.Save_PATH, 'Bg_Depth.png')
        else:
            depth_path = os.path.join(self.Save_PATH, 'Cur_Depth.png')
        '''
        return depth_path, img_path

    def local_patch(self, patch_size = (128, 128), is_init = False):
        """
            # according to the affordance map output
            # postprocess it and get the maximum local patch (4*128*128)
            # cascade the RGBD 4 channels and input to the agent
        """
        # first get the current img data first
        cur_depth_path, cur_img_path = self.get_camera_data(is_init = is_init)
        self.cur_depth_frame = cv.imread(cur_depth_path, -1)
        self.cur_color_frame = cv.imread(cur_img_path)

        # feed this image into affordance map network and get the h5 file
        cur_res_path = os.path.join(self.Save_PATH, 'Cur_results.h5')
        affordance_cmd = 'th ' + self.Lua_PATH + ' -imgColorPath ' + cur_img_path + \
                        ' -imgDepthPath ' + cur_depth_path + ' -resultPath ' + cur_res_path
        try:
            # subprocess.call(affordance_cmd)
            os.system(affordance_cmd)
        except:
            raise Exception(' [!] !!!!!!!!!!!  Error occurred during calling affordance map')
        # get the initial affordance map from h5 file
        if os.path.isfile(cur_res_path):
            cur_afford = hdf2affimg(cur_res_path)
        else:
            # Use this to catch the exception for torch itself
            raise Exception(' [!] !!!!!!!!!!!  Error occurred during creating affordance map')          

        # postprocess the affordance map
        post_afford, location_2d, self.surfaceNormalsMap = postproc_affimg( self.cur_color_frame,  self.cur_depth_frame, cur_afford, 
                                                    self.bg_color, self.bg_depth, self.intri, self.border_pos)

        print('\n -- Maximum Location at {}' .format(location_2d))
        # according to the affordance map -> get the local patch with size of 4*128*128
        return get_patch(location_2d,  self.cur_color_frame,  self.cur_depth_frame, post_afford, patch_size)

    def pixel2ur5(self, u, v, push_depth, depth_point = 0.0, is_dst = True):
        """
            from pixel u,v and correspondent depth_point z -> coor in ur5 coordinate (x,y,z)
        """
        if is_dst == False:
            depth_point =  self.cur_depth_frame[int(v)][int(u)] / self.depth_scale
        x = depth_point * (u - self.intri[0][2]) / self.intri[0][0]
        y = depth_point * (v - self.intri[1][2]) / self.intri[1][1]
        location_3d = [x, y, depth_point]
        
        normal_max_point = self.surfaceNormalsMap[int(v)][int(u)]
        # Convert to camera coor
        normal_max_point = normal_max_point * np.asarray([-1,1,-1])
        # Convert to ur5 coor
        normal_max_point = np.dot(np.linalg.inv(self.rotate_tran), [[normal_max_point[0]],[normal_max_point[1]],[normal_max_point[2]]])

        while(depth_point<0.2 or depth_point > 0.95):
            v += 1
            u += 1
            depth_point =  self.cur_depth_frame[int(v)][int(u)] / self.depth_scale
            x = depth_point * (u - self.intri[0][2]) / self.intri[0][0]
            y = depth_point * (v - self.intri[1][2]) / self.intri[1][1]
            location_3d = [x, y, depth_point]
        print('Coor in eye is '+str(location_3d))
        location_3d_homo = [[location_3d[0]], [location_3d[1]], [location_3d[2]],[1]]
        return np.dot(np.linalg.inv(self.tran), location_3d_homo), depth_point, normal_max_point

class Hand():
    def __init__(self, rec_data):
        self.rec_data = rec_data  # 初始化线程
        self.opening_Hand()

    def grasp(self):
        self.rec_data.send_data('g')

    def place(self):
        self.rec_data.send_data('p')

    def suck(self):
        self.rec_data.send_data('s')

    def loose(self):
        self.rec_data.send_data('l')

    def handin(self):
        self.rec_data.send_data('h')

    def handopen(self):
        self.rec_data.send_data('q')
    
    def cupout(self):
        self.rec_data.send_data('o')

    def cupreturn(self):
        self.rec_data.send_data('8')

    def opening_Hand(self):
        # 搜索匹配字符 ‘/dev/ttyACM0’的设备  connect to arduino
        port = '/dev/rfcomm1'
        baud = 9600
        openflag = self.rec_data.open_com(port, baud)  # 打开串口

class SerialData(threading.Thread):  # 创建threading.Thread的子类SerialData
    def __init__(self):
        threading.Thread.__init__(self)     # 初始化线程

    def open_com(self, port, baud):         # 打开串口
        self.ser = serial.Serial(port, baud, timeout=0.5)
        return self.ser

    def com_isopen(self):  # 判断串口是否打开
        return self.ser.isOpen()

    def send_data(self, data):  # 发送数据
        self.ser.write(data.encode())

    def next(self):  # 接收的数据组
        all_data = ''
        all_data = self.ser.readline().decode()  # 读一行数据
        trash = self.ser.readline().decode()
        trash = self.ser.readline().decode()
        print(all_data)
        print('into next')
        return all_data[0]

    def close_listen_com(self):  # 关闭串口
        return self.ser.close()

class UR5(object):
    """
        # ur5 arm in true experiment with command to control our hand with suckion and fingers
        # Include scene update and sucking the object
    """
    def __init__(self):
        self.robot = urx.Robot("192.168.1.111") 
        print(' [*] robot connect successfully')
        self.robot_vel = 0.3    #accerlation
        self.robot_acc = 0.2 
        self.robot_vel_safe = 0.1
        self.robot_acc_safe = 0.05
        self.hand_height = 0. # - 0.02 # mile set at the ur5
        self.safe_height = 0.18 # set at the ur5 350mm
        self.ver_angle = -(math.pi-0.001)
        #self.pos_ori = [-0.08,0.21,0.45,0,self.ver_angle,0]
        self.pos_ori = [-0.102,0.20,0.30,0,self.ver_angle,0] # lk fix
        self.pos_loose = [-0.520, 0.142, 0.30,0, self.ver_angle,0]
        rec_data = SerialData()
        self.hand = Hand(rec_data)
        # return to the initial pose
        #self.robot.movel(self.pos_ori, acc = self.robot_acc,vel = self.robot_vel) 

    def vectomatirx(self,vecz):
        vecz_start=np.array([0,0,1])
        vecx=np.cross(vecz_start,vecz)
        if np.linalg.norm(vecx)==0:
            vecx=np.array([1,0,0])
        vecy=np.cross(vecz, vecx)
        vecx=vecx/np.linalg.norm(vecx)
        vecy=vecy/np.linalg.norm(vecy)
        vecz=vecz/np.linalg.norm(vecz)
        matrix=np.array([vecx,vecy,vecz])
        matrix=np.transpose(matrix)

        theta = math.acos((matrix[0,0]+matrix[1,1]+matrix[2,2]-1)/2)
        tmp = np.array([matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]])
        rxyz = tmp/(2*math.sin(theta)) * theta
        return matrix,rxyz

    def ur5push(self, move_begin, move_to):
        """
            The action of the ur5 in a single act including:
            Get to push beginning
            Push to the destination
            Return to the init pose
        """
        # push act
        pos_in = [move_begin[0],move_begin[1],self.safe_height,0, self.ver_angle ,0] # make the hand vertical down and adjust z of xyz
        print("\n Push to " + str(pos_in))
        cmd = input(' >> Continue ')
        if cmd == 'q':
            return
        else:
            pass
        self.robot.movel(pos_in, acc = 2*self.robot_acc_safe,vel = 3*self.robot_vel_safe)  
        # self.hand.handin()
        self.hand.cupout()
        pos_in = [move_begin[0],move_begin[1],max([move_to[2]+self.hand_height,0.09]),0, self.ver_angle,0] # make the hand vertical down and adjust z of xyz
        print("\n Push to " + str(pos_in))
        input(' >> Continue ') 
        self.robot.movel(pos_in, acc = self.robot_acc_safe,vel = self.robot_vel_safe)
        pos_in = [move_to[0],move_to[1],max([move_to[2]+self.hand_height,0.09]),0, self.ver_angle,0] # make the hand vertical down and adjust z of xyz
        print("\n Push to " + str(pos_in))
        input(' >> Continue ')
        self.robot.movel(pos_in, acc = self.robot_acc_safe,vel = self.robot_vel_safe)       
        pos_in = [move_to[0],move_to[1],self.safe_height,0, self.ver_angle,0] # make the hand vertical down and adjust z of xyz
        print("\n Push to " + str(pos_in))
        input(' >> Continue ') 
        self.robot.movel(pos_in, acc = 2*self.robot_acc_safe,vel = 3*self.robot_vel_safe)
        input(' >> Continue ')
        # self.hand.handopen()
        self.hand.cupreturn()
        # return to the initial pose
        self.robot.movel(self.pos_ori, acc = 2*self.robot_acc_safe,vel = 3*self.robot_vel_safe)
    def ur5suck(self, suck_location, normal_3d):
        """
            Suck in the fixed location
        """
        print('==========================================')
        vz = np.transpose(normal_3d)[0]
        vz[2] = vz[2]
        vz[0] = 0.8*vz[0]
        vz[1] = - 0.8*vz[1]
        print(vz)
        print('\n')
        matrix,rxyz = self.vectomatirx(vz)
        theta = math.acos( np.dot(vz,np.transpose(np.array([0,0,-1]))) /np.linalg.norm(vz) )
        print(theta)
        cmd = input(' >> Continue ')
        while theta > 0.3:
            vz[0] = vz[0] / 2
            vz[1] = vz[1] / 2
            theta = math.acos( np.dot(vz,np.transpose(np.array([0,0,-1]))) /np.linalg.norm(vz) )
            print('new theta:'+str(theta))
            if abs(theta - math.pi) < 0.1:
                theta = 0
                vz = [0.01,0.01,-1.01]
        if cmd == 'q':
            return
        else:
            pass
        matrix,rxyz = self.vectomatirx(vz)
        theta = math.acos( np.dot(vz,np.transpose(np.array([0,0,-1]))) /np.linalg.norm(vz) )

        #print('the ur5 pos location :'+str(vz)))
        pos_in = [self.pos_ori[0],self.pos_ori[1],self.pos_ori[2],rxyz[0],rxyz[1],rxyz[2]] # make the hand vertical down and adjust z of xyz
        input(' >> Continue ')
        self.robot.movel(pos_in, acc = self.robot_acc,vel = self.robot_vel)

        pos_in = [suck_location[0],suck_location[1],self.safe_height,rxyz[0],rxyz[1],rxyz[2]] # make the hand vertical down and adjust z of xyz
        input(' >> Continue ')
        self.robot.movel(pos_in, acc = self.robot_acc,vel = self.robot_vel)  
        input(' >> Continue ')
        pos_in = [suck_location[0],suck_location[1],suck_location[2],rxyz[0],rxyz[1],rxyz[2]] # make the hand vertical down and adjust z of xyz
        self.robot.movel(pos_in, acc = self.robot_acc,vel = self.robot_vel)        
        input(' >> Continue ') 
        self.hand.suck()
        time.sleep(1)
        pos_in = [suck_location[0],suck_location[1],self.safe_height,rxyz[0],rxyz[1],rxyz[2]] # make the hand vertical down and adjust z of xyz
        self.robot.movel(pos_in, acc = self.robot_acc,vel = self.robot_vel)
        # return to the initial pose
        input(' >> Continue ') 
        self.robot.movel(self.pos_ori, acc = self.robot_acc_safe,vel = self.robot_vel_safe)
        #input(' >> Continue ')
        #self.robot.movel(self.pos_loose, acc = 2*self.robot_acc, vel = 3*self.robot_vel)
        self.hand.loose()
        #time.sleep(1)
        input(' >> Continue ')
        #self.robot.movel(self.pos_ori, acc = 2*self.robot_acc,vel = 3*self.robot_vel)

class REALEnvironment(object):
    """
        # environment for testing in true world of DQN
    """
    def __init__(self, config):
        self.Lua_PATH = config.Lua_PATH
        self.End_Metric = config.end_metric
        self.terminal = False
        self.reward = 0
        self.action = -1
        self.inChannel = config.inChannel
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width
        self.Img_WIDTH = 512
        self.Img_HEIGHT = 424
        self.location_2d = [self.Img_HEIGHT//2, self.Img_WIDTH//2]
        self.screen = np.empty((self.inChannel, self.screen_height//4, self.screen_width//4))
        self.metric = 0
        self.EDG2RAD = math.pi / 180

        self.camera = Camera(self.Lua_PATH)
        self.ur5 = UR5() 
        print('\n [*] Initialize the true environment')

    def new_scene(self):
        """
            Random initial the scene manually
        """
        input(' >> Please set the scene manually \n >> Finish with input "Enter" ')

        # return the camera_data
        # location_2d stores the maximum affordance value coor know in the scene
        self.screen, self.local_afford_past, self.location_2d = self.camera.local_patch((self.screen_height, self.screen_width), is_init = True)
        self.local_afford_new = self.local_afford_past
        self.metric = self.reward_metric(self.local_afford_new)
        self.terminal = self.ifterminal()
        return self.screen, 0., -1, self.terminal
        
    def exp_act(self, action):
        """
            first convert the action to (x,y,depth)
            then convert x,y,depth to pixel coor(according to the location_2d)
            then convert to coor in ur5
            then push it
            then take camera data and apply the local_patch again -> get new local_afford
            use new affordance map -> terminal
        """
        # act on the scene
        move_begin, move_to = self.action2ur5(action)
        # print(' -- Push from {} to {}' .format(move_begin, move_to))
        self.ur5.ur5push(move_begin, move_to)
        # get the new camera_data
        # location_2d stores the maximum affordance value coor know in the scene
        self.local_afford_past = self.local_afford_new
        self.screen, self.local_afford_new, self.location_2d = self.camera.local_patch((self.screen_height, self.screen_width))
        self.metric = self.reward_metric(self.local_afford_new)
        self.terminal = self.ifterminal()
        return self.screen, 0, self.terminal

    def ifterminal(self):
        """
            Use the self.local_afford_new to judge if terminal
        """
        return self.metric > self.End_Metric

    def reward_metric(self, afford_map):
        """
            -- calculate the metric on a single affordance map
            -- for now we weight three values() 
            1. distance between two peaks in the local patch
            2. flatten level of the center peaks
            3. the value of the center peaks
        """

        limit = 0.4 #32.5*255/100

        iaff_gray_ori = afford_map # 0~1 aff

        iaff_gray_bm = copy.deepcopy(iaff_gray_ori)
        iaff_gray_bm[np.where(afford_map <limit)] = 0
        iaff_gray_bm[np.where(afford_map >limit)] = 1

        ilabel = measure.label(iaff_gray_bm, connectivity = 1) #8-connect,original file use aff(255), but here using aff(1.0)
        # ilabel = np.ones(iaff_gray_bm.shape())
        icenter = copy.deepcopy(iaff_gray_bm) 
        icenter[np.where(ilabel == ilabel[64,64])] = 1
        icenter[np.where(ilabel != ilabel[64,64])] = 0

        point_arr = np.transpose(np.where(icenter == 1))
        cnt = cv.minAreaRect(point_arr)

        ipeak_dis = peak_dis(ilabel, iaff_gray_ori)
        iflatness = flatness(iaff_gray_ori, icenter, cnt)
        icenter_value = np.max(afford_map)

        if ipeak_dis == 1:
            reward_metric = 0.9 * iflatness + 0.1 * icenter_value
        else:
            reward_metric = 0.75 * iflatness + 0.15 * ipeak_dis + 0.1 * icenter_value

        print(' -- Metric for current frame: %f \n   With peak_dis: %f, flatten: %f, max_value: %f' 
                %(reward_metric, ipeak_dis, iflatness, icenter_value))
        return reward_metric

    def action2ur5(self, action):
        """
            first convert the action to (x,y,depth)
            then convert x,y,depth to pixel coor(according to the location_2d)
            then convert to coor in ur5

            Including the beginning point and destination point
        """
        # 18, 18, 8 is the output of the u-net
        idx = np.unravel_index(action, (18, 18, 8))
        relate_local = list(idx[0:2])
        ori = idx[2] * 360. / 8.
        push_depth = 0 # - 0.03
        push_dis = self.screen_height / 2 # Use big distance to encourage robot to change the scene
        # seems the output of the u-net is the same size of input so we need to resize the output idx
        relate_local = (np.asarray(relate_local) + 1.0) * self.screen_height / 18 - 1.0
        relate_local = np.round(relate_local)
        real_local = self.location_2d + relate_local - self.screen_height // 2
        # -> to the new push point with dest ori and depth
        real_dest = []
        real_dest.append (real_local[0] + push_dis * math.cos(ori*self.EDG2RAD))
        real_dest.append (real_local[1] + push_dis * math.sin(ori*self.EDG2RAD))
        print('\n -- Push from {} to {}' .format(real_local,real_dest))
        # from pixel coor to real ur5 coor
        move_begin, src_depth,_ = self.camera.pixel2ur5(min(real_local[1], 511), min(real_local[0],423), push_depth, is_dst = False)
        move_to, _, _ = self.camera.pixel2ur5(min(real_dest[1],511), min(real_dest[0],423),  push_depth, src_depth, is_dst = True)
        return move_begin, move_to
    
    def ope(self):
        """
            # Final suck operation if terminal (metric is up to desired value)
        """
        # suck in self.location_2d
        location_3d, _, normal_3d = self.camera.pixel2ur5(self.location_2d[1], self.location_2d[0], 0., is_dst = False)
        print('Coor in ur5 ' + str(location_3d))
        # push act
        self.ur5.ur5suck(location_3d, normal_3d)

    def observe_screen(self):
        self.screen, self.local_afford_new, self.location_2d = self.camera.local_patch((self.screen_height, self.screen_width))
        self.metric = self.reward_metric(self.local_afford_new)
        self.terminal = self.ifterminal()
        return self.screen, self.terminal
    
    def video(self):
        self.ur5.video_used()