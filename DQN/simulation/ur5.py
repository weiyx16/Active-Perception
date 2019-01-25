import numpy as np
import vrep
import os
import math
import time

class UR5(object):

    def __init__(self,cubenum,filenum):
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
        self.RAD2DEG = 180 / math.pi   # 常数，弧度转度数
        self.tstep = 0.005             # 定义仿真步长
        self.targetPosition=np.zeros(3,dtype=np.float)#目标位置
        self.targetQuaternion=np.zeros(4)
        # 配置关节信息
        self.jointNum = 6
        self.baseName = 'UR5'         #机器人名字
        self.ikName = 'UR5_ikTarget'
        self.jointName = 'UR5_joint'
        self.jointHandle = np.zeros((self.jointNum,), dtype=np.int) # 各关节handle
        self.jointangel=[-111.5,-22.36,88.33,28.08,-90,-21.52]
        # 配置方块信息
        self.cubenum =cubenum
        self.filenum =filenum
        self.cubename= 'imported_part_'
        self.filename= 'test-10-obj-0'
        self.cubeHandle = np.zeros((12,), dtype=np.int) # 各cubehandle
        self.obj_colors=[]
        self.obj_positions=[]
        self.obj_orientations=[]
        self.obj_order=[]
        self.connect()
        self.ankleinit()
        self.cubeinit(self.filenum)


    def connect(self):  #连接v-rep
        print('Program started') # 关闭潜在的连接 
        vrep.simxFinish(-1) # 每隔0.2s检测一次，直到连接上V-rep 
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
            if self.clientID > -1: 
                break 
            else: 
                time.sleep(0.2) 
                print("Failed connecting to remote API server!") 
        print("Connection success!")
        for i in range(12):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.cubename + str(i), vrep.simx_opmode_blocking) 
            self.cubeHandle[i] = returnHandle 
            print(returnHandle)
        vrep.simxSetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, self.tstep, vrep.simx_opmode_oneshot) # 保持API端与V-rep端相同步长
        vrep.simxSynchronous(self.clientID, True) # 然后打开同步模式 
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot) 

    def ankleinit(self):
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.jointName + str(i+1), vrep.simx_opmode_blocking) 
            self.jointHandle[i] = returnHandle 
        print('Handles available!') 
        vrep.simxSynchronousTrigger(self.clientID) # 让仿真走一步 
        for i in range(self.jointNum):
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetJointTargetPosition(self.clientID, self.jointHandle[i],self.jointangel[i]/self.RAD2DEG, vrep.simx_opmode_oneshot)  #设置关节角
            vrep.simxPauseCommunication(self.clientID, False)
            vrep.simxSynchronousTrigger(self.clientID) # 进行下一步 
            vrep.simxGetPingTime(self.clientID) # 使得该仿真步走完

    def cubeinit(self,filenum):
        fileadd=os.path.join('test-cases',self.filename+str(filenum)+'.txt')
        print(fileadd)
        file = open(fileadd, 'r')
        file_content = file.readlines() 
        for object_idx in range(self.cubenum):
            file_content_curr_object = file_content[object_idx].split()
            self.obj_order.append(file_content_curr_object[0])
            self.obj_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
            self.obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
            self.obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
        file.close()
        for j in range(self.cubenum):
            i=int(self.obj_order[j])
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetObjectOrientation(self.clientID,self.cubeHandle[i],-1,self.obj_orientations[j],vrep.simx_opmode_oneshot)
            '''print("orientation:")
            print(self.obj_orientations[i])'''
            vrep.simxPauseCommunication(self.clientID, False)
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetObjectPosition(self.clientID,self.cubeHandle[i],-1,self.obj_positions[j],vrep.simx_opmode_oneshot)
            '''print("position:")
            print(self.obj_positions[i])'''
            vrep.simxPauseCommunication(self.clientID, False)

    def disconnect(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot)
        vrep.simxFinish(self.clientID)
        print ('Program ended!')
        
    def get_clientID(self):
        return self.clientID

    def ur5moveto(self,x,y,z):
        vrep.simxSynchronousTrigger(self.clientID) # 让仿真走一步 
        self.targetQuaternion[0]=0.707
        self.targetQuaternion[1]=0
        self.targetQuaternion[2]=0.707
        self.targetQuaternion[3]=0      #四元数
        self.targetPosition[0]=x;
        self.targetPosition[1]=y;
        self.targetPosition[2]=z;
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









myur=UR5(12,2)

