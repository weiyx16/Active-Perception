import numpy as np
import vrep
import os

"""
   used to initialize the object location in the scene
   Notice the object name is imported_part_ + index
   and the config info is stored in the txt file
"""

class cube():      #cube的数量和文件名

    def __init__(self,cubenum,filenum):
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
        self.tstep = 0.005             # 定义仿真步长
        self.cubenum =cubenum
        self.filenum =filenum
        self.cubename= 'imported_part_'
        self.filename= 'test-10-obj-0' #TODO: rename the file
        self.cubeHandle = np.zeros((self.cubenum,), dtype=np.int) # 各cubehandle
        self.obj_colors=[]
        self.obj_positions=[]
        self.obj_orientations=[]


    def cubeinit(self):
        for i in range(self.cubenum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.cubename + str(i), vrep.simx_opmode_blocking) 
            self.cubeHandle[i] = returnHandle 
        fileadd=os.path.join('scenes',self.filename+str(self.filenum)+'.txt')
        print(fileadd)
        file = open(fileadd, 'r')
        file_content = file.readlines() 
        for object_idx in range(self.cubenum):
            file_content_curr_object = file_content[object_idx].split()
            self.obj_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
            self.obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
            self.obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
        file.close()
        for i in range(self.cubenum):
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetObjectOrientation(self.clientID,self.cubeHandle[i],-1,self.obj_orientations[i],vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.clientID, False)
            vrep.simxPauseCommunication(self.clientID, True) 
            vrep.simxSetObjectPosition(self.clientID,self.cubeHandle[i],-1,self.obj_positions[i],vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.clientID, False)

