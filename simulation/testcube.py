from __future__ import division
import numpy as np
import math
import vrep
import time
import select
import struct
import os

# TODO:
tstep = 0.005             # 定义仿真步长
test_obj_files = []
test_obj_colors = []
test_obj_positions = []
test_obj_orientations = []

def connect():	#连接v-rep
	print('Program started') # 关闭潜在的连接 
	vrep.simxFinish(-1) # 每隔0.2s检测一次，直到连接上V-rep 
	while True:
		clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
		if clientID > -1: 
			break 
		else: 
			time.sleep(0.2) 
			print("Failed connecting to remote API server!") 
	print("Connection success!")
	vrep.simxSetFloatingParameter(clientID, vrep.sim_floatparam_simulation_time_step, tstep, vrep.simx_opmode_oneshot) # 保持API端与V-rep端相同步长
	vrep.simxSynchronous(clientID, True) # 然后打开同步模式 
	vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot) 
	return clientID

def import_scene():
	with open('./scenes/test_scene_1.txt', 'r') as fs:
		file_content = fs.readlines()
		for object_idx in range(len(file_content)):
			file_content_curr_object = file_content[object_idx].split()
			test_obj_files.append(os.path.join('./blocks',file_content_curr_object[0]))
			test_obj_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
			test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
			test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])

def add_objects(clientID):
	for object_idx in range(len(test_obj_files)):
		curr_shape_name = 'shape_%02d' % object_idx
		curr_mesh_file = test_obj_files[object_idx]
		object_position = test_obj_positions[object_idx]
		object_orientation = test_obj_orientations[object_idx]
		object_color = test_obj_colors[object_idx]
		print(object_position + object_orientation + object_color)
		print([curr_mesh_file, curr_shape_name])
		ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(clientID, 'remoteApiCommandServer',vrep.sim_scripttype_childscript ,'simImportShape', [0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
		#ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(clientID, 'remoteApiCommandServer',vrep.sim_scripttype_childscript ,'simImportShape', [0, 0], [255.0, 0.0], curr_mesh_file, bytearray(), vrep.simx_opmode_blocking)
		if ret_resp == vrep.simx_return_ok:
			print('ok!')

		if ret_resp == 8:
			print('Failed to add new objects to simulation. Please restart.')
			exit()
		
if __name__ == "__main__":
	clientID = connect()
	import_scene()
	add_objects(clientID)