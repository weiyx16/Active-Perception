#!/usr/bin/env python  
#coding=utf-8  
import time
import threading
import glob
import urx
import time
import json
import copy
import math
import serial
import numpy



inhead = 'RECV'  # 接收数据头
outhead = 'SEND'  # 发送数据头
robot_vel = 0.5    #accerlation
robot_acc = 0.3 
handlen = 0.30 #记得在json矫正末端位置
V_kindNum = 2
suckerlength = 0.1  #0.285m = 从装置盘到吸盘收回的长度 -- 需要TCP修订  0.100m = 吸盘打出的长度  RX应该是rad 然后i

class SerialData(threading.Thread):  # 创建threading.Thread的子类SerialData

    def __init__(self):
        threading.Thread.__init__(self)     # 初始化线程

    def open_com(self, port, baud):         # 打开串口[0.030191486715516257, -0.6276073061268677, 0.5239050666771823, -1.6442984738891193, -2.497057107357831, -0.02130323021050967]
        self.ser = serial.Serial(port, baud, timeout=0.5)
        return self.ser

    def com_isopen(self):  # 判断串口是否打开
        return self.ser.isOpen()

    def send_data(self, data, outhead=outhead):  # 发送数据
        self.ser.write(outhead + data)

    def next(self):  # 接收的数据组
        all_data = ''
        # if inhead == self.ser.read(1) :
        all_data = self.ser.readline()  # 读一行数据
        return all_data

    def close_listen_com(self):  # 关闭串口
        return self.ser.close()


class Hand():
    def __init__(self, rec_data):
        self.rec_data = rec_data  # 初始化线程
    def handin(self):
        rec_data.send_data('i')  # 手指向内
    def handout(self):
        rec_data.send_data('o')  # 手指向外

    def polepush(self):
        rec_data.send_data('p')  # 吸盘伸出

    def poleback(self):
        rec_data.send_data('b')  # 吸盘收回

    def bumpopen(self):
        rec_data.send_data('s')  # 吸盘吸气

    def bumpclose(self):
        rec_data.send_data('l')  # 吸盘释放
    def handstop(self):
        rec_data.send_data('a')




def opening_Hand(rec_data):
    # 搜索匹配字符 ‘/dev/ttyACM0’的设备  connect to arduino
    allport = glob.glob('/dev/ttyACM8*')
    port = allport[0]
    baud = 9600
    openflag = rec_data.open_com(port, baud)  # 打开串口
    if openflag:
        print('i open %s at %s suc=cessfully!' % (allport[0], baud))



rec_data = SerialData()


#if __name__ == '__main__':
def mymain(posin):
     with open("./point.json", "r") as f:  #json的位置可改
        json_obj = json.loads(f.read())
        point0 = json_obj["0"]  
        point1 = json_obj["1"]
        point2 = json_obj["2"]
        point3 = json_obj["3"]
        point4 = json_obj["4"]
        point5 = json_obj["5"]

        point10 = json_obj["10"] 
        point11 = json_obj["11"] 
        point12 = json_obj["12"] 
        point13 = json_obj["13"] 
        point14 = json_obj["14"] 
        point15= json_obj["15"] 
        point16= json_obj["16"] 
        point17= json_obj["17"] 
        point18 = json_obj["18"] 

        robot = urx.Robot("192.168.1.111")     
     #rec_data = SerialData()  # 为串口开辟线程
     opening_Hand(rec_data)
     Hand1 = Hand(rec_data) 
     flag=0
     num=0 
     com=1024   
     sensor = []
     Hand1.bumpopen()
     order = 100  
     while(order != 99):
       order=input("please input the order")
       print order
       if order == 1:
          Hand1.bumpopen()
          Hand1.polepush()
          time.sleep(1.5)
       elif order == 4:
          Hand1.poleback()   
       elif order == 2: 
          num2 =0             #抓取指令     
       	  while (num<70 ):
            Hand1.handin()
            com=int(rec_data.next())
            if not com =='' :          	
              print 'look what i got :%s'%(com)
              num2 +=1
              if(com<900):
                flag=1
              if (flag==1):
                sensor.append(com)
                num+=1
          flag=0
          Hand1.handstop()
          num=0
          fw = open('arduino.txt', 'w')  
          fw.write(str(sensor))
          fw.close()
       elif order == 3:           #松开释放指令
           Hand1.handout()
           time.sleep(0.5)
           Hand1.handstop()
           Hand1.bumpclose()
       elif order == 9:           #松开释放指令
           Hand1.handin()
           time.sleep(0.2)
           Hand1.handstop()
           Hand1.bumpclose()
       elif order == 10:
           robot.movel(point0, acc = robot_acc, vel = robot_vel)
       elif order == 11:
       	   robot.movel(point1, acc = robot_acc,vel = robot_vel)
       elif order == 12:
       	   robot.movel(point2, acc = robot_acc,vel = robot_vel)
       elif order == 13:
       	   robot.movel(point3, acc = robot_acc,vel = robot_vel)
       elif order == 14:
       	   robot.movel(point4, acc = robot_acc,vel = robot_vel)
       elif order == 15:
           robot.movel(point5, acc = robot_acc,vel = robot_vel)
       elif order == 19:
           robot.movel(point18, acc = robot_acc,vel = robot_vel)
       elif order == 20:
           robot.movel(posin, acc = robot_acc,vel = robot_vel)
       elif order == 21:
           angle = 0.00
           trans = robot.get_pose()
           orderangle = input("angle? :")
           while(orderangle != 0):
               angle = orderangle*3.139/180.0
               if angle >3.139:
                   angle = 3.139 - angle
               trans.orient.rotate_zb(angle)
               robot.set_pose(trans, acc=0.5, vel=0.2)
               orderangle = input("angle? :")
       elif order == 30:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point10, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point0, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 31:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point11, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point2, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 32:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point12, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point1, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 33:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point13, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point2, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 34:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point14, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point3, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 35:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point15, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point3, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 36:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point16, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point3, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 37:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point17, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point3, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()
       elif order == 38:
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            robot.movel(point18, acc = robot_acc,vel = robot_vel)
            Hand1.bumpopen()
            Hand1.polepush()
            time.sleep(1.5)
            Hand1.poleback()
                         #抓取指令     
            while (num<70 ):
              Hand1.handin()
              com=int(rec_data.next())
              if not com =='' :           
                print 'look what i got :%s'%(com)
                if(com<900):
                  flag=1
                if (flag==1):
                  sensor.append(com)
                  num+=1
            flag=0
            Hand1.handstop()
            num=0
            fw = open('arduino.txt', 'w')  
            fw.write(str(sensor))
            fw.close()
            robot.movel(point5, acc = robot_acc,vel = robot_vel)
            time.sleep(1.5)
            robot.movel(point0, acc = robot_acc,vel = robot_vel)
            Hand1.handout()
            time.sleep(0.5)
            Hand1.handstop()
            Hand1.bumpclose()

       robot.movel(point5, acc = robot_acc,vel = robot_vel)
       