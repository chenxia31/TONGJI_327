from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "video.avi", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()



def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, (int(c1[0].item()), int(c1[1].item())), (int(c2[0].item()), int(c2[1].item())),color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0].item()), int(c1[1].item())), (int(c2[0].item()), int(c2[1].item())),color, -1)
    cv2.putText(img, label, (int(c1[0].item()), int(c1[1].item()) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#Detection phase
videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture(videofile)  

#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # 
        #         在这里修改你的代码: 比如加虚拟线圈
        #
        # output为所有识别的输出结果，通过调用write函数来绘制识别框
        # 可以参考write中一些opencv的操作
        # PyTorch的Tensor基本操作：https://pytorch.org/docs/stable/tensors.html
        # OpenCV-python对图像的基本操作：https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        list(map(lambda x: write(x, frame), output)) # 使用map和lambda函数,类似For循环操作

        # 打印output,每一个列表是tensor
        for obj in output:
            print(obj[1:5])
            print('物体种类是'+classes[int(obj[-1])])

        # 示例代码1 ：对output的结果进行计数，并write到frame上
        # 1. 对output的结果进行计数
        # res={}
        # for i in output[:,-1]:
        #     if classes[int(i)] not in res:
        #         res[classes[int(i)]]=1
        #     else:
        #         res[classes[int(i)]]=res[classes[int(i)]]+1
        # # 2. write到frame上
        # offset=0
        # for i in res:
        #     cv2.putText(frame, i+':'+str(res[i]), (0, 50+offset*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #     offset+=1

        # 示例代码2 : 判断是否在四边形区域中，可以作为虚拟线圈和车道流量识别参考
        # def isin_poly(position,poly):
        #     # postion是目标位置,poly是四边形的坐标,四个坐标
        #     px,py=position
        #     is_in=False
        #     for i,corner in enumerate(poly):
        #         next_i = i + 1 if i + 1 < len(poly) else 0
        #         x1, y1 = corner
        #         x2, y2 = poly[next_i]
        #         if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
        #             is_in = True
        #             break
        #         if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
        #             x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        #             if x == px:  # if point is on edge
        #                 is_in = True
        #                 break
        #             elif x > px:  # if point is on left-side of line
        #                 is_in = not is_in
        #     return is_in    
        # for obj in output:
        #     # 需根据拍摄视频的道路设置线圈坐标，可设置多个虚拟线圈
        #     # 或者确定车道四个角的坐标
        #     # 可根据识别结果进行下一步展示和研究
        #     poly11=[[222, 300],
        #     [425.3901036528507, 300],
        #     [425, 50],
        #     [222, 50]]
        #     x_mid=(obj[1]+obj[3])/2
        #     y_mid=(obj[2]+obj[4])/2
        #     print(isin_poly((x_mid,y_mid),poly11)) 
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     






