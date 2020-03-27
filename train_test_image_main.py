#!/usr/bin/env python
# coding: utf-8



from resnet50_app import SiameseResnet50
import cv2
import datetime
import os
import time
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from pylab import *
from scipy.io import loadmat
from torch.utils.data import Dataset,DataLoader
from build_dataset import Mydataset, process_image
from model_cnn import MatchnetFeature
from train_func import train_epoch,test_file
from loss import ContrastiveLoss



path='../dataset/DNIM/Image/train/'
path2='../dataset/DNIM/Image/test/'

root=r'../dataset/DNIM/time_stamp/train/'
root2=r'../dataset/DNIM/time_stamp/test/'
files_txt_train=os.listdir(root)
files_txt_test=os.listdir(root2)
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)




train_data=Mydataset(files_txt_train,root,path,loader=process_image,input_type='train')
train_loader=DataLoader(train_data,batch_size=64)
test_data=Mydataset(files_txt_test,root2,path2,loader=process_image,input_type='test')
test_loader=DataLoader(test_data,batch_size=1)



model=MatchnetFeature(64).cuda()
#model=SiameseResnet50().cuda()
optim=torch.optim.Adam(model.parameters(),lr=0.001)
criterion=ContrastiveLoss()
#criterion=nn.CrossEntropyLoss()
#criterion=nn.BCELoss()
start_time=time.time()
train_epoch(model,criterion,optim,train_loader,64,100)
#test_file(model,test_loader)
#calculate time cost
total_time=time.time()-start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

