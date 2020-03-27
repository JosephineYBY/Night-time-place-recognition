#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from pylab import *
import os
import random
from PIL import Image
from img_enhance import clahe_al
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms



def process_image(path):
    images=[cv2.imread(file) for file in glob.glob(path)] 
    #images=images.transpose((2,0,1))
    con_img=[]
    for i in range(len(images)):
        data=cv2.resize(images[i],(256,256),interpolation=cv2.INTER_AREA)
        clahe=cv2.createCLAHE(clipLimit=17.0,tileGridSize=(4,4))
        ch_image=clahe.apply(data)
		con_img.append((ch_image-127.723)/73.959)
    return con_img


class Mydataset(nn.Module):
    def __init__(self,files,path,default_path=None,loader=None,input_type=None):
        super(Mydataset,self).__init__()
        if input_type=='train':
            i=0
        elif input_type=='test': 
            i=15
        merge=[]
        for file in files:
            if not os.path.isdir(file):#if it is a folder, not open it
                f=open(path+'/'+file)
                lines=f.readlines()
                i+=1
                for line in lines:
                    content=line.rstrip().split(' ')
                    name=content[0]
                    filename=os.path.splitext(file)[0]
                    merge.append((filename,name,i))
        pair_img_label=[]
        for item in range(len(merge)):
            dic,fn,label=merge[item]
            path_conv=default_path+dic+'/'+fn
            image=loader(path_conv)
            pair_img_label.append((image,label))
        self.images=pair_img_label
        self.loader=loader
        self.input_type=input_type
    
    def __getitem__(self,item):
		label =None
        img0_tuple=random.choice(self.images)
        
        if item%2==1:
            label=1.0
            while True:
                img1=random.choice(self.images)
                if img0[1]==img1[1]:
                    break
        else:
            label=0.0
            while True:
                img1=random.choice(self.images)
                if img0[1]!=img1[1]:
                    break

        
        img0=torch.tensor(img0[0],dtype=torch.float)
        img1=torch.tensor(img1[0],dtype=torch.float)
        y=torch.from_numpy(np.array([label], dtype=np.float32))
        return img0,img1,y
    def __len__(self):
        return len(self.images)

