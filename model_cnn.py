#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from pylab import *


class MatchnetFeature(nn.Module):
    def __init__(self,num_class=15):
        super(MatchnetFeature,self).__init__()
        
        self.model=nn.Sequential(
            nn.Conv2d(1,24,7,stride=1,padding=0), #250*250*24
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2), #124*124*24

            nn.Conv2d(24,64,5,stride=1,padding=0), #120*120*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2), #59*59*64

            nn.Conv2d(64,96,3,stride=1,padding=0), #57*57*96
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96,64,3,stride=1,padding=0), #55*55*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
           # nn.MaxPool2d(3,stride=2), #output size 27x27x64

            nn.Conv2d(64,48,3,stride=1,padding=0), #53x53x48
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48,24,3,stride=1,padding=0), #51x51x24
			nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2)  #25x25*24

        )
        
        self.batch_size=batch_size
        self.fc=nn.Sequential(
            nn.Dropout(),
            nn.Linear(25*25*24,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,batch_size),
            nn.BatchNorm1d(batch_size)

        )
           
		self.fc2=nn.Sequential(
            nn.Linear(batch_size*2,2),
            nn.Sigmoid()
        )

    def forward_once(self,batch):
        x=self.model(batch)
        #flatten output
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return x

    def forward(self,input1,input2):
        output1=self.forward_once(input1)
        output2=self.forward_once(input2)
        output=torch.cat((output1,output2),1)
       # t=output1-output2
       # output=torch.abs(t)
        output=self.fc2(output)
        return output,output1,output2

        

   
    
    
    
    

