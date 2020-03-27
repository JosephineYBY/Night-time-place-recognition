#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision


class SiameseResnet50(nn.Module):
    def __init__(self):
        super(SiameseResnet50,self).__init__()
        self.layer1=nn.Conv2d(1,3,3,stride=1,padding=0)
        self.resnet50=torchvision.models.resnet50()
        self.ch_para=nn.Sequential(
            nn.Linear(self.resnet50.fc.out_features,17),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(17)
        )

        self.fc=nn.Sequential(
            nn.Linear(17*2,2),
            nn.Sigmoid()
        )
        
    def forward_once(self,batch):
		x=self.layer1(batch)
        x=self.resnet50(batch)
        x=self.ch_para(x)
        return x
    
    def forward(self,input1,input2):
        output1=self.forward_once(input1)
        output2=self.forward_once(input2)
        output=torch.cat((output1,output2),1)
        output=self.fc(output)
        return output, output1,output2
        
        
    
    


