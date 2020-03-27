#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from pylab import *
import os
import random
from PIL import Image
import glob
import cv2
#from resnet50_app import SiameseResnet50
#from model_cnn import MatchnetFeature
from torch.utils.data import Dataset,DataLoader
#from train_func import threashold_sigmoid



def process_image(path):
    images=[cv2.imread(file,0) for file in glob.glob(path)] 
    #images=images.transpose((2,0,1))
    con_img=[]
    image=[]
    total=0
    for i in range(len(images)):
        data=cv2.resize(images[i],(256,256),interpolation=cv2.INTER_AREA)
        clahe=cv2.createCLAHE(clipLimit=15.0,tileGridSize=(4,4))
        ch_image=clahe.apply(data)
        edges_clahe=cv2.Canny(ch_image,150,450)
        con_img.append(edges_clahe)
  
    return con_img




def build_data(files,path,default_path):
        i=0
        merge=[]
        for file in files:
            if not os.path.isdir(file):#if it is a folder, not open it
                f=open(path+'/'+file)
                #print(f)
                lines=f.readlines()
                #len_line.append(len(lines))
                #print(lines)
                i+=1
                for line in lines:
                    content=line.rstrip().split(' ')
                    name=content[0]
                    filename=os.path.splitext(file)[0]
                    #print(filename)
                    #label.append(i)
                    #images.append(name)
                    merge.append((filename,name,i))
        image=[]
        for item in range(len(merge)):
            dic,fn,label=merge[item]
            path_conv=default_path+dic+'/'+fn
            image.append([torch.tensor(process_image(path_conv),dtype=torch.float),label])
        #print(random.choice(image))
        
        return image




path='/home/byan/Documents/mid-term/dataset/DNIM/Image/test/'

root=r'/home/byan/Documents/mid-term/dataset/DNIM/time_stamp/test/'
files_txt_test=os.listdir(root)



image=build_data(files_txt_test,root,path)

test_data=DataLoader(image,batch_size=1)

model=SiameseResnet50()
model.load_state_dict(torch.load('model.pt'))


model.eval()
with torch.no_grad():
	for image,label in test_data:
        idx+=1
        query_img=image
        correct=0
        false=0
        recall=0
        Precision=0
        FN=TP=FP=TN=0
        num_img=1
		AP=0
        de_multiply=0
        for test_im, test_la in test_data:
            num_img+=1
            output,output1,output2=model(query_img.cuda(),test_im.cuda())
       # eucl_dis=F.pairwise_distance(output1,output2)
            #prediction=threashold_sigmoid(output)
            _,prediction=output.max(1)
            TP+=(prediction==test_la.cuda()==1).sum().cpu().item()
            TN+=(prediction==test_la.cuda()==0).sum().cpu().item()
            FP+=(prediction!=test_la.cuda()==1).sum().cpu().item()
            FN+=(prediction!=test_la.cuda()==0).sum().cpu().item()
            if ((TP+FP)*(TP+FN))==0:
               continue
            else:
               de_multiply+=(TP**2/((TP+FP)*(TP+FN)))
            correct+=(prediction==test_la.cuda()).sum().cpu().item()
            false+=(prediction!=test_la.cuda()).sum().cpu().item()
       # pre_recall['Precision'].append(TP/(TP+FP))
       # pre_recall['recall'].append(TP/(TP+FN))
                    AP+=de_multiply/num_img
            with open('score.csv','a',newline="") as f:
                w=csv.writer(f)
                w.writerow([idx+1,'Precision',TP/(TP+FP), 'Recall',TP/(TP+FN)])
            f.close()

        with open('mAP.csv','a',newline="") as f:
            w=csv.writer(f)
            w.writerow([idx+1,'mAP',AP/170])
        f.close()
        print('Epoch',idx+1,';','Accuracy',correct/169)







