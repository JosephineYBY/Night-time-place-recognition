#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from build_dataset import Mydataset,process_image,build_data,two_input




def train_epoch(model,criterion,optim,train_loader,batch_size,epochs):
    use_stuff={'epoch':[],'train_accuracy':[],'train_loss':[],'true sample num'$
    test_stuff={'test_accuracy':[],'true sample num':[],'false sample num':[]}
    best_acc=0
    model.train()
    for epoch in range(epochs):
        print('Epoch:', epoch+1)
        train_acc=0
        train_fal=0
        loss_total=0
        for batch_idx,(image0,image1,label) in enumerate(train_loader):
            image0,image1,label=image0.cuda(),image1.cuda(),label.cuda()
            print('batch_index:',(batch_idx+1))
            optim.zero_grad()
            output,output1,output2=model(image0,image1)
            loss=criterion(output1,output2,label)
            _,pred=output.max(1)
            train_correct=(pred==label.transpose(1,0)).sum()
            train_false=(pred!=label.transpose(1,0)).sum()
            train_acc+=train_correct.data.item()
            train_fal+=train_false.data.item()
            print('train_acc number:',train_acc)
            print('train_fal number:',train_fal)

            loss.backward()
            optim.step()
            loss_total+=loss.data.item()
            cur_acc=train_acc/len(train_loader.dataset)
        
#    best_acc=cur_acc
        torch.save(model.state_dict(),'model.pt')
        use_stuff['epoch'].append(epoch+1)
        use_stuff['train_accuracy'].append(cur_acc)
        use_stuff['train_loss'].append(loss_total/(batch_idx+1))
        use_stuff['true sample num'].append(train_acc)
        use_stuff['false sample num'].append(train_fal)
        
    with open('use_stuff.csv','a', newline="") as f:
       writer=csv.writer(f)
       for i in use_stuff:
          writer.writerow([i,use_stuff[i]])
    f.close()
    return use_stuff
		
        
def test_file(model,test_loader):
    test_stuff={'test_accuracy':[],'true sample num':[],'false sample num':[]}
    model.load_state_dict(torch.load('model.pt'))
   
    correct=0
    false=0
    score={'output_score':[]}
    num_img=0
    AP=0
    recall=0
    idx=0
    Precision=0
    FN=TP=FP=TN=0
    de_multiply=0
    with torch.no_grad():
        model.eval()
        for test_image0,test_image1,test_label in test_loader:
            idx+=1
            num_img+=2
            output,output1,output2=model(test_image0.cuda(),test_image1.cuda())
            eucl_dis=F.pairwise_distance(output1,output2)
            #output=torch.cat((output1,output2),1)
            _,z=output.max(1)
      
			correct+=(z==test_label.cuda()).sum().cpu().item()
            false+=(z!=test_label.cuda()).sum().cpu().item()
            TP+=((z==1)==(test_label==1).cuda()).sum().cpu().item()
            TN+=((z==0)==(test_label==0).cuda()).sum().cpu().item()
            FP+=((z==1)!=(test_label==1).cuda()).sum().cpu().item()
            FN+=((z==0)!=(test_label==0).cuda()).sum().cpu().item()
            de_multiply+=(TP**2/((TP+FP)*(TP+FN)))
            AP+=de_multiply/num_img
            score['output_score'].append(output.cpu())
           # print('Euclian distance:',eucl_dis.mean())
            with open('score_pr.csv','a',newline="") as f:
                w=csv.writer(f)
                w.writerow([idx+1,'Precision',TP/(TP+FP),'Recall',TP/(TP+FN)])
            f.close()
    with open('mAP.csv','a',newline="") as f:
        w=csv.writer(f)
        w.writerow([idx+1,'mAP',AP/num_img])
    f.close()

    accuracy=correct/len(test_loader.dataset)
    test_stuff['test_accuracy'].append(accuracy)
    test_stuff['true sample num'].append(correct)
    test_stuff['false sample num'].append(false)
    print('test acc:{:.6f}'.format(accuracy))

    with open('test_stuff.csv','w', newline="") as f:
       writer=csv.writer(f)
       for i in test_stuff:
          writer.writerow([i,test_stuff[i]])
    f.close()

    with open('score.csv','w', newline="") as f:
       writer=csv.writer(f)
       for i in score:
          writer.writerow([i,score[i]])
    f.close()
    return test_stuff




