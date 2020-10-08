# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:26:30 2020

@author: user
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os

class classification1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(classification1,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.fc2(out)
        # out=F.softmax(out,dim=-1)  #pytorch下交叉熵函数自带softmax,模型中一般不写
        #print(out.shape)
        return out

class classification2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super(classification2,self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(p = 0.3)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)

        # out=F.softmax(out,dim=-1)  #pytorch下交叉熵函数自带softmax,模型中一般不写
        #print(out.shape)
        return out

def dataload(dir):
    setname = dir.split('/')[-1] + '_data.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname1 = dir.split('/')[-1] + '_violence.npy'
    labelpath1 = os.path.join(dir,labelname1)
    #labelpath = dir + '_violence.npy'
    
    labelname2 = dir.split('/')[-1] + '_valenceClass.npy'
    labelpath2 = os.path.join(dir,labelname2)

    labelname3 = dir.split('/')[-1] + '_arousalClass.npy'
    labelpath3 = os.path.join(dir,labelname3)

    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label1 = np.load(labelpath1)
    label2 = np.load(labelpath2)
    label3 = np.load(labelpath3)

    return data, label1, label2, label3

def dataload2(dir):
    setname = dir.split('/')[-1] + '_all_expansion.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname1 = dir.split('/')[-1] + '_all_expansion_violence.npy'
    labelpath1 = os.path.join(dir,labelname1)
    #labelpath = dir + '_violence.npy'
    labelname2 = dir.split('/')[-1] + '_all_expansion_valenceClass.npy'
    labelpath2 = os.path.join(dir,labelname2)

    labelname3 = dir.split('/')[-1] + '_all_expansion_arousalClass.npy'
    labelpath3 = os.path.join(dir,labelname3)

    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label1 = np.load(labelpath1)
    label2 = np.load(labelpath2)
    label3 = np.load(labelpath3)

    return data, label1, label2, label3


if __name__ == "__main__":
    input_size = 640
    sequence_length = 1
    hidden_size = 512
    hidden_size1 = 512
    hidden_size2 = 256
    num_layers = 1
    num_classes1 = 3
    num_classes2 = 3
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 512

    train_dir = '../7concatenation/trainval'   #修改地址
    
    train_dataset, train_label1, train_label2, train_label3 = dataload2(train_dir)

    train_dataset = torch.Tensor(train_dataset)
    train_label1 = torch.LongTensor(train_label1)
    train_label2 = torch.LongTensor(train_label2)
    train_label3 = torch.LongTensor(train_label3)


    print(train_dataset.shape)
    print(train_label1.shape)
    print(train_label2.shape)
    print(train_label3.shape)


    trainset = Data.TensorDataset(train_dataset, train_label1, train_label2, train_label3)
    train_loader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    
    test_dir = '../7concatenation/test'   #修改地址
    test_dataset, test_label1, test_label2, test_label3 = dataload(test_dir)

    test_dataset = torch.Tensor(test_dataset)
    test_label1 = torch.LongTensor(test_label1)
    test_label2 = torch.LongTensor(test_label2)
    test_label3 = torch.LongTensor(test_label3)

    testset = Data.TensorDataset(test_dataset, test_label1, test_label2, test_label3)
    test_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    
    model1 = classification1(input_size, hidden_size, num_layers, num_classes1)  
    model1.cuda()
    model2 = classification2(input_size, hidden_size1, hidden_size2, num_layers, num_classes2)    
    model2.cuda()
    model3 = classification2(input_size, hidden_size1, hidden_size2, num_layers, num_classes2)   
    model3.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(),lr = learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(),lr = learning_rate)
    optimizer3 = torch.optim.Adam(model3.parameters(),lr = learning_rate)


    
    Line1 = np.ones([test_dataset.shape[0],1],dtype = np.int64)
    Line2 = np.zeros([test_dataset.shape[0],1],dtype = np.int64)
    Line3 = np.array(['Q0']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    Line4 = np.array(['STANDARD']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    index = ['name'+str(i) for i in range(test_dataset.shape[0])]
    index = np.array(index).reshape(test_dataset.shape[0],1)
    
    a = 0
    b = 0
    
    for epoch in range(num_epochs):
        valence_train_acc = 0
        valence_test_acc = 0
        
        arousal_train_acc = 0
        arousal_test_acc = 0

        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0

        mAP_train = 0
        mAP_test = 0
        labellist = []
        labellist = np.array(labellist)
        predlist = []
        predlist = np.array(predlist)
        for i,(X_train, y_train1, y_train2, y_train3) in enumerate(train_loader):
                      
            X_train,y_train1,y_train2,y_train3 = X_train.view(-1,  input_size),y_train1.view(-1),y_train2.view(-1),y_train3.view(-1)  #二维数据就可以，没必要用三维
            X_train,y_train1,y_train2,y_train3 = Variable(X_train).cuda(),Variable(y_train1).cuda(), Variable(y_train2).cuda(), Variable(y_train3).cuda()
            
            outputs1 = model1(X_train)
            outputs2 = model2(X_train)
            outputs3 = model3(X_train)

            loss1 = criterion(outputs1, y_train1)
            loss2 = criterion(outputs2, y_train2)    #y_train要是int64类型，给种类标号就行，不用one-hot，函数可以自己转换
            loss3 = criterion(outputs3, y_train3)

            #print(loss.cpu().detach().data)   #输出loss值，方便观察网络训练情况
            _,pred1 = torch.max(outputs1.data,1)
            _,pred2 = torch.max(outputs2.data,1)
            _,pred3 = torch.max(outputs3.data,1)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            
            loss1.backward()
            loss2.backward()
            loss3.backward()
            
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            
            running_loss1 += loss1.data
            running_loss2 += loss2.data
            running_loss3 += loss3.data

            
            valence_train_acc += torch.sum(pred2 == y_train2.data)
            arousal_train_acc += torch.sum(pred3 == y_train3.data)


            
            '''
            print(pred.shape)
            '''
            #print(y_train.cpu().reshape(batch_size,1))   
            
            #outputs = outputs.cpu().detach().numpy()
            #print(outputs.shape)
            #pred = pred.reshape(batch_size,1)
            
            #print(outputs[:,1])
            
            #out1 = np.concatenate((Line1,Line2,index,y_train.cpu().reshape(batch_size,1)),axis = 1)   #这句报了维度的错，可能和我把二维改三维有关
            #out2 = np.concatenate((Line1,Line3,index,Line1,outputs[:,0].reshape(batch_size,1),Line4),axis = 1)

            #np.savetxt('./train/train_label_'+str(i)+'.txt',out1,fmt = '%s')
            #np.savetxt('./train/train_pred_'+str(i)+'.txt',out2,fmt = '%s')
          
            

    
        for i,(X_test, y_test1, y_test2, y_test3) in enumerate(test_loader):
            
            X_test = X_test.view(-1, input_size)
            #print(y_test.shape)
            y_test1, y_test2, y_test3 = y_test1.view(-1), y_test2.view(-1), y_test3.view(-1)
            X_test, y_test1, y_test2, y_test3 = Variable(X_test).cuda(), Variable(y_test1).cuda(), Variable(y_test2).cuda(), Variable(y_test3).cuda()
            
            outputs1 = model1(X_test)
            _,pred1 = torch.max(outputs1.data,1)
            outputs2 = model2(X_test)
            _,pred2 = torch.max(outputs2.data,1)
            outputs3 = model3(X_test)
            _,pred3 = torch.max(outputs3.data,1)

            
            valence_test_acc += torch.sum(pred2 == y_test2.data)
            arousal_test_acc += torch.sum(pred3 == y_test3.data)

            #mAP_test = compute_map(pred,y_test)
            
            outputs1 = outputs1.cpu().detach().numpy()
            
            labellist = np.concatenate((labellist,y_test1.cpu()),axis = 0)
            labellist = np.array(labellist,dtype = np.int64)

            predlist = np.concatenate((predlist,outputs1[:,1]))
            #print(labellist)
            #print(predlist)
        labellist = np.concatenate((Line1,Line2,index,labellist.reshape(-1,1)),axis = 1)
        predlist = np.concatenate((Line1,Line3,index,Line1,predlist.reshape(-1,1),Line4),axis = 1)
        np.savetxt('./test/test_pred.txt',labellist,fmt = '%s')
        np.savetxt('./test/test_label.txt',predlist,fmt = '%s')
                
            
        print('Epoch[%d/%d], Loss1: %.4f, Loss2: %.2f, Loss3: %.2f, valence_train_acc: %.2f %%, valence_test_acc:%.2f %%, arousal_train_acc: %.2f %%, arousal_test_acc:%.2f %%' \
              % (epoch+1, num_epochs, running_loss1, running_loss2, running_loss3, valence_train_acc*100.0/len(trainset), valence_test_acc*100.0/len(testset),arousal_train_acc*100.0/len(trainset), arousal_test_acc*100.0/len(testset)))
    
        
        if valence_test_acc*100.0/len(testset) >= a and (valence_train_acc*100.0/len(trainset)>90):
            torch.save(model2,'./multi/classificaton_all_valence.pkl')
            print('save valence model!')
            a = valence_test_acc*100.0/len(testset)
        elif arousal_test_acc*100.0/len(testset) >= b and (arousal_train_acc*100.0/len(trainset)>90):
            torch.save(model3,'./multi/classificaton_all_arousal.pkl')
            print('save arousal model!')
            b = arousal_test_acc*100.0/len(testset)



