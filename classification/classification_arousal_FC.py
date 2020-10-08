# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:17:30 2020

@author: user
"""


#coding=utf-8
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os


class classification(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_layers, num_classes):
        super(classification,self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)

        #out=F.softmax(out,dim=-1)  #pytorch下交叉熵函数自带softmax,模型中一般不写
        #print(out.shape)
        return out

def dataload(dir):
    setname = dir.split('/')[-1] + '_set.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname = dir.split('/')[-1] + '_arousalClass.npy'
    labelpath = os.path.join(dir,labelname)
    #labelpath = dir + '_violence.npy'
    
    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label = np.load(labelpath)

    
    return data, label

def dataload2(dir):
    setname = dir.split('/')[-1] + '_arousal_expansion.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname = dir.split('/')[-1] + '_arousal_expansion_arousalClass.npy'
    labelpath = os.path.join(dir,labelname)
    #labelpath = dir + '_violence.npy'
    
    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label = np.load(labelpath)

    
    return data, label


if __name__ == "__main__":
    input_size = 1152
    sequence_length = 1
    hidden_size1 = 256
    hidden_size2 = 128
    num_layers = 1
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 200
    batch_size = 512

    train_dir = '../1concatenation/trainval'   #修改地址
    
    train_dataset, train_label = dataload(train_dir)

    train_dataset = torch.Tensor(train_dataset)
    train_label = torch.LongTensor(train_label)
    print(train_dataset.shape)
    print(train_label.shape)
    
    trainset = Data.TensorDataset(train_dataset, train_label)
    train_loader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    
    test_dir = '../1concatenation/test'   #修改地址
    test_dataset, test_label = dataload(test_dir)

    test_dataset = torch.Tensor(test_dataset)
    test_label = torch.LongTensor(test_label)
    
    testset = Data.TensorDataset(test_dataset, test_label)
    test_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    
    
    model = classification(input_size, hidden_size1, hidden_size2, num_layers, num_classes)   
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    
    Line1 = np.ones([test_dataset.shape[0],1],dtype = np.int64)
    Line2 = np.zeros([test_dataset.shape[0],1],dtype = np.int64)
    Line3 = np.array(['Q0']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    Line4 = np.array(['STANDARD']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    index = ['name'+str(i) for i in range(test_dataset.shape[0])]
    index = np.array(index).reshape(test_dataset.shape[0],1)
    a = 0
    for epoch in range(num_epochs):
        train_acc = 0
        test_acc = 0
        running_loss = 0.0
        mAP_train = 0
        mAP_test = 0
        labellist = []
        labellist = np.array(labellist)
        predlist = []
        predlist = np.array(predlist)
        for i,(X_train, y_train) in enumerate(train_loader):
                      
            X_train,y_train = X_train.view(-1,  input_size),y_train.view(-1)  #二维数据就可以，没必要用三维
            X_train,y_train = Variable(X_train).cuda(),Variable(y_train).cuda()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)    #y_train要是int64类型，给种类标号就行，不用one-hot，函数可以自己转换
            #print(loss.cpu().detach().data)   #输出loss值，方便观察网络训练情况
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _,pred = torch.max(outputs.data,1)
            running_loss += loss.data   
            train_acc += torch.sum(pred == y_train.data)
            
            
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


    
        for i,(X_test, y_test) in enumerate(test_loader):
            
            X_test = X_test.view(-1, input_size)
            #print(y_test.shape)
            y_test = y_test.view(-1)
            X_test, y_test = Variable(X_test).cuda() , Variable(y_test).cuda()
            
            outputs = model(X_test)
            _,pred = torch.max(outputs.data,1)

            test_acc += torch.sum(pred == y_test.data)
            #mAP_test = compute_map(pred,y_test)
            
            outputs = outputs.cpu().detach().numpy()
            
            labellist = np.concatenate((labellist,y_test.cpu()),axis = 0)
            labellist = np.array(labellist,dtype = np.int64)

            predlist = np.concatenate((predlist,outputs[:,2]))
            #print(labellist)
            #print(predlist)
        labellist = np.concatenate((Line1,Line2,index,labellist.reshape(-1,1)),axis = 1)
        predlist = np.concatenate((Line1,Line3,index,Line1,predlist.reshape(-1,1),Line4),axis = 1)
        np.savetxt('./test/test_pred_arousal.txt',labellist,fmt = '%s')
        np.savetxt('./test/test_label_arousal.txt',predlist,fmt = '%s')
            
            
        print('Epoch[%d/%d], Loss: %.4f, train_acc: %.4f %%, test_acc:%.4f %%' % (epoch+1, num_epochs,  running_loss, 
                                                                train_acc*100.0/len(trainset), test_acc*100.0/len(testset)))
    

        if test_acc*100.0/len(testset) >= a: #and (train_acc*100.0/len(trainset)>90):
            torch.save(model,'./model/classificaton_arousal.pkl')
            print('save model!')
            a = test_acc*100.0/len(testset)