# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:54:44 2020

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
        self.dropout = nn.Dropout(p = 0.5)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc3(x)

        # out=F.softmax(out,dim=-1)  #pytorch下交叉熵函数自带softmax,模型中一般不写
        #print(out.shape)
        return out

def dataload(dir):
    setname = dir.split('/')[-1] + '_set.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname1 = dir.split('/')[-1] + '_violence.npy'
    labelpath1 = os.path.join(dir,labelname1)
    #labelpath = dir + '_violence.npy'
    
    labelname2 = dir.split('/')[-1] + '_valenceClass.npy'
    labelpath2 = os.path.join(dir,labelname2)

    
    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label1 = np.load(labelpath1)
    label2 = np.load(labelpath2)


    
    return data, label1, label2


if __name__ == "__main__":
    input_size = 1152
    sequence_length = 1
    hidden_size1 = 256
    hidden_size2 = 128  
    num_layers = 1
    num_classes1 = 2
    num_classes2 = 3
    learning_rate = 0.0001
    num_epochs = 200
    batch_size = 512

    train_dir = '../1concatenation/trainval'   #修改地址
    
    train_dataset, train_label1, train_label2 = dataload(train_dir)

    train_dataset = torch.Tensor(train_dataset)
    train_label1 = torch.LongTensor(train_label1)
    train_label2 = torch.LongTensor(train_label2)

    print(train_dataset.shape)
    print(train_label1.shape)
    print(train_label2.shape)

    trainset = Data.TensorDataset(train_dataset, train_label1, train_label2)
    train_loader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    
    test_dir = '../1concatenation/test'   #修改地址
    test_dataset, test_label1, test_label2 = dataload(test_dir)

    test_dataset = torch.Tensor(test_dataset)
    test_label1 = torch.LongTensor(test_label1)
    test_label2 = torch.LongTensor(test_label2)
    
    testset = Data.TensorDataset(test_dataset, test_label1, test_label2)
    test_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    
    model1 = classification2(input_size, hidden_size1, hidden_size2, num_layers, num_classes1)   
    model1.cuda()
    model2 = classification2(input_size, hidden_size1, hidden_size2, num_layers, num_classes2)    
    model2.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(),lr = 0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(),lr = learning_rate)

    
    Line1 = np.ones([test_dataset.shape[0],1],dtype = np.int64)
    Line2 = np.zeros([test_dataset.shape[0],1],dtype = np.int64)
    Line3 = np.array(['Q0']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    Line4 = np.array(['STANDARD']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    index = ['name'+str(i) for i in range(test_dataset.shape[0])]
    index = np.array(index).reshape(test_dataset.shape[0],1)
    
    a = 0.0
    b = 0.0
    
    for epoch in range(num_epochs):
        valence_train_acc = 0.0
        valence_test_acc = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        mAP_train = 0
        mAP_test = 0
        labellist = []
        labellist = np.array(labellist)
        predlist = []
        predlist = np.array(predlist)
        for i,(X_train, y_train1, y_train2) in enumerate(train_loader):
                      
            X_train,y_train1,y_train2 = X_train.view(-1,  input_size),y_train1.view(-1),y_train2.view(-1)  #二维数据就可以，没必要用三维
            X_train,y_train1, y_train2 = Variable(X_train).cuda(),Variable(y_train1).cuda(), Variable(y_train2).cuda()
            
            outputs1 = model1(X_train)
            outputs2 = model2(X_train)

            loss1 = criterion(outputs1, y_train1)
            loss2 = criterion(outputs2, y_train2)    #y_train要是int64类型，给种类标号就行，不用one-hot，函数可以自己转换

            #print(loss.cpu().detach().data)   #输出loss值，方便观察网络训练情况
            _,pred1 = torch.max(outputs1.data,1)
            _,pred2 = torch.max(outputs2.data,1)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss1.backward()
            loss2.backward()
            
            optimizer1.step()
            optimizer2.step()

            
            running_loss1 += loss1.data
            running_loss2 += loss2.data
            
            valence_train_acc += torch.sum(pred2 == y_train2.data)    
            

    
        for i,(X_test, y_test1, y_test2) in enumerate(test_loader):
            
            X_test = X_test.view(-1, input_size)
            #print(y_test.shape)
            y_test1, y_test2 = y_test1.view(-1), y_test2.view(-1)
            X_test, y_test1, y_test2 = Variable(X_test).cuda(), Variable(y_test1).cuda(), Variable(y_test2).cuda()
            
            outputs1 = model1(X_test)
            _,pred1 = torch.max(outputs1.data,1)
            outputs2 = model2(X_test)
            _,pred2 = torch.max(outputs2.data,1)
            
            valence_test_acc += torch.sum(pred2 == y_test2.data)
            #mAP_test = compute_map(pred,y_test)
            
            outputs1 = outputs1.cpu().detach().numpy()
            
            labellist = np.concatenate((labellist,y_test1.cpu()),axis = 0)
            labellist = np.array(labellist,dtype = np.int64)

            predlist = np.concatenate((predlist,outputs1[:,1]))
            #print(labellist)
            #print(predlist)
        labellist = np.concatenate((Line1,Line2,index,labellist.reshape(-1,1)),axis = 1)
        predlist = np.concatenate((Line1,Line3,index,Line1,predlist.reshape(-1,1),Line4),axis = 1)
        np.savetxt('test_pred.txt',labellist,fmt = '%s')
        np.savetxt('test_label.txt',predlist,fmt = '%s')
        f = os.popen(r'trec_eval.exe -c test_pred.txt -q test_label.txt -m map')
        test_map = float(f.readlines()[0].split('\t')[-1])   
            
        print('Epoch[%d/%d], Loss1: %.4f, Loss2: %.4f, train_acc: %.4f %%, test_acc:%.4f %%, test_map:%.4f' % (epoch+1, num_epochs,  running_loss1,\
                      running_loss2,valence_train_acc*100.0/len(trainset), valence_test_acc*100.0/len(testset),test_map))
    


        if test_map >= a:
            torch.save(model1,'./multi/vio+val_vio.pkl')
            print('save violence model!')
            a = test_map
            
        if valence_test_acc*100.0/len(testset) >= b:
            torch.save(model2,'./multi/vio+val_val.pkl')
            print('save valence model!')
            b = valence_test_acc*100.0/len(testset)
print(a,b)
