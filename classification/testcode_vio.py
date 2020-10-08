# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:16:53 2020

@author: user
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from test import *
import numpy as np
import os

class classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(classification,self).__init__()
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

def dataload(dir):
    setname = dir.split('/')[-1] + '_data.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname = dir.split('/')[-1] + '_violence.npy'
    labelpath = os.path.join(dir,labelname)
    #labelpath = dir + '_violence.npy'
    
    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label = np.load(labelpath)

    
    return data, label

if __name__ == "__main__":
    input_size = 640
    sequence_length = 1
    hidden_size = 320
    num_layers = 1
    num_classes = 3
    learning_rate = 0.001
    num_epochs = 200
    batch_size = 64

    test_dir = '../7concatenation/test'   #修改地址
    test_dataset, test_label = dataload(test_dir)

    test_dataset = torch.Tensor(test_dataset)
    test_label = torch.LongTensor(test_label)
    
    testset = Data.TensorDataset(test_dataset, test_label)
    test_loader = Data.DataLoader(dataset=testset, batch_size=512, shuffle=True)
    
    model = classification(input_size, hidden_size, num_layers, num_classes)
    model.cuda()
    model = torch.load('./model/violence2.pkl')
    
    Line1 = np.ones([test_dataset.shape[0],1],dtype = np.int64)
    Line2 = np.zeros([test_dataset.shape[0],1],dtype = np.int64)
    Line3 = np.array(['Q0']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    Line4 = np.array(['STANDARD']*test_dataset.shape[0]).reshape(test_dataset.shape[0],1)
    index = ['name'+str(i) for i in range(test_dataset.shape[0])]
    index = np.array(index).reshape(test_dataset.shape[0],1)

    
    
    test_map = 0.0
    labellist = []
    labellist = np.array(labellist)
    predlist = []
    predlist = np.array(predlist)
    for i,(X_test, y_test) in enumerate(test_loader):
            
        X_test = X_test.view(-1, input_size)
            #print(y_test.shape)
        y_test = y_test.view(-1)
        X_test, y_test = Variable(X_test).cuda() , Variable(y_test).cuda()
            
        outputs = model(X_test)
        _,pred = torch.max(outputs.data,1)

        #test_acc += torch.sum(pred == y_test.data)
            
        outputs = outputs.cpu().detach().numpy()
            
        labellist = np.concatenate((labellist,y_test.cpu()),axis = 0)
        labellist = np.array(labellist,dtype = np.int64)

        predlist = np.concatenate((predlist,outputs[:,1]))
            #print(labellist)
            #print(predlist)
    labellist = np.concatenate((Line1,Line2,index,labellist.reshape(-1,1)),axis = 1)
    predlist = np.concatenate((Line1,Line3,index,Line1,predlist.reshape(-1,1),Line4),axis = 1)
    np.savetxt('test_pred.txt',labellist,fmt = '%s')
    np.savetxt('test_label.txt',predlist,fmt = '%s')
    f = os.popen(r'trec_eval.exe -c test_pred.txt -q test_label.txt -m map')
    test_map = float(f.readlines()[0].split('\t')[-1])
    
    print("——————————测试完毕———————————")
    print("测试集 violence 的 mAP 值为：%.2f %%" % (test_map*100))
