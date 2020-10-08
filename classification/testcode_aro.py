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

import numpy as np
import os

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
    setname = dir.split('/')[-1] + '_set.npy'
    setpath = os.path.join(dir,setname)
    #setpath = dir +'_set.npy'
    labelname = dir.split('/')[-1] + '_arousalClass.npy'
    labelpath = os.path.join(dir,labelname)
    #labelpath = dir + '_violence.npy'
    
    data = np.load(setpath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    label = np.load(labelpath)

    
    return data, label

if __name__ == "__main__":
    input_size = 1152
    sequence_length = 1
    hidden_size1 = 2048
    hidden_size2 = 1024
    num_layers = 1
    num_classes = 3
    learning_rate = 0.0001
    num_epochs = 200
    batch_size = 512

    test_dir = '../1concatenation/test'   #修改地址
    test_dataset, test_label = dataload(test_dir)

    test_dataset = torch.Tensor(test_dataset)
    test_label = torch.LongTensor(test_label)
    
    testset = Data.TensorDataset(test_dataset, test_label)
    test_loader = Data.DataLoader(dataset=testset, batch_size=512, shuffle=False)
    
    model = classification2(input_size, hidden_size1, hidden_size2, num_layers, num_classes)
    model.cuda()
    model = torch.load('./multi/vio+val+aro_aro.pkl')
    
    test_acc = 0.0
    for i,(X_test, y_test) in enumerate(test_loader):
            
            X_test = X_test.view(-1, input_size)
            #print(y_test.shape)
            y_test = y_test.view(-1)
            X_test, y_test = Variable(X_test).cuda() , Variable(y_test).cuda()
            
            outputs = model(X_test)
            _,pred = torch.max(outputs.data,1)

            test_acc += torch.sum(pred == y_test.data)
    print("————————————————测试完毕———————————————————")
    print("测试集 arousalClass 的 acc 值为：%.2f %%" % (test_acc*100.0/len(test_dataset)))       
