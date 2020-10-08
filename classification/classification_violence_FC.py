#coding=utf-8
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import os

#分类器模型结构
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

        return out

#加载数据集
def dataload(dir):
    setname = dir.split('/')[-1] + '_set.npy'
    setpath = os.path.join(dir,setname)
    
    labelname = dir.split('/')[-1] + '_violence.npy'
    labelpath = os.path.join(dir,labelname)
    
    data = np.load(setpath)    
    label = np.load(labelpath)

    
    return data, label


if __name__ == "__main__":
    input_size = 1152
    sequence_length = 1
    hidden_size1 = 256
    hidden_size2 = 128
    num_layers = 1
    num_classes = 2
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 512

    train_dir = '../1concatenation/trainval'   
    
    train_dataset, train_label = dataload(train_dir)

    train_dataset = torch.Tensor(train_dataset)
    train_label = torch.LongTensor(train_label)
    print(train_dataset.shape)
    print(train_label.shape)
    
    trainset = Data.TensorDataset(train_dataset, train_label)
    train_loader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)    #加载训练集

    
    test_dir = '../1concatenation/test'   #修改地址
    test_dataset, test_label = dataload(test_dir)

    test_dataset = torch.Tensor(test_dataset)
    test_label = torch.LongTensor(test_label)
    
    testset = Data.TensorDataset(test_dataset, test_label)
    test_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)    #加载测试集

    #定义模型、优化器和损失函数
    model = classification(input_size, hidden_size1, hidden_size2, num_layers, num_classes)   
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    
    #计算mAP所用到的参数
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
                      
            X_train,y_train = X_train.view(-1,  input_size),y_train.view(-1)  
            X_train,y_train = Variable(X_train).cuda(),Variable(y_train).cuda()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _,pred = torch.max(outputs.data,1)
            running_loss += loss.data   
            train_acc += torch.sum(pred == y_train.data)            

    
        for i,(X_test, y_test) in enumerate(test_loader):
            
            X_test = X_test.view(-1, input_size)
            y_test = y_test.view(-1)
            X_test, y_test = Variable(X_test).cuda() , Variable(y_test).cuda()
            
            outputs = model(X_test)
            _,pred = torch.max(outputs.data,1)

            test_acc += torch.sum(pred == y_test.data)
            
            outputs = outputs.cpu().detach().numpy()
            
            labellist = np.concatenate((labellist,y_test.cpu()),axis = 0)
            labellist = np.array(labellist,dtype = np.int64)

            predlist = np.concatenate((predlist,outputs[:,1]))
        labellist = np.concatenate((Line1,Line2,index,labellist.reshape(-1,1)),axis = 1)
        predlist = np.concatenate((Line1,Line3,index,Line1,predlist.reshape(-1,1),Line4),axis = 1)
        np.savetxt('test_pred.txt',labellist,fmt = '%s')
        np.savetxt('test_label.txt',predlist,fmt = '%s')
        #调用外部程序计算mAP值
        f = os.popen(r'trec_eval.exe -c test_pred.txt -q test_label.txt -m map')
        test_map = float(f.readlines()[0].split('\t')[-1])
            
        print('Epoch[%d/%d], Loss: %.4f, train_acc: %.4f %%, test_acc:%.4f %%,test_map:%.4f' % (epoch+1, num_epochs,  running_loss/len(testset), 
                                                                train_acc*100.0/len(trainset), test_acc*100.0/len(testset), test_map))
    
        #保存最好的模型
        if test_acc > a:
            torch.save(model.state_dict(),'./model/classificaton_violence.pkl')
            print('save model!')
            a = test_acc
