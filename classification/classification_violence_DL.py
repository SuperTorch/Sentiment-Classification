#coding=utf-8
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import svm
from sklearn import metrics
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

class classification2(nn.Module):
    def __init__(self, score_size, num_classes):
        super(classification2,self).__init__()
        self.fc = nn.Linear(score_size, num_classes)
        
        
    def forward(self,x):
        out = self.fc(x)
        #out=F.softmax(out,dim=-1)  #pytorch下交叉熵函数自带softmax,模型中一般不写
        #print(out.shape)
        return out
    
def dataload(dir):
    audioname = dir.split('/')[-1] + '_audio.npy'
    audiopath = os.path.join(dir,audioname)
    
    rgbname = dir.split('/')[-1] + '_rgb.npy'
    rgbpath = os.path.join(dir,rgbname)
    
    flowname = dir.split('/')[-1] + '_flow.npy'
    flowpath = os.path.join(dir,flowname)
    #setpath = dir +'_set.npy'
    labelname = dir.split('/')[-1] + '_violence.npy'
    labelpath = os.path.join(dir,labelname)
    #labelpath = dir + '_violence.npy'
    
    audio = np.load(audiopath)     #如果数据量较大，这种一次性读入所有数据的方式太占用内存
    rgb = np.load(rgbpath)
    flow = np.load(flowpath)    
    label = np.load(labelpath)

    
    return audio, rgb, flow, label

if __name__ == "__main__":
    input_size = 512
    sequence_length = 1
    hidden_size1 = 256
    hidden_size2 = 128
    num_layers = 1
    num_classes = 2
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 512

    train_dir = '../feature/trainval'   #修改地址
    
    train_audio, train_rgb, train_flow, train_label = dataload(train_dir)

    train_audio = torch.Tensor(train_audio)
    train_rgb = torch.Tensor(train_rgb)
    train_flow = torch.Tensor(train_flow)

    train_label = torch.LongTensor(train_label)
    print(train_audio.shape)
    print(train_label.shape)
    
    trainset = Data.TensorDataset(train_audio, train_rgb, train_flow, train_label)
    train_loader = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    
    test_dir = '../feature/test'   #修改地址
    test_audio, test_rgb, test_flow, test_label = dataload(test_dir)

    test_audio = torch.Tensor(test_audio)
    test_rgb = torch.Tensor(test_rgb)
    test_flow = torch.Tensor(test_flow)
    test_label = torch.LongTensor(test_label)
    
    testset = Data.TensorDataset(test_audio, test_rgb, test_flow, test_label)
    test_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    
    model1 = classification(128, hidden_size1, hidden_size2, num_layers, num_classes)   
    model1.cuda()
    
    model2 = classification(input_size, hidden_size1, hidden_size2, num_layers, num_classes)   
    model2.cuda()

    model3 = classification(input_size, hidden_size1, hidden_size2, num_layers, num_classes)   
    model3.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model1.parameters(),lr = learning_rate)
    optimizer2 = torch.optim.Adam(model2.parameters(),lr = learning_rate)
    optimizer3 = torch.optim.Adam(model3.parameters(),lr = learning_rate)
    
    clf = classification2(6,num_classes)
    clf.cuda()
    optimizer = torch.optim.Adam(clf.parameters(),lr = learning_rate)

    
    Line1 = np.ones([test_rgb.shape[0],1],dtype = np.int64)
    Line2 = np.zeros([test_rgb.shape[0],1],dtype = np.int64)
    Line3 = np.array(['Q0']*test_rgb.shape[0]).reshape(test_rgb.shape[0],1)
    Line4 = np.array(['STANDARD']*test_rgb.shape[0]).reshape(test_rgb.shape[0],1)
    index = ['name'+str(i) for i in range(test_rgb.shape[0])]
    index = np.array(index).reshape(test_rgb.shape[0],1)
    
    a = 0

    for epoch in range(num_epochs):
        train_acc = 0
        test_acc = 0
        test_map = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_loss = 0.0

        labellist = []
        labellist = np.array(labellist)
        predlist = []
        predlist = np.array(predlist)
        for i,(X_audio, X_rgb, X_flow, y_train) in enumerate(train_loader):
                      
            X_audio, X_rgb, X_flow, y_train = X_audio.view(-1, 128), X_rgb.view(-1, input_size), X_flow.view(-1, input_size), y_train.view(-1)  #二维数据就可以，没必要用三维
            X_audio, X_rgb, X_flow, y_train = Variable(X_audio).cuda(), Variable(X_rgb).cuda(), Variable(X_flow).cuda(), Variable(y_train).cuda()
            outputs1 = model1(X_audio)
            outputs2 = model2(X_rgb)
            outputs3 = model3(X_flow)
            
            loss1 = criterion(outputs1, y_train)    #y_train要是int64类型，给种类标号就行，不用one-hot，函数可以自己转换
            #print(loss.cpu().detach().data)   #输出loss值，方便观察网络训练情况
            loss2 = criterion(outputs2, y_train)
            loss3 = criterion(outputs3, y_train)
            
            optimizer1.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer1.step()
            
            optimizer2.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer2.step()

            optimizer3.zero_grad()
            loss3.backward(retain_graph=True)
            optimizer3.step()

            running_loss1 += loss1.data   
            running_loss2 += loss2.data   
            running_loss3 += loss3.data   

            output = torch.cat((outputs1,outputs2,outputs3),1)
                        
            out = clf(output)
            loss = criterion(out,y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _,pred = torch.max(out.data,1)
            running_loss += loss.data   
            train_acc += torch.sum(pred == y_train.data)

            #print(y_train.cpu().reshape(batch_size,1))   
            
            #outputs = outputs.cpu().detach().numpy()
            #print(outputs.shape)
            #pred = pred.reshape(batch_size,1)
            
            #print(outputs[:,1])
            
            #out1 = np.concatenate((Line1,Line2,index,y_train.cpu().reshape(batch_size,1)),axis = 1)   #这句报了维度的错，可能和我把二维改三维有关
            #out2 = np.concatenate((Line1,Line3,index,Line1,outputs[:,0].reshape(batch_size,1),Line4),axis = 1)

            #np.savetxt('./train/train_label_'+str(i)+'.txt',out1,fmt = '%s')
            #np.savetxt('./train/train_pred_'+str(i)+'.txt',out2,fmt = '%s')
          
            

    
        for i,(X_audio, X_rgb, X_flow, y_test) in enumerate(test_loader):
            
            X_audio, X_rgb, X_flow = X_audio.view(-1, 128), X_rgb.view(-1, input_size), X_flow.view(-1, input_size)  #二维数据就可以，没必要用三维
            X_audio, X_rgb, X_flow = Variable(X_audio).cuda(), Variable(X_rgb).cuda(), Variable(X_flow).cuda()
            #print(y_test.shape)
            y_test = y_test.view(-1)
            y_test = Variable(y_test).cuda()
            
            outputs1 = model1(X_audio)
            outputs2 = model2(X_rgb)
            outputs3 = model3(X_flow)
            
            output = torch.cat((outputs1,outputs2,outputs3),1)
            out = clf(output)
            _,pred = torch.max(out.data,1)

            test_acc += torch.sum(pred == y_test.data)
            #mAP_test = compute_map(pred,y_test)
            
            outputs = out.cpu().detach().numpy()          
           
            labellist = np.concatenate((labellist,y_test.cpu()),axis = 0)
            labellist = np.array(labellist,dtype = np.int64)

            predlist = np.concatenate((predlist,outputs[:,1]))
            #print(labellist)
            #print(predlist)
        labellist = np.concatenate((Line1,Line2,index,labellist.reshape(-1,1)),axis = 1)
        predlist = np.concatenate((Line1,Line3,index,Line1,predlist.reshape(-1,1),Line4),axis = 1)
        np.savetxt('./test_pred.txt',labellist,fmt = '%s')
        np.savetxt('./test_label.txt',predlist,fmt = '%s')
        f = os.popen(r'trec_eval.exe -c test_pred.txt -q test_label.txt -m map')
        test_map = float(f.readlines()[0].split('\t')[-1])
            
        print('Epoch[%d/%d], Loss: %.4f, train_acc: %.4f %%, test_acc:%.4f %%, test_map:%.4f' % (epoch+1, num_epochs,  running_loss/len(testset), 
                                                                train_acc*100.0/len(trainset), test_acc*100.0/len(testset), test_map))
    
'''        
        if test_acc > a:
            torch.save(model.state_dict(),'./model/classificaton_violence.pkl')
            print('save model!')
            a = test_acc
 '''