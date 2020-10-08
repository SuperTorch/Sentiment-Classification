#coding=utf-8
import numpy as np
import os
#import torch
import pandas as pd


def gettxt(dir):
    dataset = pd.read_table(dir)
    
    return dataset

def getlabel():
    dir = ['ACCEDEviolence.txt','ACCEDEaffect.txt','MEDIAEVALviolence.txt','MEDIAEVALaffect.txt']
    label1, label2, label3, label4  = gettxt(dir[0]), gettxt(dir[1]), gettxt(dir[2]), gettxt(dir[3])
    label2 = label2.drop(label2.columns[0:2], axis = 1)
    label4 = label4.drop(label4.columns[0:2], axis = 1)
    label1 = pd.concat([label1, label2], axis=1)
    label2 = pd.concat([label3, label4], axis=1)
    label = pd.concat([label1, label2], axis=0)
    label = label.drop(label.columns[0], axis=1)

    return label
    

def getData(dir):
    dir1 = 'violence'
    dir2 = 'non_violence'
    dir1 = os.path.join(dir,dir1)
    dir2 = os.path.join(dir,dir2)
    datalist = []
    labellist_1 = []
    labellist_2 = []
    labellist_3 = []
    labeldf = getlabel()
    file_list = os.listdir(dir1)
    
    for file in file_list:
        file_path=os.path.join(dir1,file)
        
        name = file.split('.')[0].split('_')[0]
        labelname = labeldf[(labeldf['name']==name)]


        label1 = int(labelname['violence'].values[0])
        label2 = int(labelname['valenceClass'].values[0]+1)
        label3 = int(labelname['arousalClass'].values[0]+1)
        
            
        data = np.load(file_path)
        datalist.append(data)
        labellist_1.append(label1)
        labellist_2.append(label2)
        labellist_3.append(label3)
                
        print('append ' + file + ' successfully.')
        
    file_list = os.listdir(dir2)
    for file in file_list:
        file_path=os.path.join(dir2,file)
        
        
        name = file.split('.')[0].split('_')[0]
        labelname = labeldf[(labeldf['name']==name)]

        label1 = int(labelname['violence'].values[0])
        label2 = int(labelname['valenceClass'].values[0]+1)
        label3 = int(labelname['arousalClass'].values[0]+1)
        
            
        data = np.load(file_path)
        datalist.append(data)
        labellist_1.append(label1)
        labellist_2.append(label2)
        labellist_3.append(label3)

        
        print('append ' + file + ' successfully.')
        


    datalist = np.array(datalist)
    labellist_1 = np.array(labellist_1, ndmin = 2).T
    labellist_2 = np.array(labellist_2, ndmin = 2).T
    labellist_3 = np.array(labellist_3, ndmin = 2).T

    


    savename = dir.split('/')[-1] + '_data'
    savelabel1 = dir.split('/')[-1] + '_violence'
    savelabel2 = dir.split('/')[-1] + '_valenceClass'
    savelabel3 = dir.split('/')[-1] + '_arousalClass'

    savepath = os.path.join(dir,savename)
    labelpath = os.path.join(dir, savelabel1)
    np.save(savepath,datalist)
    print('save '+ savename +' successfully.')
    
    np.save(labelpath,labellist_1)
    print('save '+ savelabel1 +' successfully.')
    
    labelpath = os.path.join(dir, savelabel2)
    np.save(labelpath,labellist_2)
    print('save '+ savelabel2 +' successfully.')
    
    labelpath = os.path.join(dir, savelabel3)
    np.save(labelpath,labellist_3)
    print('save '+ savelabel3 +' successfully.')

    print(datalist.shape)
    print(labellist_1.shape)
    print(labellist_2.shape)
    print(labellist_3.shape)
    
    return (datalist,labellist_1,labellist_2,labellist_3)
    
    
    

if __name__ == "__main__":
    dir = "../7concatenation/test"
    dataset = getData(dir)