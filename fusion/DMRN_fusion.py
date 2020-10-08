# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:13:36 2020

@author: user
"""
import numpy as np
import os 
import torch
import torch.nn as nn


tanh = nn.Tanh()

def U_v(video,hidden_dim):
    model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    model = model.cuda()
    return model(video)

def U_a(audio,hidden_dim):
    model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    model = model.cuda()

    return model(audio)
    
def TBMRF_block(audio, video, hidden_dim, nb_block):

    for i in range(nb_block):
        video_residual = video
        v = U_v(video, hidden_dim)
        audio_residual = audio
        a = U_a(audio, hidden_dim)
        merged = torch.mul(v + a, 0.5) 

        a_trans = audio_residual
        v_trans = video_residual


        video = tanh(a_trans + merged)
        audio = tanh(v_trans + merged)


    fusion = torch.mul(video + audio, 0.5)
    return fusion

def fusion(filepath1, npy, filepath2,filepath3, writepath):
    audiodir = os.path.join(filepath1,npy)
    audioname1 = npy.split('_')[0] + '_v1_a0.npy'
    audioname2 = npy.split('_')[0] + '_v1_a1.npy'
    audioname3 = npy.split('_')[0] + '_v0_a1.npy'
    if os.path.exists(audiodir):
        npyaudio = np.load(filepath1 + npy)
    elif os.path.exists(os.path.join(filepath1,audioname1)):
        npyaudio = np.load(filepath1 + audioname1)
    elif os.path.exists(os.path.join(filepath1,audioname2)):
        npyaudio = np.load(filepath1 + audioname2)
    elif os.path.exists(os.path.join(filepath1,audioname3)):
        npyaudio = np.load(filepath1 + audioname3)
        
    rgb = torch.Tensor(np.load(filepath2 + npy)).cuda()
    flow = torch.Tensor(np.load(filepath3 + npy)).cuda()
    audio = torch.Tensor(np.hstack(npyaudio).reshape((1,128))).cuda()
    
    print(audio.shape)
    print(rgb.shape)
    print(flow.shape)
    fusion1 = TBMRF_block(rgb, flow, hidden_dim=512, nb_block=1)
    fusion2 = TBMRF_block(audio, audio, hidden_dim=128, nb_block=1)
    
    fusion = torch.cat((fusion1, fusion2),1)

    np.save(writepath + npy, fusion.cpu().detach().numpy())
    print(fusion.shape)
    print("save npy "+ npy+" successfully")


if __name__ == "__main__":

    filepath1 = '../audio/test/non_violence/'
    filepath2 = '../rgb/test/non_violence/'
    filepath3 = '../flow/test/non_violence/'
    writepath = '../7concatenation/test/non_violence/'
    if os.path.exists(writepath) != 1:
        os.makedirs(writepath)
    npylist = os.listdir(filepath2)
    for npy in npylist:
        if npy.endswith('.npy'):
            fusion(filepath1, npy, filepath2, filepath3,writepath)
