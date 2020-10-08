#coding=utf-8
import sys
import os
import numpy as np

def concatenate(filepath1, npy, filepath2,filepath3, writepath):
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
    npyrgb = np.load(filepath2 + npy)
    npyflow = np.load(filepath3 + npy)
    npyaudio = np.hstack(npyaudio).reshape((1,128))
    print(npyaudio.shape)
    print(npyrgb.shape)
    print(npyflow.shape)
    npyconcatenation = np.concatenate((npyaudio,npyrgb,npyflow),axis=1)

    np.save(writepath + npy, npyconcatenation)
    print(npyconcatenation.shape)
    print("save npy "+ npy+" successfully")
    

if __name__ == "__main__":
    filepath1 = '../audio/trainval/non_violence/'
    filepath2 = '../rgb/trainval/non_violence/'
    filepath3 = '../flow/trainval/non_violence/'
    writepath = '../2concatenation/trainval/non_violence/'
    if os.path.exists(writepath) != 1:
        os.makedirs(writepath)
    npylist = os.listdir(filepath1)
    for npy in npylist:
        if npy.endswith('.npy'):
            concatenate(filepath1, npy, filepath2, filepath3,writepath)
