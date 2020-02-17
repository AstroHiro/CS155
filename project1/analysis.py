#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:57:48 2020

@author: hiroyasu
"""


import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=10)
#clf.fit(Xall,Yall)
#importances = clf.feature_importances_
#np.save('data/params/importances10RF.npy',importances)


def preprocessX(X):
    Xm = np.mean(X,axis=0)
    Xsig = np.std(X,axis=0)
    Xnor = (X-Xm)/Xsig
    return Xnor,Xm,Xsig

def LoadDict(filename):
    pkl_file = open(filename,'rb')
    varout = pickle.load(pkl_file)
    pkl_file.close()
    return varout

df_train = pd.read_csv('data/train.csv', index_col=0)
df_train = df_train.fillna(method='ffill')
df_train = df_train.fillna(method='bfill')
df_test = pd.read_csv('data/test.csv', index_col=0)
df_test = df_test.fillna(method='ffill')
df_test = df_test.fillna(method='bfill')
n = (df_train.values).shape[1]-1
Xall = df_train.values[:,0:n]
Yall = df_train.values[:,n]
Nall = Yall.size
idx = np.random.permutation(Nall)
Xall = Xall[idx,:]
Yall = Yall[idx]
Xtest = df_test.values[:,0:n]
Ntest = Xtest.shape[0]
Xstack = np.vstack((Xtest,Xall))
Xstack,Xm,Xsig = preprocessX(Xstack)
Xall = Xstack[Ntest:,:]
Xtest = Xstack[0:Ntest,:]

xbar = np.linspace(1,26,26)
xbar2 = np.linspace(0,27,100)
thres = np.ones(100)*0.025
thres2 = np.ones(100)*0.042
importances50 = np.load('data/params/importances50.npy')
importancesRF = np.load('data/params/importances10RF.npy')
importancesRF = np.load('data/params/importances100RF.npy')
importances100 = np.load('data/params/importances100.npy')
importances1000 = np.load('data/params/importances1000.npy')
importances1000 = np.load('data/params/importances1000.npy')
plt.figure()
plt.plot(xbar2,thres)
plt.bar(xbar,importances50)
plt.xlabel('Features',fontsize=18)
plt.ylabel('Importance',fontsize=18)
plt.legend(['Threshold value 0.025'],loc='best',fontsize=15)
plt.savefig('data/plots/importances50.png')
plt.show()
plt.figure()
plt.bar(xbar,importancesRF)
plt.plot(xbar2,thres2)
plt.xlabel('Features',fontsize=18)
plt.ylabel('Importance',fontsize=18)
plt.legend(['Threshold value 0.042'],loc='best',fontsize=14)
plt.savefig('data/plots/importancesRF.png')

lossdic = LoadDict('data/loss_list.pkl')
losstestdic = LoadDict('data/loss_list_test.pkl')
epochs = np.linspace(0,9,10)
plt.figure()
for i in range(11):
    plt.plot(epochs,losstestdic[i]) 
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Test error',fontsize=18)
plt.legend(['Ensemble 0','Ensemble 1','Ensemble 2','Ensemble 3','Ensemble 4','Ensemble 5','Ensemble 6','Ensemble 7','Ensemble 8','Ensemble 9','Ensemble 10'],loc='best',fontsize=12)
plt.xlim(0,15)
plt.savefig('data/plots/ensembleTe.png')
plt.show

plt.figure()
for i in range(11):
    plt.plot(epochs,lossdic[i]) 
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Training error',fontsize=18)
plt.legend(['Ensemble 0','Ensemble 1','Ensemble 2','Ensemble 3','Ensemble 4','Ensemble 5','Ensemble 6','Ensemble 7','Ensemble 8','Ensemble 9','Ensemble 10'],loc='best',fontsize=12)
plt.xlim(0,15)
plt.savefig('data/plots/ensembleTr.png')
plt.show