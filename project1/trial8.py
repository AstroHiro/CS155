#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:13:45 2020

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

def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def preprocessX(X):
    Xm = np.mean(X,axis=0)
    Xsig = np.std(X,axis=0)
    Xnor = (X-Xm)/Xsig
    return Xnor,Xm,Xsig

def get_model(Xtrain,Ytrain):
    model = Sequential()
    model.add(Conv1D(filters=64,kernel_size=2,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(Xtrain,Ytrain,epochs=3,batch_size=64)
    return model

df_train = pd.read_csv('data/train.csv', index_col=0)
df_train = df_train.fillna(method='ffill')
df_train = df_train.fillna(method='bfill')

n = (df_train.values).shape[1]-1
Xall = df_train.values[:,0:n]
Yall = df_train.values[:,n]
Xall,Xm,Xsig = preprocessX(Xall)

#model = AdaBoostClassifier(n_estimators=100).fit(Xall,Yall)
#importances = model.feature_importances_
#np.save('data/params/importances100.npy',importances)

importances100 = np.load('data/params/importances100.npy')
idx = importances100 > 0.02
Xall = Xall[:,idx]
Ninput = 9


#Ninput = 9
pca = PCA(n_components=Ninput)
pca.fit(Xall)
Xall = pca.transform(Xall)
lam = pca.explained_variance_ratio_
print(lam)



Nall = Yall.size
time_steps = 10
Nval = 12380
n_epochs = 100

Xval = Xall[0:Nval,:]
Yval = Yall[0:Nval]
Xtrain = Xall[Nval:,:]
Ytrain = Yall[Nval:]
Nsamples = int(Ytrain.size/time_steps)
Xtrain = Xtrain.reshape((Nsamples,time_steps,Ninput))
model = get_model(Xtrain,Ytrain)
print(Xtrain)