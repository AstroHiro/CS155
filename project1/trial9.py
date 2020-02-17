#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:53:08 2020

@author: hiroyasu
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier

def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def preprocessX(X):
    Xm = np.mean(X,axis=0)
    Xsig = np.std(X,axis=0)
    Xnor = (X-Xm)/Xsig
    return Xnor,Xm,Xsig

def get_model(n,n_hidden,dropout_prob):
    model = nn.Sequential(
    nn.Linear(n,n_hidden),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.Linear(n_hidden,1)
    )
    return model

def train_model(model,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    N = Ytrain.size/BatchSize
    N = int(N)
    idx = np.random.permutation(Ytrain.size)
    Xval = Np2Var(Xval)
    Yval = Np2Var((np.array([Yval])).T)
    model.train()
    Xtrain = Xtrain[idx,:]
    Ytrain = Ytrain[idx]
    for epoch in range(n_epochs):
        print('epoch = ',epoch)
        for i in range(N):
            X = Xtrain[i*BatchSize:(i+1)*BatchSize,:]
            Y = Ytrain[i*BatchSize:(i+1)*BatchSize]
            Y = np.array([Y]).T
            X = Np2Var(X) 
            Y = Np2Var(Y)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output,Y)
            loss.backward()
            optimizer.step()
        Ypred = model(Xval)
        loss_test = criterion(Ypred,Yval)
        print(loss_test)
    return model,loss_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


df_train = pd.read_csv('data/train.csv', index_col=0)
#df_train = df_train.dropna()
df_train = df_train.fillna(method='ffill')
df_train = df_train.fillna(method='bfill')
df_test = pd.read_csv('data/test.csv', index_col=0)
#df_test= df_test.dropna()
df_test = df_test.fillna(method='ffill')
df_test = df_test.fillna(method='bfill')
n = (df_train.values).shape[1]-1
Xall = df_train.values[:,0:n]
Yall = df_train.values[:,n]
Nall = Yall.size
Xtest = df_test.values[:,0:n]
Ntest = Xtest.shape[0]
Xstack = np.vstack((Xtest,Xall))
Xstack,Xm,Xsig = preprocessX(Xstack)
Xall = Xstack[Ntest:,:]
Xtest = Xstack[0:Ntest,:]

#model = AdaBoostClassifier(n_estimators=1000).fit(Xall,Yall)
#importances = model.feature_importances_
#np.save('data/params/importances1000.npy',importances)

importances100 = np.load('data/params/importances100.npy')
idx = importances100 > 0.02
Xall = Xall[:,idx]
Xtest = Xtest[:,idx]
Xstack = Xstack[:,idx]
Ninput = 9
#Ninput = 9
pca = PCA(n_components=Ninput)
pca.fit(Xall)
Xall = pca.transform(Xall)
Xtest = pca.transform(Xtest)
lam = pca.explained_variance_ratio_
print(lam)

Nall = Yall.size
Nval = 12380
n_epochs = 100
idxv = np.random.choice(Nall,Nval,replace=False)
Xtrain = np.delete(Xall,idxv,0)
Xval = Xall[idxv,:]
Ytrain = np.delete(Yall,idxv)
Yval = Yall[idxv]

Nhidden = 416
Dprob = 0.1
BatchSize = 1000
for i in range(10):
    filename = 'data/trained_nets/model'+str(i+1)+'.pt'
    model = get_model(Ninput,Nhidden,Dprob)
    model,loss_test = train_model(model,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize)
    model.eval()
    torch.save(model.state_dict(),filename)
    
df_test['Predicted'] = sigmoid(model(Np2Var(Xtest)).data.numpy())
df_test[['Predicted']].to_csv('data/submission.csv')
np.save('data/trained_nets/Ninput.npy',Ninput)
np.save('data/trained_nets/Nhidden.npy',Nhidden)
np.save('data/trained_nets/Dprob.npy',Dprob)