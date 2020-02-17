#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:59:16 2020

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
    nn.Linear(n_hidden,n_hidden),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.Linear(n_hidden,n_hidden),
    nn.ReLU(),
    nn.Dropout(dropout_prob),
    nn.Linear(n_hidden,n_hidden),
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
        for epoch in range(n_epochs):
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
Nval = 12380
n_epochs = 100
model = AdaBoostClassifier(n_estimators=1000).fit(Xall,Yall)
loss = model.score(Xall,Yall)
print(loss)

df_test = pd.read_csv('data/test.csv', index_col=0)
df_test = df_test.fillna(method='ffill')
df_test = df_test.fillna(method='bfill')
Xtest = df_test.values[:,0:n]
Xtest,Xm,Xsig = preprocessX(Xtest)
Xtest = Xtest[:,idx]
#pca = PCA(n_components=Ninput)
#pca.fit(Xtest)
Xtest = pca.transform(Xtest)
df_test['Predicted'] = model.predict_proba(Xtest)[:,1]
df_test[['Predicted']].to_csv('data/submission.csv')