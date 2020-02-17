#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:57:29 2020

@author: hiroyasu
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


class MyEnsemble(nn.Module):
    def __init__(self,models):
        super(MyEnsemble, self).__init__()
        self.N = len(models)
        self.models = models
        self.out1 = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.classifier = nn.Linear(self.N,1)
        
    def forward(self,x):
        xi = self.sig(self.models[0](x))
        for i in range(self.N-1):
            xi = torch.cat((xi,self.sig(self.models[i+1](x))),dim=1)
        xi = self.out1(xi)
        y = self.classifier(xi)
        return y

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
    nn.Linear(n_hidden,1),
    nn.Sigmoid()
    )
    return model

def preprocessX(X):
    Xm = np.mean(X,axis=0)
    Xsig = np.std(X,axis=0)
    Xnor = (X-Xm)/Xsig
    XC = np.amax(np.abs(Xnor),axis=0)
    Xnor = Xnor/XC
    return Xnor,Xm,Xsig,XC

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

def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Ninput = np.load('data/trained_nets/Ninput.npy')
Nhidden = np.load('data/trained_nets/Nhidden.npy')
Dprob = np.load('data/trained_nets/Dprob.npy')


models = {}
for i in range(5):
    model = get_model(9,100,0.1)
    filename = 'data/trained_nets/model'+str(i+1)+'.pt'
    model.load_state_dict(torch.load(filename))
    models[i] = model
'''
model1 = get_model(9,100,0.1)
model1.load_state_dict(torch.load('data/trained_nets/model1.pt'))
model2 = get_model(9,100,0.1)
model2.load_state_dict(torch.load('data/trained_nets/model2.pt'))
model3 = get_model(9,100,0.1)
model3.load_state_dict(torch.load('data/trained_nets/model3.pt'))
model4 = get_model(9,100,0.1)
model4.load_state_dict(torch.load('data/trained_nets/model4.pt'))
model5 = get_model(9,100,0.1)
model5.load_state_dict(torch.load('data/trained_nets/model5.pt'))
'''


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
Xtest = df_test.values[:,0:n]
Ntest = Xtest.shape[0]
Xstack = np.vstack((Xtest,Xall))
Xstack,Xm,Xsig,XC = preprocessX(Xstack)
Xall = Xstack[Ntest:,:]
Xtest = Xstack[0:Ntest,:]


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
n_epochs = 10
idxv = np.random.choice(Nall,Nval,replace=False)
Xtrain = np.delete(Xall,idxv,0)
Xval = Xall[idxv,:]
Ytrain = np.delete(Yall,idxv)
Yval = Yall[idxv]

net = MyEnsemble(models)
BatchSize = 1000
net,loss_test = train_model(net,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize)
net.eval()

torch.save(net.state_dict(),'data/trained_nets/model_final.pt')
df_test['Predicted'] = sigmoid(net(Np2Var(Xtest)).data.numpy())
df_test[['Predicted']].to_csv('data/submission.csv')