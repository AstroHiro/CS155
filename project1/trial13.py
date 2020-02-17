#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:17:44 2020

@author: hiroyasu
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
import pickle

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
    nn.Linear(n_hidden,1),
    )
    return model

class MyEnsemble(nn.Module):
    def __init__(self,model,n,n_hidden,dropout_prob):
        super(MyEnsemble, self).__init__()
        self.model = model
        self.new_model = get_model(n,n_hidden,dropout_prob)
        self.out = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.classifier = nn.Linear(2,1)
        
    def forward(self,x):
        x1 = self.model(x)
        x2 = self.new_model(x)
        x = torch.cat((x1,x2),dim=1)
        #x = self.out(x)
        #x = x1+x2
        x = self.classifier(x)
        return x
    
def train_model(model,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize,criterion):
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    N = Ytrain.size/BatchSize
    N = int(N)
    idx = np.random.permutation(Ytrain.size)
    Xval = Np2Var(Xval)
    Yval = Np2Var((np.array([Yval])).T)
    model.train()
    Xtrain = Xtrain[idx,:]
    Ytrain = Ytrain[idx]
    loss_list = np.zeros(n_epochs)
    loss_test_list = np.zeros(n_epochs)
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
        loss_list[epoch] = loss.data.numpy()
        loss_test_list[epoch] = loss_test.data.numpy()
        print(loss_test.data.numpy())
    return model,loss_list,loss_test_list

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SaveDict(filename,var):
    output = open(filename,'wb')
    pickle.dump(var,output)
    output.close()
    pass
    
def LoadDict(filename):
    pkl_file = open(filename,'rb')
    varout = pickle.load(pkl_file)
    pkl_file.close()
    return varout


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
Nval = 22380
n_epochs = 10
idxv = np.random.choice(Nall,Nval,replace=False)
Xtrain = np.delete(Xall,idxv,0)
Xval = Xall[idxv,:]
Ytrain = np.delete(Yall,idxv)
Yval = Yall[idxv]

BatchSize = 1000
Nhidden = 416
Dprob = 0.2

df_test = pd.read_csv('data/test.csv', index_col=0)
df_test = df_test.fillna(method='ffill')
df_test = df_test.fillna(method='bfill')
Xtest = df_test.values[:,0:n]
Xtest,Xm,Xsig = preprocessX(Xtest)
Xtest = Xtest[:,idx]
#pca = PCA(n_components=Ninput)
#pca.fit(Xtest)
model = get_model(Ninput,Nhidden,Dprob)
criterion = nn.BCEWithLogitsLoss()
lossdic = {}
losstestdic = {}
model,loss_list,loss_test_list  = train_model(model,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize,criterion)
lossdic[0] = loss_list
losstestdic[0] = loss_test_list
model.eval()
for i in range(10):
    filename = 'data/trained_nets/modelEN'+str(i+1)+'.pt'
    criterion = nn.BCEWithLogitsLoss()
    model = MyEnsemble(model,Ninput,Nhidden,Dprob)
    model,loss_list,loss_test_list = train_model(model,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize,criterion)
    lossdic[i+1] = loss_list
    losstestdic[i+1] = loss_test_list
    model.eval()
    torch.save(model.state_dict(),filename)
Xtest = pca.transform(Xtest)
df_test['Predicted'] = sigmoid(model(Np2Var(Xtest)).data.numpy())
df_test[['Predicted']].to_csv('data/submission.csv')