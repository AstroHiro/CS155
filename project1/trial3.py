#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:37:48 2020

@author: hiroyasu
"""

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
from sklearn.decomposition import PCA


from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

df_train = pd.read_csv('data/train.csv', index_col=0)
df_test = pd.read_csv('data/test.csv', index_col=0)
'''
x_cols = {
    'series_id': np.uint32,
    'last_price': np.float32,
    'mid': np.float32,
    'opened_position_qty': np.float32,
    'closed_position_qty': np.float32,
    'transacted_qty': np.float32,
    'bid1': np.float32,
    'bid2': np.float32,
    'bid3': np.float32,
    'bid4': np.float32,
    'bid5': np.float32,
    'ask1': np.float32,
    'ask2': np.float32,
    'ask3': np.float32,
    'ask4': np.float32,
    'ask5': np.float32,
    'bid1vol': np.float32,
    'bid2vol': np.float32,
    'bid3vol': np.float32,
    'bid4vol': np.float32,
    'bid5vol': np.float32,
    'ask1vol': np.float32,
    'ask2vol': np.float32,
    'ask3vol': np.float32,
    'ask4vol': np.float32,
    'ask5vol': np.float32
}
'''
df_train = df_train.fillna(method='ffill')
df_train = df_train.fillna(method='bfill')
df_test = df_test.fillna(method='ffill')
df_test = df_test.fillna(method='bfill')
n = (df_train.values).shape[1]-1
nt = (df_test.values).shape[1]-1
Xall = df_train.values[:,0:n]
Yall = df_train.values[:,n]
Xtest = df_test.values
n_input = n # rnn input size
n_hidden = 128 # number of rnn hidden units
n_output = 1 # rnn input size
n_layers = 2 # rnn number of layers

# truncate and pad input sequences

class RNN(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,n_layers):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
                input_size=n_input,
                hidden_size=n_hidden, # rnn hidden unit
                num_layers=n_layers, # number of rnn layer
                batch_first=True, # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
                )
        self.out1 = nn.Linear(n_hidden,n_output)
        #self.out2 = nn.Sigmoid()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
    def forward(self,x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out,_ = self.rnn(x)
        out = self.out1(r_out)
        #out = self.out2(out)
        return out
    
def Np2Var(X):
    X = X.astype(np.float32)
    X = torch.from_numpy(X)
    return X

def RandomizeTrainData(Xtrain,Ytrain,time_step):
    # Note that training data = original data - test data
    N = Ytrain.size
    Ntime = N/time_step
    Ntime = int(Ntime)
    XtrainR = np.zeros((Ntime,time_step,6))
    YtrainR = np.zeros((Ntime,time_step,1))
    for j in range(Ntime):
        XtrainR[j,:,:] = Xtrain[(time_step)*j:(time_step)*(j+1),:]
        YtrainR[j,:,:] = np.array([Ytrain[(time_step)*j:(time_step)*(j+1)]]).T
    idx = np.random.permutation(Ntime)
    XtrainR = XtrainR[idx,:,:]
    YtrainR = YtrainR[idx,:,:]
    return XtrainR,YtrainR

def preprocessX(X):
    n = X.shape[1]
    XC = np.amax(X,axis=0)
    for i in range(n):
        X[:,i] = X[:,i]/XC[i]
    return X,XC

def get_model(dropout_prob):
    model = nn.Sequential(
    nn.Conv1d(1, 16, 3), # output dim = (26-2) \times 16
    nn.BatchNorm1d(16),
    nn.ReLU(),
    nn.MaxPool1d(2),# output dim = 12 \times  12
    
    nn.Flatten(),
    nn.Linear(2*16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
    )
    return model

def train_model(model,Xtrain,Ytrain,Xval,Yval,n_epochs,BatchSize,ValSize):
    # For a multi-class classification problem
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(model.parameters())
    # Train the model for 1 epoch, iterating on the data in batches
    N = Ytrain.size/BatchSize
    N = int(N)
    Xval = Np2Var(Xval)
    Xval = Xval.view(ValSize+1,1,-1)
    Yval = Np2Var(np.array([Yval])).T
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/10:', end='')
        # train
        model.train()
        for epoch in range(n_epochs):
            for i in range(N):
                X = Xtrain[BatchSize*i:BatchSize*(i+1),:]
                Y = Ytrain[BatchSize*i:BatchSize*(i+1)]
                Y = np.array([Y]).T
                X = Np2Var(X) 
                Y = Np2Var(Y)
                X = X.view(BatchSize,1,-1)
                #Y = Y.view(BatchSize,1,-1)
                optimizer.zero_grad()
                # forward pass
                output = model(X)
                # calculate categorical cross entropy loss
                loss = criterion(output,Y)
                loss.backward() # backpropagation, compute gradients
                optimizer.step() # apply gradients
            Ypred = model(Xval)
            loss_test = criterion(Ypred,Yval)
            print(loss_test)
        return loss_test





if __name__ == "__main__":
    Xall,XC = preprocessX(Xall)
    Xtest,XCt = preprocessX(Xtest)
    pca = PCA(n_components=6)
    pca.fit(Xall)
    comp = pca.components_
    Xall = pca.transform(Xall)
    lam = pca.explained_variance_ratio_
    print(lam)
    Xall,XC = preprocessX(Xall)
    time_step = 1 # rnn time step
    model = get_model(0.1)
    LR = 0.2 # learning rate
    N = np.size(Xall,0)
    BatchSize = np.floor(N/100)
    BatchSize = BatchSize.astype(np.int32)
    ValSize = 1000-1
    Xval = Xall[0:ValSize+1,:]
    Yval = Yall[0:ValSize+1]
    Xval = Np2Var(Xval)
    Yval = Np2Var(Yval)
    Xval = Xval.view(ValSize+1,1,-1)
    Yval = Yval.view(ValSize+1,1,-1)
    Xtrain0 = Xall[ValSize+1:,:]
    Ytrain0 = Yall[ValSize+1:]
    #optimizer = torch.optim.Adam(model.parameters(),lr=LR) # optimize all cnn parameters
    optimizer = torch.optim.SGD(model.parameters(),lr=LR) # optimize all cnn parameters
    optimizer = torch.optim.RMSprop(model.parameters())
    loss_func = nn.BCEWithLogitsLoss()
    #loss_func = nn.MSELoss()
    h_state = None # for initial hidden state
    model = get_model(0.1)
    #loss = train_model(model,Xtrain0,Ytrain0,Xval,Yval,10,BatchSize)

    Nitrs = 1000000
    e_list = []
    t_loss_list = []
    loss_list = []
    Nepochs = 10000
    
    for epoch in range(Nepochs):
        Xtrain,Ytrain = RandomizeTrainData(Xtrain0,Ytrain0,time_step)
        Ntrain = Xtrain.shape[0]/BatchSize
        Ntrain = Ntrain.astype(np.int32)
        for i in range(Ntrain):
            Xi = Xtrain[(BatchSize)*i:(BatchSize)*(i+1),:,:]
            Yi = Ytrain[(BatchSize)*i:(BatchSize)*(i+1),:,:]
            Xi = Np2Var(Xi)
            Yi = Yi[:,0,:]
            Yi = Np2Var(Yi)
            prediction = model(Xi) # rnn output
            loss = loss_func(prediction,Yi)
            optimizer.zero_grad() # clear gradients for this training step
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradients
            # !! next step is important !!
            """
            h_state = rnn.init_hidden(BatchSize)
            h_state[0].detach_()
            h_state[1].detach_()
            """
            print(loss)
        e_list.append(epoch)
        Ypred = model(Xval)
        loss_test = loss_func(Ypred,Yval)
        t_loss_list.append(loss.data.numpy())
        loss_list.append(loss_test.data.numpy())
        print('current loss at epoch = ',epoch,': ',loss_test.data.numpy())
        if (np.absolute(loss.data.numpy()) <= 0.0005) or (epoch == 1000):
            break