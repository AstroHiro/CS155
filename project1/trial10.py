#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:43:49 2020

@author: hiroyasu
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from sklearn.decomposition import PCA


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

tuner.search_space_summary()

def preprocessX(X):
    Xm = np.mean(X,axis=0)
    Xsig = np.std(X,axis=0)
    Xnor = (X-Xm)/Xsig
    return Xnor,Xm,Xsig

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

models = tuner.get_best_models(num_models=6)
models[0].get_config()
models[1].get_config()
models[2].get_config()
models[3].get_config()
models[4].get_config()
models[5].get_config()
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
idxv = np.random.choice(Nall,Nval,replace=False)
Xtrain = np.delete(Xall,idxv,0)
Xval = Xall[idxv,:]
Ytrain = np.delete(Yall,idxv)
Yval = Yall[idxv]
tuner.search(Xtrain,Ytrain,
             epochs=5,
             validation_data=(Xval,Yval))


