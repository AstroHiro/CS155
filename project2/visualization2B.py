#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 02:00:48 2020

@author: hiroyasu
"""
import numpy as np
import pandas as pd
import surprise as sur
from surprise import accuracy
import pickle

def GetTrainSet(M,N):
    Y_train = np.loadtxt('data/train.txt',dtype=int)
    Y_test = np.loadtxt('data/test.txt',dtype=int)
    df_train = pd.DataFrame(Y_train)
    df_train.columns = ['user_id','movie_id','ratings']
    reader = sur.Reader(rating_scale=(1,5))
    sur_Y_train = sur.Dataset.load_from_df(df_train[['user_id','movie_id','ratings']],reader)
    trainset = sur_Y_train.build_full_trainset()
    trainset.n_users = M
    trainset.n_items = N
    return trainset,Y_train,Y_test

def predictY(K,trainset,eta,reg,Nepochs):
    algo = sur.SVDpp(n_factors=K,n_epochs=Nepochs,lr_all=eta,reg_all=reg,verbose=True)
    algo.fit(trainset)
    U = algo.pu
    V = algo.qi
    return algo,U,V

def GetErrors(M,N,K,trainset,Y_train,Y_test,Nepochs):
    regs = [0,10**-4,10**-3,10**-2,10**-1,1]
    etas = [0.005,0.01,0.03,0.06,0.09]
    E_ins = []
    E_outs = []
    for eta in etas:
        E_ins_for_lambda = []
        E_outs_for_lambda = []
        for reg in regs:
            print("Training model with M = %s,N = %s,k = %s,eta = %s,reg = %s"%(M,N,K,eta,reg))
            algo,U,V = predictY(K,trainset,eta,reg,Nepochs)
            predictions_train = algo.test(Y_train)
            predictions = algo.test(Y_test)
            E_ins_for_lambda.append(accuracy.rmse(predictions_train))
            E_outs_for_lambda.append(accuracy.rmse(predictions))
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)
    return regs,etas,E_ins,E_outs

def SaveDict(filename,var):
    output = open(filename,'wb')
    pickle.dump(var,output)
    output.close()
    pass

if __name__ == "__main__":
    M = 943
    N = 1682
    K = 20
    trainset,Y_train,Y_test = GetTrainSet(M,N)
    '''
    regs,etas,E_ins3,E_outs3 = GetErrors(M,N,K,trainset,Y_train,Y_test,20)
    SaveDict('data/regs3.pkl',regs)
    SaveDict('data/etas3.pkl',etas)
    SaveDict('data/E_ins3.pkl',E_ins3)
    SaveDict('data/E_outs3.pkl',E_outs3)
    '''
    algo,U,V = predictY(K,trainset,0.005,0.01,20)
    predictions = algo.test(Y_test)
    accuracy.rmse(predictions)
    
    np.save('data/U3.npy',U)
    np.save('data/V3.npy',V)