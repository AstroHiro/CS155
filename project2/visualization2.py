#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:08:33 2020

@author: hiroyasu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


def grad_U(Ui,Yij,Vj,mu,ai,bj,reg,eta):
    return eta*(reg*Ui-((Yij-mu)-(Ui@Vj+ai+bj))*Vj)

def grad_V(Ui,Yij,Vj,mu,ai,bj,reg,eta):
    return eta*(reg*Vj-((Yij-mu)-(Ui@Vj+ai+bj))*Ui)

def grad_a(Ui,Yij,Vj,mu,ai,bj,reg,eta):
    return eta*(reg*ai-((Yij-mu)-(Ui@Vj+ai+bj)))

def grad_b(Ui,Yij,Vj,mu,ai,bj,reg,eta):
    return eta*(reg*bj-((Yij-mu)-(Ui@Vj+ai+bj)))

def get_err(U,V,Y,mu,a,b,idxb,reg=0.0):
    error = 0
    N = Y.shape[0]
    for k in range(N):
        i = Y[k,0]-1
        j = Y[k,1]-1
        if idxb == 1:
            Yij = Y[k,2]-mu
            Yij_hat = U[i,:]@V[j,:]+a[i]+b[j]
        else:
            Yij = Y[k,2]
            Yij_hat = U[i,:]@V[j,:]
        error += (Yij-Yij_hat)**2/2
    error = error/N
    return error


def train_model(M,N,K,eta,reg,Y,mu,idxb,eps=0.0001,max_epochs=300):
    U = np.random.uniform(-0.5,0.5,(M,K))
    V = np.random.uniform(-0.5,0.5,(N,K))
    if idxb == 1:
        a = np.random.uniform(-0.5,0.5,M)
        b = np.random.uniform(-0.5,0.5,N)
    else:
        a = np.zeros(M)
        b = np.zeros(N)
        mu = 0
    DataNum = Y.shape[0]
    e = eps
    err_tm1 = get_err(U,V,Y,mu,a,b,idxb)
    eta_da = 0
    eta_db = 0
    for epoch in range(max_epochs):
        for k in range(DataNum):
            i = Y[k,0]-1
            j = Y[k,1]-1
            Yij = Y[k,2]
            if idxb ==1:
                eta_da = grad_a(U[i,:],Yij,V[j,:],mu,a[i],b[j],reg,eta)
            eta_du = grad_U(U[i,:],Yij,V[j,:],mu,a[i],b[j],reg,eta)
            if idxb ==1:
                eta_db = grad_b(U[i,:],Yij,V[j,:],mu,a[i],b[j],reg,eta)
            eta_dv = grad_V(U[i,:],Yij,V[j,:],mu,a[i],b[j],reg,eta)
            U[i,:] = U[i,:]-eta_du
            a[i] = a[i]-eta_da
            V[j,:] = V[j,:]-eta_dv
            b[i] = b[i]-eta_db
        err_t = get_err(U,V,Y,mu,a,b,idxb)
        Deltm1t = err_tm1-err_t
        if epoch == 0:
            Del01 = err_tm1-err_t
        if Deltm1t/Del01 < e:
            print(Deltm1t/Del01)
            break
        err_tm1 = err_t
    return U,V,a,b,err_t

def GetErrors(M,N,K,Y_train,Y_test,mu,idxb):
    regs = [0,10**-4,10**-3,10**-2,10**-1,1]
    etas = [0.01,0.03,0.06,0.09]
    E_ins = []
    E_outs = []
    for eta in etas:
        E_ins_for_lambda = []
        E_outs_for_lambda = []
        for reg in regs:
            print("Training model with M = %s,N = %s,k = %s,eta = %s,reg = %s"%(M,N,K,eta,reg))
            U,V,a,b,e_in = train_model(M,N,K,eta,reg,Y_train,mu,idxb)
            E_ins_for_lambda.append(e_in)
            eout = get_err(U,V,Y_test,mu,a,b,idxb)
            E_outs_for_lambda.append(eout)
        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)
    return regs,etas,E_ins,E_outs

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

if __name__ == "__main__":
    df = pd.read_csv('data/movies.csv',index_col=0)
    M = 943
    N = 1682
    Y_train = np.loadtxt('data/train.txt',dtype=int)
    mu = np.mean(Y_train[:,2])
    Y_test = np.loadtxt('data/test.txt',dtype=int)
    K = 20
    
    '''
    regs,etas,E_ins1,E_outs1 = GetErrors(M,N,K,Y_train,Y_test,mu,0)
    SaveDict('data/regs1.pkl',regs)
    SaveDict('data/etas1.pkl',etas)
    SaveDict('data/E_ins1.pkl',E_ins1)
    SaveDict('data/E_outs1.pkl',E_outs1)
    
    regs,etas,E_ins2,E_outs2 = GetErrors(M,N,K,Y_train,Y_test,mu,1)
    SaveDict('data/regs2.pkl',regs)
    SaveDict('data/etas2.pkl',etas)
    SaveDict('data/E_ins2.pkl',E_ins2)
    SaveDict('data/E_outs2.pkl',E_outs2)
    '''
    
    U,V,a,b,e_in = train_model(M,N,K,0.01,0.1,Y_train,mu,0)
    eout = get_err(U,V,Y_test,mu,a,b,0)
    np.save('data/U1.npy',U)
    np.save('data/V1.npy',V)
    
    U,V,a,b,e_in = train_model(M,N,K,0.03,0.1,Y_train,mu,1)
    eout = get_err(U,V,Y_test,mu,a,b,1)
    np.save('data/U2.npy',U)
    np.save('data/V2.npy',V)
    
    it = 5
    i = Y_test[it,0]-1
    j = Y_test[it,1]-1
    y1 = Y_test[it,2]
    y2 = U[i,:]@V[j,:]+a[i]+b[j]+mu
    print(y1)
    print(y2)