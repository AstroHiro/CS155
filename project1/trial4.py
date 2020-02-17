#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:24:00 2020

@author: hiroyasu
"""
# Setup.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from IPython.display import HTML



df_train = pd.read_csv('data/train.csv', index_col=0)
df_test = pd.read_csv('data/test.csv', index_col=0)
df_train = df_train.fillna(method='ffill')
df_train = df_train.fillna(method='bfill')
df_test = df_test.fillna(method='ffill')
df_test = df_test.fillna(method='bfill')
n = (df_train.values).shape[1]-1
nt = (df_test.values).shape[1]-1
Xall = df_train.values[:,0:n]
Yall = df_train.values[:,n]
Xtest = df_test.values

class AdaBoost():
    def __init__(self, n_clfs=100):
        '''
        Initialize the AdaBoost model.

        Inputs:
            n_clfs (default 100): Initializer for self.n_clfs.        
                
        Attributes:
            self.n_clfs: The number of DT weak classifiers.
            self.coefs: A list of the AdaBoost coefficients.
            self.clfs: A list of the DT weak classifiers, initialized as empty.
        '''
        self.n_clfs = n_clfs
        self.coefs = []
        self.clfs = []

    def fit(self, X, Y, n_nodes=4):
        '''
        Fit the AdaBoost model. Note that since we are implementing this method in a class, rather
        than having a bunch of inputs and outputs, you will deal with the attributes of the class.
        (see the __init__() method).
        
        This method should thus train self.n_clfs DT weak classifiers and store them in self.clfs,
        with their coefficients in self.coefs.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of the data points.
               (Even though the labels are ints, we treat them as floats.)
            n_nodes: The max number of nodes that the DT weak classifiers are allowed to have.
            
        Outputs:
            A (N, T) shaped numpy array, where T is the number of iterations / DT weak classifiers,
            such that the t^th column contains D_{t+1} (the dataset weights at iteration t+1).
        '''
        n_clfs = self.n_clfs
        N = Y.size
        
        Dt = np.ones(N)/N
        Dthis = np.zeros((N,n_clfs))
        for i in range(n_clfs):
            clf = DecisionTreeClassifier(max_leaf_nodes = n_nodes)
            clf = clf.fit(X, Y, sample_weight = Dt)
            zo_loss = np.zeros(N)
            zo_loss[np.where(clf.predict(X) != Y)[0]] = 1
            et = Dt@zo_loss
            at = np.log((1-et)/et)/2
            Zt = (1-et)*np.exp(-at)+et*np.exp(at)
            Dt = Dt*np.exp(-at*Y*clf.predict(X))/Zt
            self.coefs.append(at)
            self.clfs.append(clf)
            Dthis[:, i] = Dt
        return Dthis

    
    def predict(self, X):
        '''
        Predict on the given dataset.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            
        Outputs:
            A (N, ) shaped numpy array containing the (float) labels of the data points.
            (Even though the labels are ints, we treat them as floats.)
        '''
        # Initialize predictions.
        Y_pred = np.zeros(len(X))
        
        # Add predictions from each DT weak classifier.
        for i, clf in enumerate(self.clfs):
            Y_curr = self.coefs[i] * clf.predict(X)
            Y_pred += Y_curr

        # Return the sign of the predictions.
        return Y_pred

    def loss(self, X, Y):
        '''
        Calculate the classification loss.

        Inputs:
            X: A (N, D) shaped numpy array containing the data points.
            Y: A (N, ) shaped numpy array containing the (float) labels of the data points.
               (Even though the labels are ints, we treat them as floats.)
            
        Outputs:
            The classification loss.
        '''
        # Calculate the points where the predictions and the ground truths don't match.
        Y_pred = self.predict(X)
        misclassified = np.where(Y_pred != Y)[0]

        # Return the fraction of such points.
        return float(len(misclassified)) / len(X)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def preprocessX(X):
    n = X.shape[1]
    XC = np.amax(X,axis=0)
    for i in range(n):
        X[:,i] = X[:,i]/XC[i]
    return X,XC
  
if __name__ == "__main__":
    Xall,XC = preprocessX(Xall)
    time_step = 10 # rnn time step
    N = np.size(Xall,0)
    BatchSize = np.floor(N/100)
    BatchSize = BatchSize.astype(np.int32)
    ValSize = 1000-1
    Xval = Xall[0:ValSize+1,:]
    Yval = Yall[0:ValSize+1]
    Xtrain0 = Xall[ValSize+1:,:]
    Ytrain0 = Yall[ValSize+1:]
    ada = AdaBoost()
    Yall[Yall == 0] = -1
    ada.fit(Xall,Yall)
    df_test['Predicted'] = sigmoid(ada.predict(Xtest))
    df_test[['Predicted']].to_csv('data/submission.csv')
    