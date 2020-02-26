#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:46:50 2020

@author: hiroyasu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
import matplotlib
matplotlib.rcParams.update({'font.size': 10})

def LoadDict(filename):
    pkl_file = open(filename,'rb')
    varout = pickle.load(pkl_file)
    pkl_file.close()
    return varout

def UVprojection(U,V,idxm):
    K = U.shape[0]
    if idxm == 1:
        mV = np.reshape(np.mean(V,axis=1),(K,1))
        V = V-mV
        U = U-mV
    A,S,Bh = np.linalg.svd(V)
    A12 = A[:,0:2]
    Uout = A12.T@U
    Vout = A12.T@V
    return Uout,Vout

def GetData():
    df = pd.read_csv('data/movies.csv',index_col=0,usecols=lambda x: x != 'Titles')
    df_titles = pd.read_csv('data/movies.csv',index_col=0,usecols=[0,1])
    titles = df_titles.Titles.tolist()
    genres = df.values
    data = np.loadtxt('data/data.txt',dtype=int)
    Nmovies = 1682
    Ndata = data.shape[0]
    Ratings = np.zeros(Nmovies)
    AveRatings = np.zeros(Nmovies)
    RatingsSF = np.zeros(Nmovies)
    RatingsAN = np.zeros(Nmovies)
    RatingsMS = np.zeros(Nmovies)
    idxSF = genres[:,15]
    idxAN = genres[:,3]
    idxMS = genres[:,12]
    for i in range(Ndata):
        Ratings[data[i,1]-1] += 1
        AveRatings[data[i,1]-1] += data[i,2]
        if idxSF[data[i,1]-1] == 1:
            RatingsSF[data[i,1]-1] += 1
        elif idxAN[data[i,1]-1] == 1:
            RatingsAN[data[i,1]-1] += 1
        elif idxMS[data[i,1]-1] == 1:
            RatingsMS[data[i,1]-1] += 1
    AveRatings = AveRatings/Ratings
    return titles,Ratings,AveRatings,RatingsSF,RatingsAN,RatingsMS,idxSF,idxAN,idxMS

def GetTitlesList(idx_list,titles):
    titles_out = []
    for idx in idx_list:
        titles_out.append(titles[idx])
    return titles_out

def PlotV2(idx,titles_genre,V2,t_kind,p_kind,fac_idx,Nclusters):
    figtitle = 'Factorization '+str(fac_idx)+': '+t_kind
    filename = 'data/plots/'+p_kind+'_'+str(fac_idx)+'.png'
    Vp = V2[:,idx]
    kmeans = KMeans(n_clusters=Nclusters,random_state=0).fit(Vp.T)
    labels = kmeans.labels_
    plt.figure(figsize=(10,10))
    for i in range(Nclusters):
        idxS = np.where(labels==i)
        plt.scatter(Vp[0,idxS],Vp[1,idxS])
    for i in range(10):
        plt.annotate(titles_genre[i],(Vp[0,:][i],Vp[1,:][i]))
    plt.grid()
    plt.title(figtitle)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()
    return Vp

def PlotIdealV2(idx,titles_genre,V2,t_kind,p_kind):
    figtitle = 'Factorization '+str(4)+': '+t_kind
    filename = 'data/plots/'+p_kind+'_'+str(4)+'.png'
    Vp = V2[:,idx]
    if p_kind == 'SF':
        labels = np.array([0,1,2,0,0,2,1,1,1,2])
        #labels = np.array([0,1,2,0,0,2,1,2,0,2])
    elif p_kind == 'AN':
        labels = np.array([2,0,1,1,2,1,2,0,1,0])
    plt.figure(figsize=(10,10))
    for i in range(3):
        idxC = np.where(labels==i)
        plt.scatter(Vp[0,idxC],Vp[1,idxC])
    for i in range(10):
        plt.annotate(titles_genre[i],(Vp[0,:][i],Vp[1,:][i]))
    plt.grid()
    plt.title(figtitle)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()
    return Vp

if __name__ == "__main__":
    titles,Ratings,AveRatings,RatingsSF,RatingsAN,RatingsMS,idxSF,idxAN,idxMS = GetData()
    idx_pop = np.argsort(-Ratings)[0:10]
    idx_best = np.argsort(-AveRatings)[0:10]
    idx10 = [23-1,28-1,64-1,98-1,127-1,131-1,135-1,272-1,313-1,655-1]
    idxSF = np.where(idxSF == 1)[0]
    idxSF = [50-1,82-1,96-1,172-1,181-1,195-1,204-1,252-1,423-1,426-1]
    idxAN = np.where(idxAN == 1)[0]
    idxAN = [1-1,71-1,95-1,99-1,404-1,418-1,420-1,501-1,588-1,969-1]
    idxMS = np.where(idxMS == 1)[0]
    idxMS = [91-1,132-1,143-1,432-1,451-1,543-1,624-1,705-1,1037-1,1286-1]
    
    titles10 = GetTitlesList(idx10,titles)
    titles_pop = GetTitlesList(idx_pop,titles)
    titles_best = GetTitlesList(idx_best,titles)
    titlesSF = GetTitlesList(idxSF,titles)
    titlesAN = GetTitlesList(idxAN,titles)
    titlesMS = GetTitlesList(idxMS,titles)

    Ua = np.load('data/U1.npy').T
    Va = np.load('data/V1.npy').T
    Ub = np.load('data/U2.npy').T
    Vb = np.load('data/V2.npy').T
    Uc = np.load('data/U3.npy').T
    Vc = np.load('data/V3.npy').T
    U2a,V2a = UVprojection(Ua,Va,1)
    U2b,V2b = UVprojection(Ub,Vb,1)
    U2c,V2c = UVprojection(Uc,Vc,1)
    
    
    Nclusters = 3
    PlotV2(idx10,titles10,V2a,'my favorite movies','10fav',1,Nclusters)
    Nclusters = 3
    PlotV2(idx_pop,titles_pop,V2a,'10 popular movies','10pop',1,Nclusters)
    Nclusters = 3
    PlotV2(idx_best,titles_best,V2a,'10 best movies','10best',1,Nclusters)
    Nclusters = 3
    PlotV2(idxSF,titlesSF,V2a,'Sci-Fi','SF',1,Nclusters)
    Nclusters = 3
    PlotV2(idxAN,titlesAN,V2a,'Animation','AN',1,Nclusters)
    Nclusters = 3
    PlotV2(idxMS,titlesMS,V2a,'Musical','MS',1,Nclusters)

    Nclusters = 3
    PlotV2(idx10,titles10,V2b,'my favorite movies','10fav',2,Nclusters)
    Nclusters = 3
    PlotV2(idx_pop,titles_pop,V2b,'10 popular movies','10pop',2,Nclusters)
    Nclusters = 3
    PlotV2(idx_best,titles_best,V2b,'10 best movies','10best',2,Nclusters)
    Nclusters = 3
    PlotV2(idxSF,titlesSF,V2b,'Sci-Fi','SF',2,Nclusters)
    Nclusters = 3
    PlotV2(idxAN,titlesAN,V2b,'Animation','AN',2,Nclusters)
    Nclusters = 3
    PlotV2(idxMS,titlesMS,V2b,'Musical','MS',2,Nclusters)

    Nclusters = 3
    PlotV2(idx10,titles10,V2c,'my favorite movies','10fav',3,Nclusters)
    Nclusters = 3
    PlotV2(idx_pop,titles_pop,V2c,'10 popular movies','10pop',3,Nclusters)
    Nclusters = 3
    PlotV2(idx_best,titles_best,V2c,'10 best movies','10best',3,Nclusters)
    Nclusters = 3
    PlotV2(idxSF,titlesSF,V2c,'Sci-Fi','SF',3,Nclusters)
    Nclusters = 3
    PlotV2(idxAN,titlesAN,V2c,'Animation','AN',3,Nclusters)
    Nclusters = 3
    PlotV2(idxMS,titlesMS,V2c,'Musical','MS',3,Nclusters)
    
    PlotIdealV2(idxSF,titlesSF,V2c,'Sci-Fi','SF')
    PlotIdealV2(idxAN,titlesAN,V2c,'Animation','AN')