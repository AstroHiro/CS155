#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:22:48 2020

@author: hiroyasu
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import projection as proj
matplotlib.rcParams.update({'font.size': 10})

def GetPlots(x_hist,idx,figtitle,filename):
    Ntitles = idx.shape[0]
    xh = np.ones(np.shape(x_hist[x_hist==idx[0]+1])[0])*0
    for i in range(Ntitles-1):
        xh = np.hstack((xh,np.ones(np.shape(x_hist[x_hist==idx[i+1]+1])[0])*(i+1)))
    plt.figure()
    plt.title(figtitle)
    plt.hist(xh,bins=Ntitles)
    plt.xlabel('Movies ID',fontsize=LabelSize)
    plt.ylabel('Total number of ratings',fontsize=LabelSize)
    plt.xticks(np.arange(0,Ntitles),idx+1)
    plt.yticks(np.array([0,1,2,3]))
    plt.savefig(filename,bbox_inches='tight')
    plt.show()
    pass

def GetPlotsAll(x_hist,idx,Nmovies,figtitle,filename):
    Ntitles = idx.shape[0]
    xh = x_hist[x_hist==idx[0]+1]
    for i in range(Ntitles-1):
        xh = np.hstack((xh,x_hist[x_hist==idx[i+1]+1]))
    plt.figure(figsize=(10,10))
    plt.title(figtitle)
    plt.hist(xh,bins=Nmovies)
    plt.xlabel('Movies ID',fontsize=LabelSize)
    plt.ylabel('Total number of ratings',fontsize=LabelSize)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()
    pass

if __name__ == "__main__":
    LabelSize = 15
    Nmovies = 1682
    data = np.loadtxt('data/data.txt',dtype=int)
    x_hist = data[:,1]
    titles,Ratings,AveRatings,RatingsSF,RatingsAN,RatingsMS,idxSF,idxAN,idxMS = proj.GetData()
    idx_pop = np.argsort(-Ratings)[0:10]
    idx_best = np.argsort(-AveRatings)[0:10]
    idxSF = np.where(idxSF == 1)[0]
    idxAN = np.where(idxAN == 1)[0]
    idxMS = np.where(idxMS == 1)[0]
    
    plt.figure(figsize=(10,10))
    plt.hist(x_hist,bins=Nmovies)
    plt.xlabel('Movies ID',fontsize=LabelSize)
    plt.ylabel('Total number of ratings',fontsize=LabelSize)
    xt = np.arange(0,2000,250)
    xt[0] = 1
    plt.xticks(xt)
    plt.title('All movies')
    plt.savefig('data/plots/hist_all.png',bbox_inches='tight')
    plt.show()
    
    GetPlots(x_hist,idx_pop,'10 most popular movies','data/plots/hist_10pop.png')
    GetPlots(x_hist,idx_best,'10 best movies','data/plots/hist_10best.png')
    GetPlotsAll(x_hist,idxSF,Nmovies,'Sci-Fi','data/plots/histSF.png')
    GetPlotsAll(x_hist,idxAN,Nmovies,'Animation','data/plots/histAN.png')
    GetPlotsAll(x_hist,idxMS,Nmovies,'Musical','data/plots/histMS.png')



