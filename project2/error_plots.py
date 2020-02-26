#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:28:08 2020

@author: hiroyasu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.rcParams.update({'font.size': 15})

def LoadDict(filename):
    pkl_file = open(filename,'rb')
    varout = pickle.load(pkl_file)
    pkl_file.close()
    return varout

def PlotErrors(fac_idx):
    fin = 'data/E_ins'+str(fac_idx)+'.pkl'
    fout = 'data/E_outs'+str(fac_idx)+'.pkl'
    freg = 'data/regs'+str(fac_idx)+'.pkl'
    feta = 'data/etas'+str(fac_idx)+'.pkl'
    E_ins = LoadDict(fin)
    E_outs = LoadDict(fout)
    regs = LoadDict(freg)
    etas = LoadDict(feta)
    plt.figure()
    leds = []
    for i in range(len(etas)):
        plt.semilogx(regs,E_ins[i])
        leds.append(r'$\eta = $'+str(etas[i]))
    plt.legend(leds,loc='best') 
    plt.xlabel(r'Regularization $\lambda$')
    plt.ylabel('Training error')
    plt.savefig('data/plots/Ein'+str(fac_idx)+'.png',bbox_inches='tight')
    plt.show()
    
    plt.figure()
    leds = []
    for i in range(len(etas)):
        plt.semilogx(regs,E_outs[i])
        leds.append(r'$\eta = $'+str(etas[i]))
    plt.legend(leds,loc='best') 
    plt.xlabel(r'Regularization $\lambda$')
    plt.ylabel('Test error')
    plt.savefig('data/plots/Eout'+str(fac_idx)+'.png',bbox_inches='tight')
    plt.show()
    pass


if __name__ == "__main__":
    PlotErrors(1)
    PlotErrors(2)
    PlotErrors(3)