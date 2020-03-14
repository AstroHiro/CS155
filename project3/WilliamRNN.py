#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:48:39 2020

@author: hiroyasu
""" 
import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Activation
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks.callbacks import EarlyStopping

######### LSTM-RNN for generating Shakespearean sonets #########

class William:
    def __init__(self,seq_length=41,Nsemi=1):
        self.seq_length = seq_length
        self.Nsemi = Nsemi
        self.text = open(os.path.join(os.getcwd(),'data/shakespeare.txt')).read()
        self.GetPoemData()
        self.GetCharacterMap()
        self.MapReverser()
        self.GetTrainingData()
        
    def GetPoemData(self):
        # Convert text data into poem-wise observation sequences
        # obtain training sequences with length = seq_length
        # poem 99 has 15 lines and poem 126 has 12 lines
        Npoems = 0
        lines = [line for line in self.text.split('\n') if line.split()]
        lines_t = []       
        for i in range(len(lines)):
            if lines[i] == '                   '+str(Npoems+1):
                Npoems = Npoems+1
            else:
                lines_t.append(lines[i])
        cur_line = 0
        poems_rnn = []
        for p in range(Npoems):
            Nlines = 14
            if p == 99-1:
                Nlines = 15
            if p == 126-1:
                Nlines = 12
            text_rnn = ''
            for j in range(Nlines):
                text_rnn += lines_t[cur_line+j]
                if j != Nlines-1:
                    text_rnn += '\n'
                text_rnn = text_rnn.lower()
            cur_line += Nlines
            poems_rnn.append(text_rnn)
        self.poems_rnn = poems_rnn
        pass

    def GetCharacterMap(self):
        # Obtain dictionary from character to int
        poems_rnn = self.poems_rnn
        seq_length = self.seq_length
        text_list = list(''.join(poems_rnn))
        char_counter = 0
        char_map = {}
        for char in text_list:
            if char not in char_map:
                char_map[char] = char_counter
                char_counter += 1
        obs = []
        for p in range(len(poems_rnn)):
            poem = poems_rnn[p]
            poem_list = list(poem)
            Nchar_p = len(poem)
            for c in range(Nchar_p+1-seq_length):
                obs_seq = []
                for s in range(seq_length):
                    obs_seq.append(char_map[poem_list[c+s]])
                obs.append(obs_seq)
        self.char_map = char_map
        self.char_size = len(char_map)
        self.obs = obs
        pass

    def MapReverser(self):
        # Obtain dictionary from int to character
        char_map = self.char_map
        char_map_r = {}
        for key in char_map:
            char_map_r[char_map[key]] = key
        self.char_map_r = char_map_r
        pass
    
    def GetTrainingData(self):
        # Obtain split training sequences into X and y
        # Nsemi > 1 -> semi-redundant sequences
        obs_semi = []
        obs = self.obs
        Nsemi = self.Nsemi
        for i in range(len(obs)):
            if np.remainder(i,Nsemi) == 0:
                obs_semi.append(obs[i])
        data = np.array(obs_semi)
        char_size = self.char_size
        X,y = data[:,:-1],data[:,-1]
        X = np.array([to_categorical(x,num_classes=char_size) for x in X])
        y = to_categorical(y,num_classes=char_size)
        self.X = X
        self.y = y
        pass

    def LSTMmodel(self,T,Nunit):
        # LSTM model used for this task
        char_size = self.char_size
        X = self.X
        model = Sequential()
        model.add(LSTM(Nunit,input_shape=(X.shape[1],X.shape[2])))
        model.add(Dense(char_size))
        model.add(Lambda(lambda x: x/T))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        model.summary()
        return model

    def RNNTraining(self,T,Nunit,eps):
        # LSTM-RNN training
        X = self.X
        y = self.y
        model = self.LSTMmodel(T,Nunit)
        es = EarlyStopping(monitor='accuracy',patience=20)
        model.fit(X,y,epochs=eps,callbacks=[es],verbose=2)
        return model
    
    def ContinueTraining(self,model,eps):
        # Continue training for model
        X = self.X
        y = self.y
        es = EarlyStopping(monitor='accuracy',patience=20)
        model.fit(X,y,epochs=eps,callbacks=[es],verbose=2)
        return model
    
    def RNNValidation(self,epochs):
        # Train models for different hyperparameters
        Ts = [1.5,0.75,0.25]
        Nunits = [100,150,200]
        iT = 0
        for T in Ts:
            for Nunit in Nunits:
                model = self.RNNTraining(T,Nunit,epochs)
                model.save('models/Nsemi'+str(self.Nsemi)+'/'+'rnn_iT_'+str(iT)+'_units_'+str(Nunit)+'.h5')
            iT += 1
        pass
    
    def GenerateAIpoem(self,input_char,model,Nchar):
        # Generate poems using the trained model
        char_map = self.char_map
        char_map_r = self.char_map_r
        seq_length = self.seq_length
        char_size = self.char_size
        for i in range(Nchar):
            x = [char_map[char] for char in input_char]
            x = pad_sequences([x],maxlen=seq_length-1,truncating='pre')
            x = to_categorical(x,num_classes=char_size)
            yhat = model.predict_classes(x)
            next_char = char_map_r[yhat[0]]
            input_char += next_char
        return ''.join(input_char)

def ContinueTraining(model,eps):
    # Continue training for model
    X = will.X
    y = will.y
    es = EarlyStopping(monitor='accuracy',patience=20)
    model.fit(X,y,epochs=eps,callbacks=[es],verbose=2)
    return model

if __name__ == "__main__":
    will = William()
    will.RNNValidation(1000) # Takes days!!!!
    will = William(seq_length=41,Nsemi=5)
    will.RNNValidation(1000)
    will = William(seq_length=41,Nsemi=10)
    will.RNNValidation(1000)