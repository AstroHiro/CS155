#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:30:10 2020

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

class Hitomaro:
    def __init__(self,seq_length=9,Nsemi=1,whitespace=True):
        self.seq_length = seq_length
        self.Nsemi = Nsemi
        self.whitespace = whitespace
        self.text = open(os.path.join(os.getcwd(),'data/hyakuni_isshu.txt'),encoding='utf-8').read()
        self.GetTankaData()
        self.GetCharacterMap()
        self.MapReverser()
        self.GetTrainingData()
        
    def GetTankaData(self):
        tankas_org = [line for line in self.text.split('\n') if line.split()]
        tankas = []    
        for i in range(len(tankas_org)):
            if self.whitespace == True:
                tankas.append(tankas_org[i].strip(str(i+1)+' '))
            else:
                tankas.append(''.join(tankas_org[i].strip(str(i+1)+' ').split()))
        self.tankas = tankas
        pass

    def GetCharacterMap(self):
        tankas = self.tankas
        seq_length = self.seq_length
        text_list = list(' '.join(tankas))
        char_counter = 0
        char_map = {}
        for char in text_list:
            if char not in char_map:
                char_map[char] = char_counter
                char_counter += 1
        obs = []
        for p in range(len(tankas)):
            tanka = tankas[p]
            tanka_list = list(tanka)
            Nchar_p = len(tanka)
            for c in range(Nchar_p+1-seq_length):
                obs_seq = []
                for s in range(seq_length):
                    obs_seq.append(char_map[tanka_list[c+s]])
                obs.append(obs_seq)
        self.char_map = char_map
        self.char_size = len(char_map)
        self.obs = obs
        pass

    def MapReverser(self):
        char_map = self.char_map
        char_map_r = {}
        for key in char_map:
            char_map_r[char_map[key]] = key
        self.char_map_r = char_map_r
        pass
    
    def GetTrainingData(self):
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
        X = self.X
        y = self.y
        model = self.LSTMmodel(T,Nunit)
        es = EarlyStopping(monitor='accuracy',patience=20)
        model.fit(X,y,epochs=eps,callbacks=[es],verbose=2)
        return model
    
    def RNNValidation(self,epochs):
        Ts = [1.5,0.75,0.25]
        Nunits = [100,150,200]
        iT = 0
        for T in Ts:
            for Nunit in Nunits:
                model = self.RNNTraining(T,Nunit,epochs)
                model.save('models/Nsemi'+str(self.Nsemi)+'/'+'rnn_iT_'+str(iT)+'_units_'+str(Nunit)+'.h5')
            iT += 1
        pass
    
    def RNNValidationTemp(self,epochs):
        Ts = [1.5,0.75,0.25]
        Nunits = [100,150,200]
        iT = 0
        for T in Ts:
            print(T)
            for Nunit in Nunits:
                print('T = ',T)
                print('Nunit = ',Nunit)
                if (T != 1.5) or (Nunit != 100):
                    model = self.RNNTraining(T,Nunit,epochs)
                    model.save('models/Nsemi'+str(self.Nsemi)+'/'+'rnn_iT_'+str(iT)+'_units_'+str(Nunit)+'.h5')
            iT += 1
        pass
    
    def obs_sentence(self,obsi):
        char_map_r = self.char_map_r
        sentence = [char_map_r[i] for i in obsi]
        return ''.join(sentence)
    
    def obs_sentences(self):
        obs = self.obs
        out = ''
        for obsi in obs:
            tanka = self.obs_sentence(obsi)
            out += tanka+'\n'
        return out
    
    def GenerateAItanka(self,input_char,model,Nchar):
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

if __name__ == "__main__":
    hito9 = Hitomaro(seq_length=9)
    model9 = hito9.RNNTraining(1,200,1000)
    model9.save('models/Hitomaro/hito9.h5')
    hito7 = Hitomaro(seq_length=7)
    model7 = hito7.RNNTraining(1,200,1000)
    model7.save('models/Hitomaro/hito7.h5')
    hito8 = Hitomaro(seq_length=8,Nsemi=1,whitespace=False)
    model8 = hito8.RNNTraining(1,200,1000)
    model8.save('models/Hitomaro/hito8_w_ws.h5')
    hito6 = Hitomaro(seq_length=6,Nsemi=1,whitespace=False)
    model6 = hito6.RNNTraining(1,200,1000)
    model6.save('models/Hitomaro/hito6_w_ws.h5')
    