#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:56:24 2020

@author: hiroyasu
"""
import re
import os
import pickle
import HMM as hmm
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize

######### HMM for generating Shakespearean sonets #########

class Williamhmm:
    def __init__(self):
        self.text = open(os.path.join(os.getcwd(),'data/shakespeare.txt')).read()
        self.GetData()
        self.GetDataPunctuation()
        self.MapReverser()
        
    def GetData(self):
        # convert text data into poem-wise observation sequences
        # poem 99 has 15 lines and poem 126 has 12 lines
        Npoems = 0
        SSP = SyllableTokenizer()
        lines = [line for line in self.text.split('\n') if line.split()]
        lines_t = []       
        for i in range(len(lines)):
            if lines[i] == '                   '+str(Npoems+1):
                Npoems = Npoems+1
            else:
                lines_t.append(lines[i])
        words_counter = 0
        words = []
        words_map = {}
        syls_counter = 0
        syls = []
        syls_map = {}
        cur_line = 0
        poems = []
        for p in range(Npoems):
            Nlines = 14
            if p == 99-1:
                Nlines = 15
            if p == 126-1:
                Nlines = 12
            text_hmm = ''
            for j in range(Nlines):
                text_hmm += lines_t[cur_line+j]
                if j != Nlines-1:
                    text_hmm += ' '
                text_hmm = (text_hmm.lower())
            cur_line += Nlines
            poems.append(text_hmm.split())
        self.poems = poems
        poems_syls = []
        for poem in poems:
            words_elem = []
            syls_elem = []
            poem_syls_elem = []
            for word in poem:
                word = re.sub(r'[^\w]', '', word).lower()
                if word not in words_map:
                    words_map[word] = words_counter
                    words_counter += 1
                words_elem.append(words_map[word])
                syllables = SSP.tokenize(word)
                for syllable in syllables:
                    poem_syls_elem.append(syllable)
                    if syllable not in syls_map:
                        syls_map[syllable] = syls_counter
                        syls_counter += 1
                    syls_elem.append(syls_map[syllable])
            poems_syls.append(poem_syls_elem)
            words.append(words_elem)
            syls.append(syls_elem)
        
        text_words = ''
        text_syls = ''
        for i in range(len(poems)):
            text_words += ' '.join(poems[i])+' '
            text_syls += ' '.join(poems_syls[i])+' '
        self.poems_syls = poems_syls
        self.words_map = words_map
        self.syls_map = syls_map
        self.words = words
        self.syls = syls
        self.word_size = len(words_map)
        self.syl_size = len(syls_map)
        self.text_words = text_words
        self.text_syls = text_syls
        pass

    def GetDataPunctuation(self,whitespace=True,poemIDX=1):
        # Convert text data into poem-wise observation sequences
        # tokenized into syllables with punctuation
        # poem 99 has 15 lines and poem 126 has 12 lines
        lines = [word_tokenize(line) for line in self.text.split('\n') if line.split()]
        obs_counter = 0
        obs = []
        obs_map = {}
        if whitespace == True:
            obs_map[' '] = obs_counter
            obs_counter += 1
        if poemIDX == 1:
            obs_map['\n'] = obs_counter
            obs_counter += 1
        N = len(lines)
        SSP = SyllableTokenizer()
        Npoems = 0
        for i in range(N):
            line = lines[i]
            obs_elem = []
            if line[0] == str(Npoems+1):
                Npoems += 1
            else:
                for iw in range(len(line)):
                    word = line[iw].lower()
                    syllables = SSP.tokenize(word)
                    for syllable in syllables:
                        if syllable not in obs_map:
                            obs_map[syllable] = obs_counter
                            obs_counter += 1
                        obs_elem.append(obs_map[syllable])
                    if (whitespace == True) and (iw != len(line)-1) \
                    and (line[iw+1] != "'s") and (line[iw+1] != ",") \
                    and (line[iw+1] != ".") and (line[iw+1] != ":") \
                    and (line[iw+1] != "?") and (line[iw+1] != "!") \
                    and (line[iw+1] != "'") and (line[iw+1] != ";") \
                    and (line[iw+1] != ")") and (line[iw] != "("):
                        obs_elem.append(obs_map[' '])
                obs.append(obs_elem)
        if poemIDX == 1:
            obs_p = []
            cur_line = 0
            for i in range(Npoems):
                obs_pi = []
                Nlines = 14
                if i == 99-1:
                    Nlines = 15
                if i == 126-1:
                    Nlines = 12
                for j in range(Nlines-2):
                    obs_pi += obs[cur_line+j]
                    obs_pi.append((obs_map["\n"]))
                for j in range(Nlines-2,Nlines):
                    obs_pi.append(obs_map[' '])
                    obs_pi.append(obs_map[' '])
                    obs_pi += obs[cur_line+j]
                    obs_pi.append((obs_map["\n"]))
                obs_p.append(obs_pi)
                cur_line += Nlines
            obs = obs_p
        self.syls_p_map = obs_map
        self.syls_p = obs
        pass

    def MapReverser(self):
        # Obtain dictionary from int to obs
        words_map = self.words_map
        words_map_r = {}
        for key in words_map:
            words_map_r[words_map[key]] = key
        self.words_map_r = words_map_r
        syls_map = self.syls_map
        syls_map_r = {}
        for key in syls_map:
            syls_map_r[syls_map[key]] = key
        self.syls_map_r = syls_map_r
        syls_p_map = self.syls_p_map
        syls_p_map_r = {}
        for key in syls_p_map:
            syls_p_map_r[syls_p_map[key]] = key
        self.syls_p_map_r = syls_p_map_r
        pass
    
    def syls_p_sentence(self,obsi):
        # Sample sentence for a single observation sequence
        syls_map_p_r = self.syls_p_map_r
        sentence = [syls_map_p_r[i] for i in obsi]
        return ''.join(sentence)
    
    def syls_p_sentences(self):
        # Sample sentence for all the observation sequences
        syls_p = self.syls_p
        out = ''
        for obsi in syls_p:
            poem = self.syls_p_sentence(obsi)
            out += poem
        return out
    
    def HMMtrainings(self,Nepochs):
        # Training algorithm using the Baum-Welch algorithm
        hidden_states = [4,8,16]
        for hstate in hidden_states:
            print('words hmm: Nstates = ',hstate)
            hmmi = hmm.unsupervised_HMM(self.words,hstate,Nepochs)
            SaveHMM('hmms/words_hmm'+str(hstate)+'.pkl',hmmi)
        for hstate in hidden_states:
            print('syllables hmm: Nstates = ',hstate)
            hmmi = hmm.unsupervised_HMM(self.syls,hstate,Nepochs)
            SaveHMM('hmms/syllables_hmm'+str(hstate)+'.pkl',hmmi)
        for hstate in hidden_states:
            print('syllables with punctuation hmm: Nstates = ',hstate)
            hmmi = hmm.unsupervised_HMM(self.syls_p,hstate,Nepochs)
            SaveHMM('hmms/syllables_with_punct_hmm'+str(hstate)+'.pkl',hmmi)
        pass
        

def SaveHMM(filename,var):
    output = open(filename,'wb')
    pickle.dump(var,output)
    output.close()
    pass
    
def LoadHMM(filename):
    pkl_file = open(filename,'rb')
    varout = pickle.load(pkl_file)
    pkl_file.close()
    return varout

if __name__ == "__main__":
    will = Williamhmm()
    poems = will.poems
    poems_syls = will.poems_syls
    words_map = will.words_map
    syls_map = will.syls_map
    will.HMMtrainings(100)
    hmm8 = hmm.unsupervised_HMM(will.words,8,1000)
    SaveHMM('hmms/words_hmm8_1000.pkl',hmm8)