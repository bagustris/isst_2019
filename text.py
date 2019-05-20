#!/usr/bin/env python3.6

# importing necessary module
import numpy as np
import os
import sys
import copy

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, Embedding, Dropout, GRU
from keras.layers.wrappers import TimeDistributed
#from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

#from features import *
#from helper import *

code_path = os.path.dirname(os.path.realpath(os.getcwd()))
emotions_used = np.array(['ang', 'dis', 'fea', 'exc', 'sad', 'sur'])
data_path = '/media/bagus/data01/dataset/IEMOCAP_full_release/'
sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
framerate = 16000

import pickle
with open('/media/bagustris/bagus/dataset/IEMOCAP_full_release/data_collected_full.pickle', 'rb') as handle:
    data2 = pickle.load(handle)

text = []

for ses_mod in data2:
    text.append(ses_mod['transcription'])

MAX_SEQUENCE_LENGTH = 500

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

file_loc = '/media/bagustris/atmaja/github/IEMOCAP-Emotion-Detection/data/glove.840B.300d.txt'

print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding
#
f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])
    
Y = label_binarize(Y,emotions_used)

Y.shape

# starting deeplearning
model = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))

# LSTM architecture
model.add(GRU(512, return_sequences=True))
#model.add(GRU(512, return_sequences=True))
#model.add(LSTM(512, return_sequences=True))
#model.add(LSTM(512, return_sequences=True))
model.add(GRU(256, return_sequences=False))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
		optimizer='adam', 
		metrics=['acc'])

#model.compile()
model.summary()

print("Model1 Built")

hist = model.fit(x_train_text, Y, 
                 batch_size=4, epochs=10, verbose=1, 
                 validation_split=0.2)
                 
acc1 = max(hist.history['val_acc'])
print(acc1)
