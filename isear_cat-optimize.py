#!/usr/bin/env python

import sys
import os
sys.path.insert(0, os.path.dirname('./py_isear_dataset/'))

from py_isear.isear_loader import IsearLoader
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, Embedding, Dropout, CuDNNGRU, Bidirectional, CuDNNLSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from plot_confusion_matrix import *

np.random.seed(3)
data = ['TEMPER', 'TROPHO']
target = ['EMOT']
loader = IsearLoader(data,target)
dataset = loader.load_isear('./py_isear_dataset/isear.csv')

text = dataset.get_freetext_content()
target_set = dataset.get_target()
target_chain = itertools.chain(*target_set)
target = list(target_chain)
Y = to_categorical(target)
Y = Y[:,1:]
       
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

file_loc = '/media/bagus/data01/github/IEMOCAP-Emotion-Detection/data/glove.840B.300d.txt'

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

# starting deeplearning: RNN architecture, varying: Bidirectional, Attention, weighting
def build_model(gru_unit=32, do_rate=0.2, dense_unit=0):
    model = Sequential()
    model.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
    model.add(CuDNNGRU(gru_unit, return_sequences=True))
    model.add(CuDNNGRU(gru_unit, return_sequences=False))
    model.add(Dropout(do_rate))
    model.add(Dense(dense_unit, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['acc'])
    return model

#model.summary()

# define parameter to search here
batch_sizes = [16, 32, 128]
#gru_units = [16, 64, 256]
#dense_units = [0, 16, 32]
#do_rates = [0.2, 0.4, 0.6]

param_grids = dict(batch_size=batch_sizes)#, gru_unit=gru_units, 
                    #dense_unit=dense_units, do_rate=do_rates)

model = KerasClassifier(build_fn=build_model, epochs=30, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grids, n_jobs=-1)
result = grid.fit(x_train_text[:6130], Y[:6130])

print("Best: {} using {}".format(result.best_score_, result.best_params_))


