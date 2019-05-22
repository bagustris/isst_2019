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

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
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
model = Sequential()
model.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
model.add(CuDNNGRU(512, return_sequences=True))
model.add(CuDNNGRU(256, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['acc'])
model.summary()

hist = model.fit(x_train_text, Y, batch_size=32, epochs=30,  validation_split=0.2, shuffle=True, verbose=1) #
#max(hist.history['val_acc'])

loss, acc1 = model.evaluate(x_train_text[6130:],Y[6130:])
print(acc1)

y_pred = model.predict(x_train_text[6130:])
y_pred = np.argmax(y_pred, axis=-1)
y_true = np.argmax(Y[6130:], axis=-1)
precision_recall_fscore_support(y_true, y_pred, average='weighted')

# plot confusion matrix
emotions_used = np.array(['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt'])
ax = plot_confusion_matrix(y_true, y_pred, classes=emotions_used, normalize=True,
                      title='Normalized confusion matrix')

ax.figure.savefig('confmat_isear.pdf', bbox_inches="tight")
