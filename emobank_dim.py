# test VAD prediction from emobank dataset

import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, CuDNNLSTM

import codecs

data = pd.read_csv('emobank.csv')
list(data.keys())

# read transcription
text = data.text

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

MAX_SEQUENCE_LENGTH = 500
x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found {} unique index'.format(len(word_index)))

# load glove embedding
data_path = '/media/bagus/data01/github/IEMOCAP-Emotion-Detection/data/'
file_loc = data_path + 'glove.840B.300d.txt'
print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding

f.close()
print('G Word embeddings:', len(gembeddings_index))

# dim of embedding is taken from glove
EMBEDDING_DIM = 300
nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))
# state output
vad = data[['V', 'A', 'D']]
vad = np.array(vad)

# build model1
def build_model1():
    input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    trainable = True)(input)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = CuDNNLSTM(64, return_sequences=False)(net)
    net = Dropout(0.2)(net)
    net = Dense(3)(net)
    
    model1 = Model(inputs=input, outputs=net)
    model1.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
    return model1

model1 = build_model1()
model1.summary()

hist1 = model1.fit(x_train_text, vad, epochs=20, batch_size=16, verbose=1, validation_split=0.3)
min_hist1 = hist1.history['val_mean_absolute_percentage_error']
print(min(min_hist1))

# model2
def build_model2():
    input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(input)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = CuDNNLSTM(64, return_sequences=False)(net)
    net = Dropout(0.2)(net)
    net = Dense(3)(net)
    
    model2 = Model(inputs=input, outputs=net)
    model2.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
    return model2

model2 = build_model2()
model2.summary()

hist2 = model2.fit(x_train_text, vad, epochs=20, batch_size=16, verbose=1, validation_split=0.3)
min_hist2 = hist2.history['val_mean_absolute_percentage_error']
print(min(min_hist2))

model2.predict(x_train_text[-15:-1], batch_size=None, verbose=0, steps=None)
data[['V', 'A', 'D']][-15:-1]

# found maxlen of seq
# data.text.str.len().max()
