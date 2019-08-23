# test VAD prediction from emobank dataset
import os
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Dropout, CuDNNLSTM, CuDNNGRU

import codecs

emo_data = '/media/bagus/data01/s3/course/minor/dimensional_emotion/emobank.csv'
data = pd.read_csv(emo_data)
list(data.keys())

# read transcription
text = data.text

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

#MAX_SEQUENCE_LENGTH = 500
MAX_SEQUENCE_LENGTH = len(max(text, key=len))

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
def build_model():
    input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    trainable = True)(input)
    net = CuDNNGRU(64, return_sequences=True)(net)
    net = CuDNNGRU(64, return_sequences=False)(net)
    #net = Dense(32)(net)
    net = Dropout(0.5)(net)
    net = Dense(3)(net)
    
    model = Model(inputs=input, outputs=net)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
    return model

model = build_model()
model.summary()

hist = model.fit(x_train_text[:9000], vad[:9000], epochs=30, batch_size=32, verbose=1, validation_split=0.2)

# evaluation 
eval_metrik = model.evaluate(x_train_text[9000:], vad[9000:])
print(eval_metrik)

# print([min(hist.history['val_loss']), min(hist.history['val_mean_absolute_percentage_error'])])
# print([min(hist.history['loss']), min(hist.history['mean_absolute_percentage_error'])])

# # predict
# y_predict = model.predict(x_train_text[8000:], batch_size=None, verbose=0, steps=None)
# y_predict
# print(vad[8000:])

# import matplotlib.pyplot as plt
# filename = os.path.basename("__file__")
# fig, ax = plt.subplots()
# ax.plot(hist.history['mean_absolute_percentage_error'], label='train mape')
# ax.plot(hist.history['val_mean_absolute_percentage_error'], label='val mape')
# ax.legend(loc='best', fontsize=10)
# ax.set_xlabel('epochs')
# ax.set_ylabel('MAPE(%)')
# ax.figure.savefig('err_{}.pdf'.format(filename), bbox_inches='tight')
