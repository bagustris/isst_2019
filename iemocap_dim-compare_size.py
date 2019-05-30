# Dimenisonal emotion recognition, IEMOCAP dataset
# step: pipeline
# make it word for regression problem (1 output) first --> expand to 3
# Experimentation done, and should be run, in JupyterLab

import os
import sys
import numpy as np
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from keras.models import Model
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, CuDNNGRU

with open('/media/bagus/data01/dataset/IEMOCAP_full_release/data_collected_full.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
# load data
text = [t['transcription'] for t in data]

# load output
v = [v['v'] for v in data]
a = [a['a'] for a in data]
d = [d['d'] for d in data]

vad = np.array([v, a, d])
vad = vad.T
print(vad.shape)

# Word embedding calculation
MAX_SEQUENCE_LENGTH = 50

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

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

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

# Now train with 3 outputs VAD. It works!
## build model
def emotion_model():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    #net = Embedding(2737, 128, input_length=500)(inputs)
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(inputs)
    net = Bidirectional(CuDNNGRU(64, return_sequences=True))(net)
    net = Bidirectional(CuDNNGRU(64, return_sequences=False))(net)
    net = Dense(32)(net)
    net = Dropout(0.5)(net)
    net = Dense(3)(net) #linear activation
    model = Model(inputs=inputs, outputs=net) #[out1, out2, out3]
    model.compile(optimizer='rmsprop', loss='mse', metrics= ['mape'])
    
    return model

model = emotion_model()
hist = model.fit(x_train_text[:4800], vad[:4800], epochs=30, batch_size=32, verbose=1, validation_split=0.2)
print(min(hist.history['val_mean_absolute_percentage_error']))

# evaluation 
eval_metrik = model.evaluate(x_train_text[4800:6000], vad[4800:6000])
print(eval_metrik)

# uncomment to print prediction
#y_predict = model.predict(x_train_text[8000:], batch_size=None, verbose=0, steps=None)
#y_predict
#print(vad[8000:])

# Uncomment to plot the error
import matplotlib.pyplot as plt
filename = os.path.basename("__file__")
fig, ax = plt.subplots()
ax.plot(hist.history['mean_absolute_percentage_error'], label='train mape')
ax.plot(hist.history['val_mean_absolute_percentage_error'], label='val mape')
ax.legend(loc='best', fontsize=10)
ax.set_xlabel('epochs')
ax.set_ylabel('MAPE(%)')
ax.figure.savefig('err_{}.pdf'.format(filename), bbox_inches='tight')

