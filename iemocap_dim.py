# Dimenisonal emotion recognition, IEMOCAP dataset
# step: pipeline
# make it word for regression problem (1 output) first --> expand to 3
# Experimentation done, and should be run, in JupyterLab

import numpy as np
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from keras.models import Model
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout

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

## build model
def emotion_model():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    #net = Masking()(input)
    net = Embedding(2737, 128, input_length=500)(inputs)
    net = Bidirectional(CuDNNLSTM(64, return_sequences=True))(net)
    net = Bidirectional(CuDNNLSTM(64, return_sequences=False))(net)
    net = Dropout(0.2)(net)
    
    #output = []
    out1 = Dense(1)(net)
    #out2 = TimeDistributed(Dense(1))(net)
    #out3 = TimeDistributed(Dense(1))(net)
    
    #output = output.append(out1)
    model = Model(input=inputs, outputs=out1)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
    
    return model

model = emotion_model()
model.summary()

# test for one output
v_np = np.array(v)
hist = model.fit(x_train_text, v_np, epochs = 16, batch_size = 32, verbose=1, validation_split=0.3)

print(min(hist.history['val_mean_absolute_percentage_error']))

# Now train with 3 outputs VAD. It works!
## build model
def emotion_model2():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(inputs)
    net = Bidirectional(CuDNNLSTM(64, return_sequences=True))(net)
    net = Bidirectional(CuDNNLSTM(64, return_sequences=False))(net)
    net = Dropout(0.2)(net)
    
    #output = []
    net = Dense(3)(net)
    #out1 = TimeDistributed(Dense(1))(net)
    #out2 = TimeDistributed(Dense(1))(net)
    #out3 = TimeDistributed(Dense(1))(net)
    model2 = Model(inputs=inputs, outputs=net) #[out1, out2, out3]
    model2.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])
    
    return model2

model2 = emotion_model2()
hist2= model2.fit(x_train_text, vad, epochs = 16, batch_size = 32, verbose=1, validation_split=0.3)
print(min(hist2.history['val_mean_absolute_percentage_error']))

# predict
model2.predict(x_train_text[9990:10000], batch_size=None, verbose=0, steps=None)
vad[9990:10000]
