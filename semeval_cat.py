# semeval_cat.py: semeval categorical emotion

# uncomment these to run on GPU
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input, Flatten, Embedding, Dropout, CuDNNGRU, Bidirectional, CuDNNLSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import label_binarize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adadelta, Adadelta

#from attention_helper import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import *

np.random.seed(78)

# load 250 trial/train data
file_data_train = '/media/bagus/data01/dataset/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.txt'
file_target_train = '/media/bagus/data01/dataset/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'

# load 1000 test data
file_data_test = '/media/bagus/data01/dataset/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.txt'
file_target_test = '/media/bagus/data01/dataset/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold'

text = pd.read_csv(file_data_train, header=None, error_bad_lines=False, sep='delimiter', engine='python')
text_test = pd.read_csv(file_data_test, header=None, error_bad_lines=False, sep='delimiter', engine='python')

print(text.iloc[3,0])    # print first text to check

text = list(text.iloc[:,0])
text[0]

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

# define x_test_text
text_test = list(text_test.iloc[:,0])
tokenizer.fit_on_texts(text_test)

token_ts_X = tokenizer.texts_to_sequences(text_test)
x_test_text = []

x_test_text = sequence.pad_sequences(token_ts_X, maxlen=MAX_SEQUENCE_LENGTH)

# define target_test for text_test
Yt = np.loadtxt(file_target_test)
Yt = Yt[:,1:]
Yt_max = Yt.argmax(axis=-1)
out = np.zeros_like(Yt)
out[np.arange(len(Yt)), Yt_max] = 1
Yt = out

# load label/target data
Y = np.loadtxt(file_target_train)
Y = Y[:,1:]
Y[2]
Y.shape
a_max = Y.argmax(axis=-1)
a_max[0]
out = np.zeros_like(Y)
out[np.arange(len(Y)), a_max] = 1
Y = out

Y.shape

# concatenate train and test data to make data bigger
data = np.vstack((x_train_text, x_test_text))
target = np.vstack((Y, Yt))

rmsprop = RMSprop(lr=0.000001, decay=0.01)
sgd=SGD(lr=0.000001)

# starting deeplearning: RNN architecture, varying: Bidirectional, Attention, weighting
model = Sequential()
#model.add(Embedding(nb_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
model.add(CuDNNGRU(512, return_sequences=True))
model.add(CuDNNGRU(256, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', 
              metrics=['acc'])

#model.compile()
model.summary()

hist = model.fit(data[:1200], target[:1200], batch_size=10, epochs=30, shuffle=True, validation_split=0.2, verbose=1) #
#max(hist.history['val_acc'])
loss, acc1 = model.evaluate(data[1200:], target[1200:]) #data[1200:], target[1200:]) 
print(acc1)
# 0.4

y_predict = model.predict(data[1200:]) #data[1200:])

# compute precision recall and fscore
y_pred = np.argmax(y_predict, axis=-1)
y_pred
y_true = np.argmax(target[1200:], axis=-1) #target[1200:], axis=-1)
y_true
prec = precision_recall_fscore_support(y_true, y_pred, average='weighted')
prec
class_names = np.array(['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'])
ax = plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

#ax.figure.savefig('confmat_semeval.pdf', bbox_inches="tight")
