########################################################################################################################
# First Attempt
#   FirstModel.add(Embedding(max_features, 32))
#   FirstModel.add(SimpleRNN(32))
#   FirstModel.add(Dense(1, activation = 'sigmoid'))
#       91% on first shallow network
#   SecondModel
#       92% on two layre shallow network
#
#
#
#
########################################################################################################################

import re
import pandas as pd
import numpy as np
import os, re
import bz2
import gc


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPool1D, Dropout, concatenate, Layer, InputSpec, CuDNNLSTM
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.utils.conv_utils import conv_output_length
from keras.regularizers import l2
from keras.constraints import maxnorm


test_file = bz2.BZ2File('C:\\Users\szreb\Documents\CodeSandBox\Amazon\data\Input\\test.ft.txt.bz2')
train_file = bz2.BZ2File('C:\\Users\szreb\Documents\CodeSandBox\Amazon\data\Input\\train.ft.txt.bz2')

train_file_lines = train_file.readlines()
test_file_lines = test_file.readlines()

del train_file, test_file
gc.collect()

###### Convert from raw binary strings to strings that can be parsed ######

train_file_lines = [x.decode('utf-8') for x in train_file_lines]
test_file_lines = [x.decode('utf-8') for x in test_file_lines]

train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])

for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in \
            train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in \
            test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

del train_file_lines, test_file_lines
gc.collect()
##### tokenize data #####

max_features = 20000
maxlen = 100

tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(train_sentences)

tokenized_train = tokenizer.texts_to_sequences(train_sentences)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

#X_train[0]

tokenized_test = tokenizer.texts_to_sequences(test_sentences)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

##### Apply glove for Twitter ######

EMBEDDING_FILE = 'C:\\Users\\szreb\\Documents\\CodeSandBox\\CoreData\\glove_twitter_27B_100d\\glove.twitter.27B.100d.txt'

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))


all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
#change below line if computing normal stats is too slow
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
#embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del tokenized_test, tokenized_train, tokenizer, train_sentences, test_sentences, word_index, embeddings_index, all_embs, nb_words
gc.collect()

##### Model Design ######

batch_size = 2048
embed_size = 100

from keras.layers import SimpleRNN

SecondModel = Sequential()
SecondModel.add(Embedding(max_features, 64))
SecondModel.add(SimpleRNN(64, return_sequences=True))
SecondModel.add(SimpleRNN(64, return_sequences=True))
#return_sequencnes = True... all timesteps.

SecondModel.add(SimpleRNN(32))
SecondModel.add(Dense(1, activation = 'sigmoid'))
SecondModel.compile(optimizer = 'rmsprop',
                    loss = 'binary_crossentropy',
                    metrics = ['acc'])
#                   epochs = 10,
#                   batch_size = 128)

SecondModel.summary()

history = SecondModel.fit(X_train,
                         train_labels,
                         batch_size=batch_size,
                         epochs = 10,
                         shuffle = True,
                         validation_split=0.20)


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, val_acc)
plt.plot( range(1, len(acc) + 1) , acc)

plt.show()

val_acc
max(val_acc)
