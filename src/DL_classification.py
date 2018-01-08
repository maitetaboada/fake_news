
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

#os.environ['KERAS_BACKEND'] = 'theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

CLASSES = 6


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def load_data_imdb():
    print "Loading data..."
    data_train = pd.read_csv('../data/imdbReviews/labeledTrainData.tsv', sep='\t')
    print data_train.shape
    texts = []
    labels = []
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.sentiment[idx])
    labels = to_categorical(np.asarray(labels))

    return texts, labels

def load_data_liar(file_name= "../data/liar_dataset/train.tsv"):
    print "Loading data..."
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["id", "label","data"], usecols=[0,1,2])
    print data_train.shape
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])

    transdict = {
        'true': 0,
        'mostly-true': 1,
        'half-true': 2,
        'barely-true': 3,
        'false': 4,
        'pants-fire': 5
    }
    labels = [transdict[i] for i in labels]
    labels = to_categorical(np.asarray(labels))

    print(texts[0:6])
    print(labels[0:6])

    return texts, labels







def sequence_processing(texts):
    """
    word indexing and padding of the sequences
    """

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    texts = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return texts, word_index


def load_embeddings( word_index , GLOVE_FILE = "../pretrained/glove.6B.100d.txt"):
   print "Loading embeddings..."

   embeddings_index = {}
   f = open(GLOVE_FILE)
   for line in f:
       values = line.split()
       word = values[0]
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
   f.close()

   print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

   embedding_matrix = np.random.random((len(word_index ) + 1, EMBEDDING_DIM))
   for word, i in word_index.items():
       embedding_vector = embeddings_index.get(word)
       if embedding_vector is not None:
           # words not found in embedding index will be all-zeros.
           embedding_matrix[i] = embedding_vector

   return embedding_matrix



def prepare_cnn_model_1(word_index, embedding_matrix):

   embedding_layer = Embedding(len(word_index) + 1,
                               EMBEDDING_DIM,
                               weights=[embedding_matrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=True)
   sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
   embedded_sequences = embedding_layer(sequence_input)
   l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
   l_pool1 = MaxPooling1D(5)(l_cov1)
   l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
   l_pool2 = MaxPooling1D(5)(l_cov2)
   l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
   l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
   l_flat = Flatten()(l_pool3)
   l_dense = Dense(128, activation='relu')(l_flat)
   preds = Dense(CLASSES, activation='softmax')(l_dense)

   model = Model(sequence_input, preds)
   model.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['acc'])
   return model




#texts_train, labels_train = load_data_liar("~/workspace/temp/liar_dataset/train.tsv")
#texts_test,labels_test = load_data_liar("~/workspace/temp/liar_dataset/test.tsv")


texts_train, labels_train = load_data_liar("../data/liar_dataset/train.tsv")
texts_test,labels_test = load_data_liar("../data/liar_dataset/test.tsv")


texts = texts_train + texts_test
texts, word_index = sequence_processing(texts)
texts_train = texts[:len(labels_train)]
texts_test = texts[len(labels_train):]

labels_train = np.asarray(labels_train)
labels_test = np.asarray(labels_test)

print('Shape of data tensor:', texts_train.shape)
print('Shape of label tensor:', labels_train.shape)

indices = np.arange(texts_train.shape[0])
np.random.shuffle(indices)
texts = texts_train[indices]
labels = labels_train[indices]

embedding_matrix = load_embeddings(word_index)

nb_validation_samples = int(VALIDATION_SPLIT * texts.shape[0])
x_train = texts[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = texts[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of instances from each class')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

model = prepare_cnn_model_1(word_index, embedding_matrix)
print("model fitting - simplified convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
             nb_epoch=10, batch_size=128)



p = model.evaluate( x_val, y_val, verbose=0)
#print(pd.DataFrame({'Predicted': p, 'Expected': y_val}))
print p

'''
f1 = f1_score(y_val, p, average=None)
pre = precision_score(y_val, p, average=None)
rec = recall_score(y_val, p, average=None)
accuracy = accuracy_score(y_val, p)
print("\n************ SUMMARY ***********")
print 'Test data size: ' + str(len(y_val))
print 'Test F1-Score: ' + str(f1)
print 'Test Precision: ' + str(pre)
print 'Test Recall: ' + str(rec)
print 'Test Accuracy: ' + str(accuracy)
print("********************************")

'''

'''
   embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
   for word, i in word_index.items():
       embedding_vector = embeddings_index.get(word)
       if embedding_vector is not None:
           # words not found in embedding index will be all-zeros.
           embedding_matrix[i] = embedding_vector
   
   embedding_layer = Embedding(len(word_index) + 1,
                               EMBEDDING_DIM,
                               weights=[embedding_matrix],
                               input_length=MAX_SEQUENCE_LENGTH,
                               trainable=True)
   
   
   # applying a more complex convolutional approach
   convs = []
   filter_sizes = [3, 4, 5]
   
   sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
   embedded_sequences = embedding_layer(sequence_input)
   
   for fsz in filter_sizes:
       l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
       l_pool = MaxPooling1D(5)(l_conv)
       convs.append(l_pool)
   
   l_merge = Merge(mode='concat', concat_axis=1)(convs)
   l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
   l_pool1 = MaxPooling1D(5)(l_cov1)
   l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
   l_pool2 = MaxPooling1D(30)(l_cov2)
   l_flat = Flatten()(l_pool2)
   l_dense = Dense(128, activation='relu')(l_flat)
   preds = Dense(2, activation='softmax')(l_dense)
   
   model = Model(sequence_input, preds)
   model.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['acc'])
   
   print("model fitting - more complex convolutional neural network")
   model.summary()
   model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=20, batch_size=50)
'''