

## If you want to force CPU use instead of GPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

## Configuration for GPU limits:
from keras import backend as K
if 'tensorflow' == K.backend():
    import tensorflow as tf
    print(tf.__version__)
    print(K.tensorflow_backend._get_available_gpus())
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
print("*** Before setting allow_growth:")
print(config.gpu_options.per_process_gpu_memory_fraction)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
print("*** After setting allow_growth:")
print(config.gpu_options.per_process_gpu_memory_fraction)
#session = tf.Session(config=config)
set_session(tf.Session(config=config))

## Check if GPU is being used:
from tensorflow.python.client import device_lib
print("*** Listing devices:")
print(device_lib.list_local_devices())



import numpy as np
import pandas as pd

import re

from bs4 import BeautifulSoup


import os
#os.environ['KERAS_BACKEND'] = 'theano'



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical as to_cat
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import shuffle



MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

CLASSES = 4
EPOCS = 20
USEKERAS = True


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string.decode("utf-8"))
    string = re.sub(r"\'", "", string.decode("utf-8"))
    string = re.sub(r"\"", "", string.decode("utf-8"))
    return string.strip().lower()

def load_data_imdb():
    print("Loading data...")
    data_train = pd.read_csv('../data/imdbReviews/labeledTrainData.tsv', sep='\t')
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.sentiment[idx])
    labels = to_cat(np.asarray(labels))

    return texts, labels

def load_data_liar(file_name):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["id", "label","data"], usecols=[0,1,2])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'true': 1,
        'mostly-true': 1,
        'half-true': 2,
        'barely-true': 2,
        'false': 3,
        'pants-fire': 3
    }
    labels = [transdict[i] for i in labels]
    labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

def load_data_buzzfeed(file_name = "../data/buzzfeed-facebook/bf_fb.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header= None, names=["ID",	"URL",	"label",	"data",	"error"], usecols=[2,3])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'no factual content': 1,
        'mostly true': 1,
        'mixture of true and false': 2,
        'mostly false': 3
    }
    labels = [transdict[i] for i in labels]
    labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels


def sequence_processing(texts):
    """
    word indexing and padding of the sequences
    """

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    texts = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return texts, word_index


def load_embeddings( word_index , GLOVE_FILE = "../pretrained/glove.6B.100d.txt"): ## "../pretrained/glove.6B.100d.txt"):
   print("Loading embeddings...")
   embeddings_index = {}
   f = open(GLOVE_FILE)
   for line in f:
       values = line.split()
       word = values[0]
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
   f.close()
   print('Total %s word vectors in file.' % len(embeddings_index))

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
   l_cov1 = Conv1D(128, 2, activation='relu')(embedded_sequences)
   l_pool1 = MaxPooling1D()(l_cov1)
   l_dropout1 = Dropout(0.8)(l_pool1)
   l_cov2 = Conv1D(128, 3, activation='relu')(l_dropout1)
   l_pool2 = MaxPooling1D()(l_cov2)
   l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
   l_pool3 = MaxPooling1D()(l_cov3)  # global max pooling
   l_flat = Flatten()(l_pool3)
   l_dense = Dense(128, activation='relu')(l_flat)
   l_dropout2 = Dropout(0.8)(l_dense)
   preds = Dense(CLASSES, activation='softmax')(l_dropout2)
   model = Model(sequence_input, preds)
   model.compile(loss='categorical_crossentropy',
                 optimizer='rmsprop',
                 metrics=['acc'])
   return model


def prepare_cnn_model_2(word_index, embedding_matrix):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
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
    preds = Dense(CLASSES, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model


def prepare_rnn_model_1(word_index, embedding_matrix):
    print("*** 1")
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    print("*** 2")
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    print("*** 3")
    embedded_sequences = embedding_layer(sequence_input)
    print("*** 4")
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    print("*** 5")
    preds = Dense(CLASSES, activation='softmax')(l_lstm)
    print("*** 6")
    model = Model(sequence_input, preds)
    print("*** 7")
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model


def prepare_rnn_model_2(word_index, embedding_matrix):
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer()(l_gru)
    preds = Dense(CLASSES, activation='softmax')(l_att)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model



# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


###### Modeling using tensorflow directly with tflearn methods
import tflearn
#import tensorflow.nn.static_bidirectional_rnn as bidirectional_rnn
#import tensorflow.contrib.rnn.static_bidirectional_rnn as bidirectional_rnn
#from tensorflow.contrib.rnn import static_bidirectional_rnn as bidirectional_rnn
from tflearn.data_utils import pad_sequences
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import BasicLSTMCell ,bidirectional_rnn
from tflearn.layers.estimator import regression





def prepare_rnn_model_tf(word_index, embedding_matrix):
    # tf.reset_default_graph()
    net = input_data(shape=[None, MAX_SEQUENCE_LENGTH])
    net = embedding(net, input_dim=len(word_index)+1, output_dim=EMBEDDING_DIM, trainable=False, name="EmbeddingLayer")
    net = bidirectional_rnn(net, BasicLSTMCell(EMBEDDING_DIM), BasicLSTMCell(EMBEDDING_DIM))
    net = dropout(net, 0.5)
    net = fully_connected(net, CLASSES, activation='softmax')
    net = regression(net, optimizer='adam', loss='categorical_crossentropy')
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
    # Retrieve embedding layer weights (only a single weight matrix, so index is 0)
    embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
    print('default embeddings: ', embeddingWeights[0])
    model.set_weights(embeddingWeights, embedding_matrix )
    return model


# texts_train, labels_train = load_data_liar("~/workspace/temp/liar_dataset/train.tsv")
# texts_test,labels_test = load_data_liar("~/workspace/temp/liar_dataset/test.tsv")


## USE LIAR DATA FOR TRAINING A MODEL AND TEST DATA BOTH FROM LIAR AND BUZZFEED
texts_train, labels_train = load_data_liar("../data/liar_dataset/train.tsv")
texts_valid, labels_valid = load_data_liar("../data/liar_dataset/valid.tsv")
texts_test1, labels_test1 = load_data_liar("../data/liar_dataset/test.tsv")

texts_test2, labels_test2 = load_data_buzzfeed()

#texts, labels = shuffle(texts, labels,random_state=123)
#labels_train = labels[:len(labels)/2]
#labels_valid = labels[len(labels)/2: -len(labels)/4]
#labels_test = labels[-len(labels)/4:]

texts = texts_train + texts_valid + texts_test1 + texts_test2
texts, word_index = sequence_processing(texts)
texts_train = texts[:len(labels_train)]
texts_valid = texts[len(labels_train): len(labels_train) + len(labels_valid)]
texts_test1 = texts[len(labels_train) + len(labels_valid): len(labels_train) + len(labels_valid) + len(labels_test1)]
texts_test2 = texts[len(labels_train) + len(labels_valid) + len(labels_test1):]

labels_train = np.asarray(labels_train)
labels_valid = np.asarray(labels_valid)
labels_test1 = np.asarray(labels_test1)
labels_test2 = np.asarray(labels_test2)

print('Shape of data tensor:', texts_train.shape)
print('Shape of label tensor:', labels_train.shape)

print('Shape of data tensor:', texts_valid.shape)
print('Shape of label tensor:', labels_valid.shape)

print('Shape of data tensor:', texts_test1.shape)
print('Shape of label tensor:', labels_test1.shape)

print('Shape of data tensor:', texts_test2.shape)
print('Shape of label tensor:', labels_test2.shape)



embedding_matrix = load_embeddings(word_index)

'''
print("Preparing validation/training data split...")
nb_validation_samples = int(VALIDATION_SPLIT * texts.shape[0])
x_train = texts[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = texts[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
'''

x_train = texts_train
y_train = labels_train
x_val = texts_valid
y_val = labels_valid
x_test1 = texts_test1
y_test1 = labels_test1
x_test2 = texts_test2
y_test2 = labels_test2


print('Number of instances from each class')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

print('Baseline accuracies:')
print (y_train.sum()/1.0*len(y_train))
print (y_val.sum()/1.0*len(y_val))
print (y_test1.sum()/1.0*len(y_test1))
print (y_test2.sum()/1.0*len(y_test2))


print("Preparing the deep learning model...")
model = prepare_cnn_model_1(word_index, embedding_matrix)
# model.summary()
print("Model fitting...")

current_loss = 10000

for i in range(0, EPOCS):
    x_train, y_train = shuffle(x_train, y_train)
    if( USEKERAS ):
        model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=1, batch_size=64)
    else:
        model.fit(x_train, y_train, validation_set=0.1, n_epoch=1, show_metric=True, batch_size=64)
    prev_loss = current_loss
    current_loss = round(model.evaluate(x_val, y_val)[0],1)
    print("Loss on validation set: " + str(current_loss))
    if( current_loss > prev_loss):
        print("SHOULD STOP HERE!")
    p = model.evaluate(x_test1, y_test1)
    print("Accuracy on liar test: " + str(p))
    p = model.evaluate(x_test2, y_test2)
    print("Accuracy on buzzfeed test: " + str(p))

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
