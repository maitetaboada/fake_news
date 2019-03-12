import numpy as np
import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Concatenate
from keras.models import Model
import keras_metrics as km

from sklearn.model_selection import StratifiedKFold
from textutils import DataLoading

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


cnn_docs = np.load("../dump/cnn_dump.npy")
cnn_labels = np.zeros(cnn_docs.shape)

dm_docs = np.load("../dump/dm_dump.npy")
dm_labels = np.ones(dm_docs.shape)

docs = np.concatenate((cnn_docs, dm_docs))
labels = np.concatenate((cnn_labels, dm_labels))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(docs)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

sequences = tokenizer.texts_to_sequences(docs)
x_news = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_news = to_categorical(np.asarray(labels))

n_rows_total = x_news.shape[0]
n_rows_train = int(n_rows_total * 0.95)
indices = np.arange(n_rows_total)
np.random.shuffle(indices)
x_news = x_news[indices]
y_news = y_news[indices]
x_news_train = x_news[:n_rows_train]
x_news_valid = x_news[n_rows_train:]
y_news_train = y_news[:n_rows_train]
y_news_valid = y_news[n_rows_train:]

texts_train = np.load("../dump/trainRaw")
labels_train = np.load("../dump/trainlRaw")
sequences = tokenizer.texts_to_sequences(texts_train)

x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_train = to_categorical(np.asarray(labels_train))

print('Shape of data tensor:', x_train.shape)
print('Shape of label tensor:', y_train.shape)

texts_valid = np.load("../dump/validRaw")
labels_valid = np.load("../dump/validlRaw")

sequences = tokenizer.texts_to_sequences(texts_valid)
x_valid = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_valid = to_categorical(np.asarray(labels_valid))

texts_test = np.load("../dump/testRaw")
sequences = tokenizer.texts_to_sequences(texts_test)
x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

y_test = np.load("../dump/testlRaw")
y_test = to_categorical(np.asarray(y_test))

snopes_filename = "../data/snopes/snopes_checked_v02_right_forclassificationtest.csv"
texts_snopesChecked, labels_snopesChecked = DataLoading.load_data_snopes(snopes_filename, 2)

print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
snopestext_test, snopeslabels_test, _, _ = DataLoading.balance_data(texts_snopesChecked, labels_snopesChecked, 40, [2, 5])
sequences = tokenizer.texts_to_sequences(snopestext_test)
snopestext_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
snopeslabels_test = to_categorical(np.asarray(snopeslabels_test))


# Maybe add Dropout(0.5)?
def create_simple_model():
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)

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
    preds = Dense(2, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    return model


# pre-train simple model
print("Pre-training simple CNN model")
model = create_simple_model()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', km.f1_score()])
model.summary()
model.fit(x_news_train, y_news_train, validation_data=(x_news_valid, y_news_valid),
          epochs=10, batch_size=128)
model.save_weights("cnn_pretrain_simple.h5")


x_combined = np.concatenate((x_train, x_valid), axis=0)
y_combined = np.concatenate((y_train, y_valid), axis=0)
n_rows_total = x_combined.shape[0]
n_rows_train = x_train.shape[0]

# Use k-fold cross validation
cv_scores = []

skf = StratifiedKFold(n_splits=3)

for train_index, valid_index in skf.split(x_combined, y_combined[:, 1]):
    print("model fine-tuning - simplified convolutional neural network")

    x_train = x_combined[train_index]
    x_valid = x_combined[valid_index]
    y_train = y_combined[train_index]
    y_valid = y_combined[valid_index]

    model = create_simple_model()
    model.load_weights("cnn_pretrain_simple.h5")
    for l in model.layers[:9]:
        l.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', km.f1_score()])

    model.summary()
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
              epochs=10, batch_size=128)

    scores = model.evaluate(x_valid, y_valid, verbose=0)
    cv_scores.append(scores)

print("Scores for CV on simple model")
print(list(zip(model.metrics_names, np.mean(cv_scores, axis=0))))


# train on the full dataset

model = create_simple_model()
model.load_weights("cnn_pretrain_simple.h5")
for l in model.layers[:9]:
    l.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', km.f1_score()])

model.summary()
model.fit(x_combined, y_combined, validation_data=(x_test, y_test),
          epochs=10, batch_size=128)

# test on heldout test and snopes data

print("Scores for Test set:")
scores = model.evaluate(x_test, y_test, verbose=0)
print(list(zip(model.metrics_names, scores)))

print("Scores for Snopes:")
scores = model.evaluate(snopestext_test, snopeslabels_test, verbose=0)
print(list(zip(model.metrics_names, scores)))


####################################################################
# More complicated model below
####################################################################

def create_complex_model():
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH)

    # applying a more complex convolutional approach
    convs = []
    filter_sizes = [3, 4, 5]

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate(axis=1)(convs)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    return model


print("Pre-training complex CNN model")
model = create_complex_model()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', km.f1_score()])
model.summary()
model.fit(x_news_train, y_news_train, validation_data=(x_news_valid, y_news_valid),
          epochs=20, batch_size=128)
model.save_weights("cnn_pretrain_complex.h5")

# Use k-fold cross validation
cv_scores = []

skf = StratifiedKFold(n_splits=3)

for train_index, valid_index in skf.split(x_combined, y_combined[:, 1]):
    print("model fitting - simplified convolutional neural network")

    x_train = x_combined[train_index]
    x_valid = x_combined[valid_index]
    y_train = y_combined[train_index]
    y_valid = y_combined[valid_index]

    model = create_complex_model()
    model.load_weights("cnn_pretrain_complex.h5")
    for l in model.layers[:14]:
        l.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', km.f1_score()])

    model.summary()
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
              epochs=20, batch_size=50)

    scores = model.evaluate(x_valid, y_valid, verbose=0)
    cv_scores.append(scores)

print("Scores for CV on complex model")
print(list(zip(model.metrics_names, np.mean(cv_scores, axis=0))))


# train on the full dataset

model = create_complex_model()
model.load_weights("cnn_pretrain_complex.h5")
for l in model.layers[:14]:
    l.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', km.f1_score()])

model.summary()
model.fit(x_combined, y_combined, validation_data=(x_test, y_test),
          epochs=20, batch_size=128)

# test on heldout test and snopes data

print("Scores for Test set:")
scores = model.evaluate(x_test, y_test, verbose=0)
print(list(zip(model.metrics_names, scores)))

print("Scores for Snopes:")
scores = model.evaluate(snopestext_test, snopeslabels_test, verbose=0)
print(list(zip(model.metrics_names, scores)))
