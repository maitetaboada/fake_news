import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical as to_cat
import random


def generalfunct():
    print("yes")







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

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", str(string)) #.decode())
    string = re.sub(r"\'", "", str(string)) #.decode())
    string = re.sub(r"\"", "", str(string)) #.decode())
    string = ''.join(e for e in string if (e.isspace() or e.isalnum()))  # comment the if part for Mehvish parser
    return string.strip().lower()

def load_data_liar(file_name):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["id", "label", "data"], usecols=[0, 1, 2])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'true': 0,
        'mostly-true': 0,
        'half-true': 0,
        'barely-true': 1,
        'false': 1,
        'pants-fire': 1
    }
    labels = [transdict[i] for i in labels]
    labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

def load_data_combined(file_name="../data/buzzfeed-debunk-combined/all-v02.txt", classes = 2 ):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None,
                               names=["id", "url", "label", "data", "domain", "source"], usecols=[2, 3])
    print(data_train.shape)
    print(data_train.label[0:10])
    print(data_train.label.unique())
    # print(data_train[data_train["label"].isnull()])
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        labels.append(data_train.label[idx])
    if (classes == 2):
        transdict = {
            'ftrue': 0,
            'mtrue': 0,
            'mfalse': 1,
            'ffalse': 1,

            'mixture': 4,
            'pantsfire': 5,
            'nofact': 6
        }
    else:
        transdict = {
            'ftrue': 0,
            'mtrue': 1,
            'mixture': 2,
            'mfalse': 3,
            'ffalse': 4,

            'pantsfire': 5,
            'nofact': 6
        }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print(labels[0:6])
    return texts, labels

def load_data_rashkin(file_name="../data/rashkin/train.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["label", "data"], usecols=[0, 1],
                               dtype={"label": np.str, "data": np.str})
    print(data_train.shape)
    print(data_train[0:6])
    texts = []
    labels = []
    # for i in range(data_train.data.shape[0]):
    #   print(i, type(data_train.data[i]))
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(str(data_train.label[idx]))
    transdict = {
        '1': 2,  # Satire
        '2': 3,  # Hoax
        '3': 1,  # Propaganda
        '4': 0  # Truested
    }
    labels = [transdict[i] for i in labels]
    labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

def load_data_buzzfeed(file_name="../data/buzzfeed-facebook/bf_fb.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["ID", "URL", "label", "data", "error"],
                               usecols=[2, 3])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'no factual content': 0,
        'mostly true': 1,
        'mixture of true and false': 2,
        'mostly false': 3
    }
    labels = [transdict[i] for i in labels]
    labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

def balance_data(texts, labels, sample_size, discard_labels=[]):
    ## sample size is the number of items we want to have from EACH class
    unique, counts = np.unique(labels, return_counts=True)
    print(np.asarray((unique, counts)).T)
    all_index = np.empty([0], dtype=int)
    for l, f in zip(unique, counts):
        if (l in discard_labels):
            print ("Discarding items for label " + str(l))
            continue
        l_index = (np.where(labels == l)[0]).tolist()  ## index of input data with current label
        if (sample_size - f > 0):
            # print "Upsampling ", sample_size - f, " items for class ", l
            x = np.random.choice(f, sample_size - f).tolist()
            l_index = np.append(np.asarray(l_index), np.asarray(l_index)[x])
        else:
            # print "Downsampling ", sample_size , " items for class ", l
            l_index = random.sample(l_index, sample_size)
        all_index = np.append(all_index, l_index)
    bal_labels = np.asarray(labels)[all_index.tolist()]
    bal_texts = np.asarray(texts)[all_index.tolist()]
    remaining = [i for i in range(0, np.sum(counts)) if i not in all_index.tolist()]
    rem_texts = np.asarray(texts)[remaining]
    rem_labels = np.asarray(labels)[remaining]
    print ("Final size of dataset:")
    unique, counts = np.unique(bal_labels, return_counts=True)
    print (np.asarray((unique, counts)).T)
    print ("Final size of remaining dataset:")
    unique, counts = np.unique(rem_labels, return_counts=True)
    print (np.asarray((unique, counts)).T)
    return bal_texts, bal_labels, rem_texts, rem_labels



