import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import random



def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string.decode("utf-8"))
    string = re.sub(r"\'", "", string.decode("utf-8"))
    string = re.sub(r"\"", "", string.decode("utf-8"))
    return string.strip().lower()


def load_data_combined(file_name = "../data/buzzfeed-debunk-combined/all-v02.txt"):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header= None, names=["id",	"url",	"label", "data", "source", "domain"], usecols=[2,3])
    print(data_train.shape)
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    transdict = {
        'ftrue': 1,
        'mtrue': 2,
        'mixture': 3 ,
        'mfalse':4,
        'ffalse': 5,
        'pantsfire': 6,
        'nofact': 7
    }
    labels = [transdict[i] for i in labels]
    #labels = to_cat(np.asarray(labels))
    print(labels[0:6])
    return texts, labels

def balance_data(texts, labels, sample_size, discard_labels = [] , seed = 123):
    np.random.seed(seed)
    ## sample size is the number of items we want to have from EACH class
    unique, counts = np.unique(labels, return_counts=True)
    print np.asarray((unique, counts)).T
    all_index = np.empty([0], dtype=int)
    for l, f in zip(unique, counts):
        if (l in discard_labels ):
            print "Discarding items for label " , l
            continue
        l_index = (np.where( labels ==  l )[0]).tolist()  ## index of input data with current label
        if( sample_size - f > 0 ):
            print "Upsampling ", sample_size - f, " items for class ", l
            x = np.random.choice(f, sample_size - f).tolist()
            l_index = np.append(np.asarray(l_index) , np.asarray(l_index)[x])
        else:
            print "Downsampling ", sample_size , " items for class ", l
            l_index = random.sample(l_index, sample_size)
        all_index = np.append(all_index, l_index)
    bal_labels = np.asarray(labels)[all_index.tolist()]
    bal_texts = np.asarray(texts)[all_index.tolist()]
    remaining = [i for i in range(0, np.sum(counts)) if i not in all_index.tolist()]
    rem_texts = np.asarray(texts)[remaining]
    rem_labels = np.asarray(labels)[remaining]
    print "Final size of dataset:"
    unique, counts = np.unique(bal_labels, return_counts=True)
    print np.asarray((unique, counts)).T
    print "Final size of remaining dataset:"
    unique, counts = np.unique(rem_labels, return_counts=True)
    print np.asarray((unique, counts)).T
    return bal_texts, bal_labels, rem_texts, rem_labels

def news_data_summary(file_name = "../data/buzzfeed-debunk-combined/all-v02.txt"):
    df= pd.read_table(file_name, sep='\t', header=None, names=["id",	"url",	"label", "data", "source", "domain"])
    print df.shape
    #print(df.agg([]))
    print pd.crosstab(df.domain, df.label)


def news_data_sampler(texts, labels,  train_size, dev_size, test_size, seed = 123):
    print("Sampling data for seed " + str(seed))
    texts_test, labels_test, texts, labels = balance_data(texts, labels, test_size, [6, 7], seed )
    pd.DataFrame(data= [texts_test, labels_test],columns=["text", "label"]).to_pickle("../pickle/test", protocol=2)
    texts_dev, labels_dev, texts, labels = balance_data(texts, labels, dev_size, [6, 7], seed )
    pd.DataFrame(data=[texts_dev, labels_dev], columns=["text", "label"]).to_pickle("../pickle/dev", protocol=2)
    texts_train, labels_train, texts, labels = balance_data(texts, labels, train_size, [6, 7], seed )



news_data_summary()
texts, labels =  load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt")
news_data_sampler(texts, labels,  train_size = 700, dev_size = 200, test_size = 200)





