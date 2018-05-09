## Base code taken from: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html


from __future__ import print_function

import pickle
from time import time

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import density

from web.src.textutils import DataLoading

LOAD_DATA_FROM_DISK = True
CLASSES = 2

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")

    print("Sample training vectors:")
    # print(X_train[0])
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    trainScore = metrics.accuracy_score(y_train, clf.predict(X_train))
    score = metrics.accuracy_score(y_test, pred)
    precision = metrics.precision_score(y_test, pred, average='macro')
    recall = metrics.recall_score(y_test, pred, average='macro')
    f1 = metrics.f1_score(y_test, pred, average='macro')
    mse = metrics.mean_squared_error(y_test, pred)

    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    print(pd.DataFrame({'Predicted': pred, 'Expected': y_test}))

    print()
    clf_descr = str(clf).split('(')[0]
    print(str([clf_descr, trainScore, score, precision, recall, f1, mse]))
    return clf



### PREPARE


# texts_test1, labels_test1 = load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_rubin()#load_data_liar("../data/liar_dataset/test.tsv")
## USE LIAR DATA FOR TRAINING A MODEL AND TEST DATA BOTH FROM LIAR AND BUZZFEED
# texts_train, labels_train = load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/rashkin/xtrain.txt")#load_data_liar("../data/liar_dataset/train.tsv")#
##texts_valid, labels_valid = load_data_liar("../data/liar_dataset/valid.tsv")
# texts_test1, labels_test1 = load_data_rashkin("../data/rashkin/balancedtest.txt")

def prepare_data(data = "combined"):

    if (data == "rashkin"):

        texts_train, labels_train = DataLoading.load_data_rashkin("../data/rashkin/xtrain.txt")
        texts_valid, labels_valid = DataLoading.load_data_rashkin("../data/rashkin/xdev.txt")
        texts_test1, labels_test1 = DataLoading.load_data_rashkin("../data/rashkin/balancedtest.txt")

    else:
        if LOAD_DATA_FROM_DISK:
            texts_train = np.load("../dump/trainRaw")
            texts_valid = np.load("../dump/validRaw")
            texts_test1 = np.load("../dump/testRaw")
            labels_train = np.load("../dump/trainlRaw")
            labels_valid = np.load("../dump/validlRaw")
            labels_test1 = np.load("../dump/testlRaw")

            print("Data loaded from disk!")

        else:
            texts, labels = DataLoading.load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt")
            print("Maximum string length:")
            mylen = np.vectorize(len)
            print(mylen(texts))

            if (CLASSES == 2):
                texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts, labels, 400, [2, 3, 4, 5, 6])
                texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 400, [2, 3, 4, 5, 6])
                texts_train, labels_train, texts, labels = DataLoading.balance_data(texts, labels, 1400, [2, 3, 4, 5, 6])

            else:
                texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts, labels, 200, [6, 5])
                texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 200, [6, 5])
                texts_train, labels_train, texts, labels = DataLoading.balance_data(texts, labels, 700, [6, 5])

            texts_train.dump("../dump/trainRaw")
            texts_valid.dump("../dump/validRaw")
            texts_test1.dump("../dump/testRaw")
            labels_train.dump("../dump/trainlRaw")
            labels_valid.dump("../dump/validlRaw")
            labels_test1.dump("../dump/testlRaw")

            print("Data dumped to disk!")

    y_train = labels_train
    y_test = labels_test1
    target_names, counts = np.unique(y_train, return_counts=True)
    print(np.asarray((target_names, counts)).T)
    print("Extracting features from the training data using a sparse vectorizer")
    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.5,
                                 stop_words='english', ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(texts_train)
    features = vectorizer.get_feature_names()
    print(features[300000:300200])
    print("n_samples: %d, n_features: %d" % X_train.shape)

    print("Extracting features from the test data using the same vectorizer")
    X_test = vectorizer.transform(texts_test1)
    print("n_samples: %d, n_features: %d" % X_test.shape)

    return X_train, y_train, X_test, y_test, vectorizer




'''
##*********************************** MISINFORMATION ********************************##

### PREPARE DATA
print("\n\nPREPARE DATA\n\n")
X_train, y_train, X_test, y_test, vectorizer = prepare_data("combined")

### TRAIN and SAVE
print("\n\nTRAIN and SAVE\n\n")
clf = benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"), X_train, y_train, X_test, y_test)
pickle.dump(clf, open("./models/tf-idf_SGD_misinformation_binary.pkl", 'wb'))
pickle.dump(vectorizer, open("./models/tf-idf_vectorizer_misinformation_binary.pkl", 'wb'))

#### LOAD and REUSE ###
print("\n\nLOAD and REUSE\n\n")
vectorizer = pickle.load(open("./models/tf-idf_vectorizer_misinformation_binary.pkl", 'rb'))
print("Extracting features from the test data using the same vectorizer")
text = "This is a test text"
text = BeautifulSoup(text)
text = DataLoading.clean_str(text.get_text().encode('ascii', 'ignore'))
X_test = vectorizer.transform([text])
print("n_samples: %d, n_features: %d" % X_test.shape)
clf = pickle.load(open("./models/tf-idf_SGD_misinformation_binary.pkl", 'rb'))
pred = clf.predict(X_test)
print("Prediction for this input is:")
print(pred)





##*********************************** RASHKIN GENERA ********************************##

### PREPARE DATA
print("\n\nPREPARE DATA\n\n")
X_train, y_train, X_test, y_test, vectorizer = prepare_data("rashkin")

### TRAIN and SAVE
print("\n\nTRAIN and SAVE\n\n")
clf = benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"), X_train, y_train, X_test, y_test)
pickle.dump(clf, open("./models/tf-idf_SGD_genera_4-way.pkl", 'wb'))
pickle.dump(vectorizer, open("./models/tf-idf_vectorizer_genera_4-way.pkl", 'wb'))

'''


#### LOAD and REUSE ###
print("\n\nLOAD and REUSE\n\n")
vectorizer = pickle.load(open("./models/tf-idf_vectorizer_genera_4-way.pkl", 'rb'))
print("Extracting features from the test data using the same vectorizer")
text = "Clinton election day president 2016 2015    "
text = BeautifulSoup(text)
text = DataLoading.clean_str(text.get_text().encode('ascii', 'ignore'))
X_test = vectorizer.transform([text])
print("n_samples: %d, n_features: %d" % X_test.shape)
clf = pickle.load(open("./models/tf-idf_SGD_genera_4-way.pkl", 'rb'))
pred = clf.predict(X_test)
print("Prediction for this input is:")
print(pred)

