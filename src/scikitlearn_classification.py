## Taken from: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html


# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from bs4 import BeautifulSoup
import re
import pandas as pd
#import pickle
from textutils import DataLoading




LOAD_DATA_FROM_DISK = False
CLASSES = 2



# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16, #default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


#texts_test1, labels_test1 = load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_rubin()#load_data_liar("../data/liar_dataset/test.tsv")
## USE LIAR DATA FOR TRAINING A MODEL AND TEST DATA BOTH FROM LIAR AND BUZZFEED
#texts_train, labels_train = load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/rashkin/xtrain.txt")#load_data_liar("../data/liar_dataset/train.tsv")#
##texts_valid, labels_valid = load_data_liar("../data/liar_dataset/valid.tsv")
#texts_test1, labels_test1 = load_data_rashkin("../data/rashkin/balancedtest.txt")



#texts, labels =  DataLoading.load_data_combined(classes = CLASSES, file_name = "../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")
#texts_test1, labels_test1 = np.asarray(texts), np.asarray(labels)





if LOAD_DATA_FROM_DISK:
    texts_train = np.load("../dump/trainAnon")#np.load("../dump/trainRaw")
    texts_valid = np.load("../dump/validAnon")#np.load("../dump/validRaw")
    texts_test1 = np.load("../dump/testAnon")#np.load("../dump/testRaw")
    labels_train = np.load("../dump/trainlRaw")
    labels_valid = np.load("../dump/validlRaw")
    labels_test1 = np.load("../dump/testlRaw")

    print("Data loaded from disk!")

else:
    texts, labels =  DataLoading.load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt") #load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt")
    #texts, labels = DataLoading.load_data_snopes(file_name="../data/snopes/snopes_leftover_v00_ready.csv", classes=2)
    print("Maximum string length:")
    mylen = np.vectorize(len)
    print (mylen(texts))

    checked_texts, checked_labels = DataLoading.load_data_snopes(file_name="../data/snopes/snopes_checked_v00_ready.csv", classes=2)
    print("Maximum string length:")
    mylen = np.vectorize(len)
    print(mylen(checked_texts))


    if( CLASSES == 2 ):
        #texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts, labels, 400, [2,3,4,5,6])
        #texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 400, [2,3,4,5,6])
        #texts_train, labels_train, texts, labels = DataLoading.balance_data(texts, labels, 100, [2,3,4,5,6])
        texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts, labels, 400, [2, 3, 4, 5, 6])
        texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 400, [2, 3, 4, 5, 6])
        texts_train, labels_train, texts, labels = DataLoading.balance_data(texts, labels, 1400, [2, 3, 4, 5, 6])
        texts_test, labels_test, texts, labels = DataLoading.balance_data(checked_texts, checked_labels, 40, [2, 3, 4, 5, 6])

    else:
        texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts, labels, 200, [6,5])
        texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 200, [6,5])
        texts_train, labels_train, texts, labels = DataLoading.balance_data(texts, labels, 700, [6,5])

    texts_train.dump("../dump/trainRaw")
    texts_valid.dump("../dump/validRaw")
    texts_test.dump("../dump/testRaw")
    labels_train.dump("../dump/trainlRaw")
    labels_valid.dump("../dump/validlRaw")
    labels_test.dump("../dump/testlRaw")

    print("Data dumped to disk!")



y_train = labels_train
y_test = labels_test
target_names, counts = np.unique(y_train, return_counts= True)
print(np.asarray((target_names, counts)).T)

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if (False): #opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(texts_train)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.5,
                                 stop_words='english',ngram_range=(1,2), lowercase = True)
    X_train = vectorizer.fit_transform(texts_train)
    features = vectorizer.get_feature_names()
    print(features[100:110])
print("n_samples: %d, n_features: %d" % X_train.shape)

print("Extracting features from the test data using the same vectorizer")
X_test = vectorizer.transform(texts_test)
print("n_samples: %d, n_features: %d" % X_test.shape)


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()
#opts.select_chi2 = 1000
if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()
if feature_names:
    feature_names = np.asarray(feature_names)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."





## Making sense of the tf-idf matrix:

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

features = vectorizer.get_feature_names()

indices = np.argsort(vectorizer.idf_)[::-1]
top_n = 40
top_features = [features[i] for i in indices[:top_n]]
print(top_features)

dfs = top_feats_by_class(X_train, y_train, features, min_tfidf=0.2, top_n=40)
print(dfs)
#plot_tfidf_classfeats_h(dfs)








# #############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")

    print("Sample training vectors:")
    #print(X_train[0])
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
    precision = metrics.precision_score(y_test, pred, average = 'macro')
    recall = metrics.recall_score(y_test, pred, average = 'macro')
    f1 = metrics.f1_score(y_test, pred, average = 'macro')
    mse = metrics.mean_squared_error(y_test, pred)

    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if (False):
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
    if (True):
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
        print(pd.DataFrame({'Predicted': pred, 'Expected': y_test}))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, trainScore, score, precision, recall, f1, mse


results = []

'''
for clf, name in (
        #(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        #(Perceptron(n_iter=50), "Perceptron"),
        #(PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        #(KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))
'''

for penalty in ["l2"]:#, "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))


'''
# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

'''

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

'''
print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))
  
'''

print("="*100)
print(results)


'''

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

'''