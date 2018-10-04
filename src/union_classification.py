# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC


from textutils import DataLoading
import os
import numpy as np
from nltk.corpus import stopwords
import nltk


class SurfaceFeatures(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    XXX = None

    def __init__(self):
        self.XXX = "Inside the init function of SurfaceFeatures()"
        print(self.XXX)

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        #posts[['feature1','feature2',...,'feature100']].to_dict('records')[0].to_dict('records')
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]

class LiwcFeatures(BaseEstimator, TransformerMixin):

    liwcDic = None #= map of text to dataframe row (this dataframe should be read from the file including liwc features)
    def __init__(self, liwcDataframe):
        self.liwcDic  = 0 # a map of text to row number in liwcDataframe

    def fit(self, x, y=None):
        return self

    def getLiwcRow(self, text):
        return 0

    def transform(self, posts):
        #find the line related to this text in dataframe posts[['feature1','feature2',...,'feature100']].to_dict('records')
        return [self.getLiwcRow(text).to_dict('records')
                for text in posts]


class LexiconFeatures(BaseEstimator, TransformerMixin):
    lexicons = None
    lexiconNames = None


    def __init__(self):
        print("Inside the init function of LexiconFeatures(0")
        lexicon_directory = "/Users/fa/workspace/shared/sfu/fake_news/data/bias_related_lexicons"
        self.lexicons = []
        self.lexiconNames = []
        print("LexiconFeatures() init: loading lexicons")
        for filename in os.listdir(lexicon_directory):
            if filename.endswith(".txt"):
                file = os.path.join(lexicon_directory, filename)
                words = open(file, encoding = "ISO-8859-1").read()
                lexicon = {k: 0 for k in nltk.word_tokenize(words)}
                print("Number of terms in the lexicon " + filename + " : " + str(len(lexicon)))
                self.lexicons.append(lexicon)
                self.lexiconNames.append(filename)
                continue
            else:
                continue


    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        stoplist = stopwords.words('english')
        #print("transforming:...")
        textvecs = []
        for text in texts:
            #print "*** Current text:\n" + text  + "\n***"
            tokens = nltk.word_tokenize(text)
            textvec = {}
            for lexicon, lexiconName in zip(self.lexicons, self.lexiconNames):
                lexicon_words = [i for i in tokens if i in lexicon]
                count = len(lexicon_words)
                #print ("Count: " + str(count))
                count = count * 1.0 / len(tokens)  #Continous treatment
                #count = 1 if (count > 0) else 0     #Binary treatment
                textvec[lexiconName] =  count
            #print(textvec)
            textvecs.append(textvec)
        return np.array(textvecs)

class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # construct object dtype array with two columns
        # first column = 'subject' and second column = 'body'
        features = np.empty(shape=(len(posts), 2), dtype=object)
        for i, text in enumerate(posts):
            features[i, 0] = text[0:300]
            features[i, 1] = text
        return features


pipeline = Pipeline([
    # Extract the subject & body
    ('subjectbody', SubjectBodyExtractor()),

    # Use ColumnTransformer to combine the features from subject and body
    ('union', ColumnTransformer(
        [
            # Pulling features from the post's subject line (first column)
            ('subject', TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', ngram_range=(1,3)), 0),

            # Pipeline for standard bag-of-words model for body (second column)
            ('body_bow', TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', ngram_range=(1,3)), 1),


            #('body_bow', Pipeline([
            #    ('tfidf', TfidfVectorizer()),
            #    ('best', TruncatedSVD(n_components=10)),
            #]), 1),

            # Pipeline for pulling ad hoc features from post's body
            ('surface_features', Pipeline([
                ('stats', SurfaceFeatures()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]), 1),
            ('lexicon_features', Pipeline([
                ('stats', LexiconFeatures()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]), 1),
        ],

        # weight components in ColumnTransformer
        transformer_weights={
            'subject': 0.0,
            'body_bow': 0.8,
            'surface_features': 0.8,
            'lexicon_features': 0.8
        }
    )),

    # Use a SVC classifier on the combined features
    ('svc', LinearSVC(penalty="l2", dual=False,
                                       tol=1e-3)),
])

'''
# limit the list of categories to make running this example faster.
categories = ['alt.atheism', 'talk.religion.misc']
train = fetch_20newsgroups(random_state=1,
                           subset='train',
                           categories=categories,
                           )
test = fetch_20newsgroups(random_state=1,
                          subset='test',
                          categories=categories,
                          )
'''
import pandas as pd
CLASSES = 2

#test = pd.read_csv("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv")
#train = pd.read_csv("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv")#"../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv")

texts_snopesCheked, labels_snopesCheked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)#load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_data_rubin()#load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_rubin()#load_data_liar("../data/liar_dataset/test.tsv")
texts_emergent, labels_emergent = DataLoading.load_data_emergent()
texts_buzzfeedTop, labels_buzzfeedTop = DataLoading.load_data_buzzfeedtop()

## USE LIAR DATA FOR TRAINING A MODEL AND TEST DATA BOTH FROM LIAR AND BUZZFEED
texts_train_snopes, labels_train_snopes = DataLoading.load_data_snopes("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES )#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/rashkin/xtrain.txt")#load_data_liar("../data/liar_dataset/train.tsv")#
texts_train_buzzfeed, labels_train_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)

len(texts_train_snopes)
len(texts_train_buzzfeed)

texts_train = (pd.concat([pd.Series(texts_train_snopes) ,  pd.Series(texts_train_buzzfeed)]))
labels_train = (pd.concat([pd.Series(labels_train_snopes) ,  pd.Series(labels_train_buzzfeed)]))


texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_train, labels_train, 1400, [2,5])
#texts_test1, labels_test1, texts, labels =  DataLoading.balance_data(texts, labels, 40, [2,5])
#texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts_snopesCheked, labels_snopesCheked , 40, [2,5])
#texts_test1, labels_test1, texts, labels = DataLoading.balance_data(texts_emergent, labels_emergent , 300, [2,5])


pipeline.fit(texts_train, labels_train)

#Tests:

print("Test results on data sampled from same distribution (snopes + buzzfeed):")
texts_test, labels_test, texts, labels =  DataLoading.balance_data(texts, labels, 40, [2,5])
y = pipeline.predict(texts_test)
print(classification_report(y, labels_test))


print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_snopesCheked, labels_snopesCheked , 40, [2,5])
y = pipeline.predict(texts_test)
print(classification_report(y, labels_test))


print("Test results on data sampled from emergent dataset (a broad distribution acc. to topic modeling -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_emergent, labels_emergent , 300, [2,5])
y = pipeline.predict(texts_test)
print(classification_report(y, labels_test))
