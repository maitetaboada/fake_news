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
from sklearn.ensemble import RandomForestClassifier


from textutils import DataLoading
import os
import numpy as np
from nltk.corpus import stopwords
import nltk

import pandas as pd
CLASSES = 2


from sklearn.feature_selection import SelectFromModel







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
        features = [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]



        print("In SurfaceFeatures: avg length and avg num_sentences ")
        avgLength = sum(item['length'] for item in features) / len(features)
        avgSentences = sum(item['num_sentences'] for item in features) / len(features)
        print(avgLength)
        print(avgSentences)
        print(features[0:10])

        return features

class LiwcFeatures(BaseEstimator, TransformerMixin):

    liwcDic = {} #= map of text to dataframe row (this dataframe should be read from the file including liwc features)

    def __init__(self):
        liwcFile = "/Users/fa/workspace/shared/sfu/fake_news/data/lwic/vocabliwc_cats.csv"
        cols = list(pd.read_csv(liwcFile, nrows=1))
        df = pd.read_csv(liwcFile, index_col="Source (A)",
                         usecols =[i for i in cols if i not in ['WC',	'Analytic',	'Clout'	,'Authentic',	'Tone',	'WPS']])
        df = df.T
        #df = df.replace(0, np.nan).replace(100, 1).to_sparse()
        keys = df.index
        df = df.reset_index().drop('index', axis ='columns')
        #print(df[0:10])
        cols = df.columns
        values = df.apply(lambda x: x > 0).apply(lambda x: list(cols[x.values]), axis=1)
        self.liwcDic = dict(zip(keys, values))
        print("Testing the dictionary made for liwc feature 'function':")
        print(self.liwcDic['function'])



    def fit(self, x, y=None):
        return self


    def transform(self, texts):
        #find the line related to this text in dataframe posts[['feature1','feature2',...,'feature100']].to_dict('records')
        #return [self.getLiwcRow(text).to_dict('records')
        #        for text in posts]
        textvecs = []
        for text in texts:
            #print "*** Current text:\n" + text  + "\n***"
            tokens = nltk.word_tokenize(text)
            textvec = {}
            for category in self.liwcDic.keys():
                lexicon_words = [i for i in tokens if i in self.liwcDic[category]]
                count = len(lexicon_words)
                #print ("Count: " + str(count))
                count = count * 1.0 / len(tokens)  #Continous treatment
                #count = 1 if (count > 0) else 0     #Binary treatment
                textvec[category] =  count
            #print(textvec)
            textvecs.append(textvec)
        return np.array(textvecs)



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
            features[i, 0] = text[0:100]
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
            ('liwc_features', Pipeline([
                ('stats', LiwcFeatures()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]), 1),



        ],

        # weight components in ColumnTransformer
        transformer_weights={
            'subject': 0.0,
            'body_bow': 0.0,
            'surface_features': 0.9,
            'lexicon_features': 0.9,
            'liwc_features':0.9
        }
    )),

    # Feature selection
    #('feature_selection', SelectFromModel(LinearSVC(penalty="l2", dual=False,
    #                                   tol=1e-3))),

    # Use a SVC classifier on the combined features
    ('svc', LinearSVC(penalty="l2", dual=False,
                                       tol=1e-3)),

    # Use a RandomForest classifier
    #('classification', RandomForestClassifier())
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
#test = pd.read_csv("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv")
#train = pd.read_csv("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv")#"../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv")

texts_snopesCheked, labels_snopesCheked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)#load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_data_rubin()#load_data_combined("../data/buzzfeed-debunk-combined/rumor-v02.txt")#load_data_rubin()#load_data_liar("../data/liar_dataset/test.tsv")
texts_emergent, labels_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
texts_buzzfeedTop, labels_buzzfeedTop = DataLoading.load_data_buzzfeedtop()

## USE LIAR DATA FOR TRAINING A MODEL AND TEST DATA BOTH FROM LIAR AND BUZZFEED
texts_train_snopes, labels_train_snopes = DataLoading.load_data_snopes("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES )#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/rashkin/xtrain.txt")#load_data_liar("../data/liar_dataset/train.tsv")#
texts_train_buzzfeed, labels_train_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)
texts_train_emergent, labels_train_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)

len(texts_train_snopes)
len(texts_train_buzzfeed)
len(texts_train_emergent)
print("Loading data is finished!")

print("Preparing training data...")

texts_train = (pd.concat([pd.Series(texts_train_snopes) ,  pd.Series(texts_train_buzzfeed), pd.Series(texts_train_emergent)]))
labels_train = (pd.concat([pd.Series(labels_train_snopes) ,  pd.Series(labels_train_buzzfeed), pd.Series(labels_train_emergent )]))

#texts_train = (pd.concat([pd.Series(texts_train_snopes) ,   pd.Series(texts_train_buzzfeed)]))
#labels_train = (pd.concat([pd.Series(labels_train_snopes) ,   pd.Series(labels_train_buzzfeed )]))

texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_train, labels_train, 800, [2,5])
print(texts_train[0:25][0:100])
print(labels_train[0:100])


print("Fitting the model...")
pipeline.fit(texts_train, labels_train)




#Tests:

print("Test results on data sampled from same distribution as training data:")
texts_test, labels_test, texts, labels =  DataLoading.balance_data(texts, labels, 200, [2,5])
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



print("Test results on data sampled from buzzfeed dataset (a narrow distribution : US election topic -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_train_buzzfeed, labels_train_buzzfeed , 70, [2,5])
y = pipeline.predict(texts_test)
print(classification_report(y, labels_test))


