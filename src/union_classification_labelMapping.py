

## Used ideas from https://www.kaggle.com/metadist/work-like-a-pro-with-pipelines-and-feature-unions
## Used ideas from https://www.kaggle.com/edolatabadi/feature-union-with-grid-search
## Used ideas from https://github.com/scikit-learn/scikit-learn/issues/6122 feature selection output



from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


from textutils import DataLoading
import os
import numpy as np
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


#Reproducibility
import random
np.random.seed(0)
random.seed(0)



LOAD_DATA_FROM_DISK = True
CLASSES = 2

#################

print("Preparing lexicons & lwicDic")
lexicon_directory = "../data/bias_related_lexicons"
lexicons = []
lexiconNames = []
#print("LexiconFeatures() init: loading lexicons")
for filename in os.listdir(lexicon_directory):
    if filename.endswith(".txt"):
        file = os.path.join(lexicon_directory, filename)
        words = open(file, encoding = "ISO-8859-1").read()
        lexicon = {k: 0 for k in nltk.word_tokenize(words)}
        print("Number of terms in the lexicon " + filename + " : " + str(len(lexicon)))
        lexicons.append(lexicon)
        lexiconNames.append(filename)
        continue
    else:
        continue

liwcDic = {}
liwcFile = "../data/lwic/vocabliwc_cats.csv"
cols = list(pd.read_csv(liwcFile, nrows=1))
df = pd.read_csv(liwcFile, index_col="Source (A)",
                 usecols=[i for i in cols if i not in ['WC', 'Analytic', 'Clout', 'Authentic', 'Tone', 'WPS']])
df = df.T
keys = df.index
df = df.reset_index().drop('index', axis='columns')
cols = df.columns
values = df.apply(lambda x: x > 0).apply(lambda x: list(cols[x.values]), axis=1)
liwcDic = dict(zip(keys, values))

#####################


class PosTagFeatures(BaseEstimator, TransformerMixin):
    #normalise = True - devide all values by a total number of tags in the sentence
    #tokenizer - take a custom tokenizer function
    def __init__(self, tokenizer=lambda x: x.split(), normalize=True):
        print("Inside the init function of PosTagFeatures")
        self.tokenizer=tokenizer
        self.normalize=normalize

    #helper function to tokenize and count parts of speech
    def pos_func(self, sentence):
        return Counter(tag for word,tag in nltk.pos_tag(self.tokenizer(sentence), tagset='universal'))

    # fit() doesn't do anything, this is a transformer class
    def fit(self, X, y = None):
        return self

    #all the work is done here
    def transform(self, X):
        X = pd.Series(X)
        X_tagged = X.apply(self.pos_func).apply(pd.Series).fillna(0)
        X_tagged['n_tokens'] = X_tagged.apply(sum, axis=1)
        if self.normalize:
            X_tagged = X_tagged.divide(X_tagged['n_tokens'], axis=0)

        return X_tagged



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
        #print(avgLength)
        #print(avgSentences)
        #print(features[0:10])

        return features

class LiwcFeatures(BaseEstimator, TransformerMixin):

    liwcDic = {} #= map of text to dataframe row (this dataframe should be read from the file including liwc features)

    def __init__(self):
        '''
        liwcFile = "../data/lwic/vocabliwc_cats.csv"
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
        #print("Testing the dictionary made for liwc feature 'function':")
        #print(self.liwcDic['function'])
        '''
        self.liwcDic = liwcDic

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
        #lexicon_directory = "../data/bias_related_lexicons"
        self.lexicons = lexicons
        self.lexiconNames = lexiconNames
        #print("LexiconFeatures() init: loading lexicons")
        '''
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
        '''

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
            #('subject', TfidfVectorizer(sublinear_tf=True, max_df=0.5,
            #                     stop_words='english', ngram_range=(1,4)), 0),

            # Pipeline for standard bag-of-words model for body (second column)
            ('body_bow', TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english', ngram_range=(1,2)), 1),

            #('pos_features', PosTagFeatures(tokenizer=nltk.word_tokenize), 0 ),
            #('body_bow', Pipeline([
            #    ('tfidf', TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df= 5,
            #                     stop_words='english', ngram_range=(1,3))
            #     ),
            #    ('best', TruncatedSVD(n_components=10))
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
            ]), 1)



        ],

        # weight components in ColumnTransformer
        transformer_weights={
            #'subject': 0.5,
            'body_bow': 0.8,
            'surface_features': 0.0,
            'lexicon_features': 0.1,
            'liwc_features':0.1
            #'pos_features':0.1
        }
    )),

    # Feature selection
    #('feature_selection', SelectFromModel(LinearSVC(penalty="l2", dual=False,
    #                                   tol=1e-3))),

    # Use a SVC classifier on the combined features
    ('svc', LinearSVC(penalty= "l2", dual=False, tol=1e-3)),

    # Use a RandomForest classifier
    #('classification', RandomForestClassifier())
])



if LOAD_DATA_FROM_DISK:
	texts_train = np.load("../dump/trainRaw_rashkin")
	texts_test = np.load("../dump/testRaw")
	labels_train = np.load("../dump/trainlRaw_rashkin")
	labels_test = np.load("../dump/testlRaw")
	print("Data loaded from disk!")

else:
	# Data sources used for training:
	texts_train_rashkin, labels_train_rashkin = DataLoading.load_data_rashkin("../data/rashkin/xtrain.txt", CLASSES )#load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/r$
	print(len(texts_train_rashkin))
	texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_train_rashkin, labels_train_rashkin, 4000, [4])
	texts_train.dump("../dump/trainRaw_rashkin")
	labels_train.dump("../dump/trainlRaw_rashkin")
	print("Data dumped to disk!")

print("Size of train and test sets: " + str(len(labels_train)) + " , "  +  str(len(labels_test)))
texts_train_valid = texts_train
labels_train_valid = labels_train
print(texts_train_valid[0:3][0:10])
print(labels_train_valid[0:3])


'''
#Train:

parameters = {
    'union__body_bow__ngram_range':  ((1, 2),(1,3)),
    'union__body_bow__min_df': (3, 5),
    #'union__transformer_weights': (dict('lwic_features'=.1, 'lexicon_features'=.1, 'surface_features'=.1, 'body_bow'=.7 )),
    'svc__penalty':("l2", "l1"),
    'svc__tol': (1e-2, 1e-3)
}

grid_search = GridSearchCV(pipeline, parameters, verbose=True, scoring="f1")


print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)
grid_search.fit(texts_train_valid, labels_train_valid)
#train = grid_search.transform(texts_train)
#print(train.shape)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

joblib.dump(grid_search.best_estimator_, '../dump/bestSVClassifier.pkl', compress = 1)
'''

#Load from disk:
grid_search = joblib.load('../dump/bestSVClassifier.pkl')



#Tests:
print("Results on training data:")
y = grid_search.predict(texts_train)
print(classification_report(labels_train,   y))

print("Results on test data:")
y = grid_search.predict(texts_test)
print(classification_report( labels_test,  y))


# Data sources used for testing:
texts_snopesChecked, labels_snopesChecked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)#load_data_combined("../data/buzzfeed-debunk-combined/buzzfeed-v02.txt")#load_d$
texts_emergent, labels_emergent = DataLoading.load_data_emergent("../data/emergent/url-versions-2015-06-14.csv", CLASSES)
texts_buzzfeed, labels_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/bf_fb.txt", CLASSES)
#texts_buzzfeedTop, labels_buzzfeedTop = DataLoading.load_data_buzzfeedtop()

print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_snopesChecked, labels_snopesChecked ,sample_size=None , discard_labels=[2,5])
y = grid_search.predict(texts_test)
print(classification_report(labels_test, y))
print("confusion matrix:")
print(confusion_matrix(labels_test, y))
print(pd.DataFrame({'Predicted': y, 'Expected': labels_test}))

print("Test results on data sampled from emergent dataset (a broad distribution acc. to topic modeling -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_emergent, labels_emergent , 300, [2,5])
y = grid_search.predict(texts_test)
print(classification_report(labels_test,y))

print("Test results on data sampled from buzzfeed dataset (a narrow distribution : US election topic -- possibly some overlapping claims):")
texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_buzzfeed, labels_buzzfeed , 70, [2,5])
y = grid_search.predict(texts_test)
print(classification_report(labels_test, y))



