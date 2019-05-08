
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
from sklearn.model_selection import cross_validate

from textutils import DataLoading
import os
import numpy as np
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import string
import textstat
import textblob

#################

print("Preparing lexicons & lwicDic")
lexicon_directory = "../data/bias_related_lexicons"
lexicons = []
lexiconNames = []
# print("LexiconFeatures() init: loading lexicons")
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
    pos_family = {}

    def __init__(self):
        print("Inside the init function of PosTagFeatures()")

    # fit() doesn't do anything, this is a transformer class
    def fit(self, texts, y=None):
        return self

    # all the work is done here
    def transform(self, texts):
        allTags = ['NOUN', 'PRON', 'ADJ', 'ADV', 'VERB', 'ADP', 'NUM', 'PRT', 'DET', 'X', 'CONJ', '.']
        tokenizer = lambda x: x.split()
        features = [dict(Counter(allTags + [tag for word, tag in nltk.pos_tag(tokenizer(text), tagset='universal')]))
                    for text in texts]
        # normalize by the number of all tags (words in the text + 12 smoothing factor)
        features = [{key: val / (len(text.split()) + 12) for key, val in d.items()} for text, d in zip(texts, features)]
        features = np.array(features)
        return features


class SurfaceFeatures(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    XXX = None

    def __init__(self):
        self.XXX = "Inside the init function of SurfaceFeatures()"
        print(self.XXX)

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # posts[['feature1','feature2',...,'feature100']].to_dict('records')[0].to_dict('records')
        features = [{'num_char': len(text),
                     'num_sentence': text.count('.'),
                     'num_punc/num_char': len("".join(_ for _ in text if _ in string.punctuation)) / (len(text) + 1),
                     'num_upper/num_char': len([wrd for wrd in text.split() if wrd.isupper()]) / (len(text) + 1),
                     'num_word/num_sentence': len(text.split()) / (text.count('.') + 1)
                     }
                    for text in posts]
        return features


class LiwcFeatures(BaseEstimator, TransformerMixin):
    liwcDic = {}  # = map of text to dataframe row (this dataframe should be read from the file including liwc features)

    def __init__(self):
        print("Inside the init function of LiwcFeatures()")
        self.liwcDic = liwcDic

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        textvecs = []
        for text in texts:
            # print "*** Current text:\n" + text  + "\n***"
            tokens = nltk.word_tokenize(text)
            textvec = {}
            for category in self.liwcDic.keys():
                lexicon_words = [i for i in tokens if i in self.liwcDic[category]]
                count = len(lexicon_words)
                count = count * 1.0 / len(tokens)  # Continous treatment
                # count = 1 if (count > 0) else 0     #Binary treatment
                textvec[category] = count
            textvecs.append(textvec)
        textvecs = np.array(textvecs)
        return textvecs


class LexiconFeatures(BaseEstimator, TransformerMixin):
    lexicons = None
    lexiconNames = None

    def __init__(self):
        print("Inside the init function of LexiconFeatures()")
        self.lexicons = lexicons
        self.lexiconNames = lexiconNames

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        stoplist = stopwords.words('english')
        textvecs = []
        for text in texts:
            tokens = nltk.word_tokenize(text)
            textvec = {}
            for lexicon, lexiconName in zip(self.lexicons, self.lexiconNames):
                lexicon_words = [i for i in tokens if i in lexicon]
                count = len(lexicon_words)
                count = count * 1.0 / len(tokens)  # Continous treatment
                # count = 1 if (count > 0) else 0     #Binary treatment
                textvec[lexiconName] = count
            textvecs.append(textvec)
        textvecs = np.array(textvecs)
        return textvecs


class ReadabilityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Inside the init function of readabilityFeatures()")

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # posts[['feature1','feature2',...,'feature100']].to_dict('records')[0].to_dict('records')
        features = [{'flesch_reading_ease': textstat.flesch_reading_ease(text),
                     'smog_index': textstat.smog_index(text),
                     'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                     'coleman_liau_index': textstat.coleman_liau_index(text),
                     'automated_readability_index': textstat.automated_readability_index(text),
                     'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
                     'linsear_write_formula': textstat.linsear_write_formula(text),
                     'gunning_fog': textstat.gunning_fog(text)
                     # 'text_standard': textstat.text_standard(text)
                     }
                    for text in posts]
        return features


class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
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



#################################

LOAD_DATA_FROM_DISK = True
CLASSES = 2

if LOAD_DATA_FROM_DISK:
    texts_train = np.load("../dump/trainRaw")
    texts_valid = np.load("../dump/validRaw")
    texts_test = np.load("../dump/testRaw")
    labels_train = np.load("../dump/trainlRaw")
    labels_valid = np.load("../dump/validlRaw")
    labels_test = np.load("../dump/testlRaw")
    print("Data loaded from disk!")
    print(texts_train.shape)

else:
    # Data sources used for training:
    texts_snopes, labels_snopes = DataLoading.load_data_snopes \
        ("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES  )  # load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/r$
    texts_buzzfeed, labels_buzzfeed = DataLoading.load_data_buzzfeed("../data/buzzfeed-facebook/buzzfeed-v02-originalLabels.txt",
                                                                                 CLASSES)
    texts_emergent, labels_emergent = DataLoading.load_data_emergent(
        "../data/emergent/url-versions-2015-06-14.csv", CLASSES)
    len(texts_snopes)
    len(texts_buzzfeed)
    len(texts_emergent)

    texts_snopes, labels_snopes, texts, labels = DataLoading.balance_data(texts_snopes, labels_snopes, 259, [2, 3, 4, 5])
    texts_buzzfeed, labels_buzzfeed, texts, labels = DataLoading.balance_data(texts_buzzfeed, labels_buzzfeed, 64, [2, 3, 4, 5])
    texts_emergent, labels_emergent, texts, labels = DataLoading.balance_data(texts_emergent,labels_emergent, 0, [2, 3, 4, 5])

    texts_all = pd.concat(
        [pd.Series(texts_snopes), pd.Series(texts_buzzfeed), pd.Series(texts_emergent)])
    labels_all = pd.concat(
        [pd.Series(labels_snopes), pd.Series(labels_buzzfeed), pd.Series(labels_emergent)])


    texts_train, labels_train, texts, labels = DataLoading.balance_data(texts_all, labels_all, 682,
                                                                        [2, 3, 4, 5])
    texts_valid, labels_valid, texts, labels = DataLoading.balance_data(texts, labels, 0  , [2, 3, 4, 5])
    texts_test, labels_test, texts, labels = DataLoading.balance_data(texts, labels, 0, [2, 3, 4, 5])
    texts_train.dump("../dump/trainRaw")
    texts_valid.dump("../dump/validRaw")
    texts_test.dump("../dump/testRaw")
    labels_train.dump("../dump/trainlRaw")
    labels_valid.dump("../dump/validlRaw")
    labels_test.dump("../dump/testlRaw")
    print("Data dumped to disk!")

print("Size of train, validataion and test sets: " + str(len(labels_train)) + " , " + str(
    len(labels_valid)) + " , " + str(len(labels_test)))
texts_train_valid = pd.concat(
    [pd.Series(texts_train), pd.Series(texts_valid)])
labels_train_valid = pd.concat(
    [pd.Series(labels_train), pd.Series(labels_valid)])
print(texts_train_valid[0:3][0:10])
print(labels_train_valid[0:3])



# Train & Test Loop

tws = [
    {'body_bow': 0.2,
            'surface_features': 0.0,
            'lexicon_features': 0.2,
            'liwc_features' :0.2,
            'pos_features':0.2,
            'readability_features': 0.0},
    {'body_bow': 0.2,
            'surface_features': 0.2,
            'lexicon_features': 0.2,
            'liwc_features' :0.0,
            'pos_features':0.2,
            'readability_features': 0.0},
    {'body_bow': 0.2,
     'surface_features': 0.2,
     'lexicon_features': 0.2,
     'liwc_features': 0.0,
     'pos_features': 0.0,
     'readability_features': 0.0},
    {'body_bow': 0.2,
     'surface_features': 0.2,
     'lexicon_features': 0.0,
     'liwc_features': 0.0,
     'pos_features': 0.2,
     'readability_features': 0.0},
    {'body_bow': 0.2,
     'surface_features': 0.0,
     'lexicon_features': 0.2,
     'liwc_features': 0.0,
     'pos_features': 0.2,
     'readability_features': 0.0},
    {'body_bow': 0.2,
     'surface_features': 0.2,
     'lexicon_features': 0.2,
     'liwc_features': 0.2,
     'pos_features': 0.2,
     'readability_features': 0.0},
    {'body_bow': 0.2,
     'surface_features': 0.2,
     'lexicon_features': 0.2,
     'liwc_features': 0.2,
     'pos_features': 0.0,
     'readability_features': 0.0},
    {'body_bow': 0.2,
            'surface_features': 0.0,
            'lexicon_features': 0.2,
            'liwc_features' :0.2,
            'pos_features':0.0,
            'readability_features': 0.0},
    {'body_bow': 0.2,
        'surface_features': 0.2,
        'lexicon_features': 0.2,
        'liwc_features' :0.2,
        'pos_features':0.2,
        'readability_features': 0.2},
    {'body_bow': 0.2,
     'surface_features': 0.0,
     'lexicon_features': 0.0,
     'liwc_features': 0.0,
     'pos_features': 0.0,
     'readability_features': 0.0},
    {'body_bow': 0.0,
        'surface_features': 0.2,
        'lexicon_features': 0.0,
        'liwc_features' :0.0,
        'pos_features':0.0,
        'readability_features': 0.0},
    {'body_bow': 0.0,
        'surface_features': 0.0,
        'lexicon_features': 0.2,
        'liwc_features' :0.0,
        'pos_features':0.0,
        'readability_features': 0.0},
    {'body_bow': 0.0,
        'surface_features': 0.0,
        'lexicon_features': 0.0,
        'liwc_features' :0.2,
        'pos_features':0.0,
        'readability_features': 0.0},
    {'body_bow': 0.0,
        'surface_features': 0.0,
        'lexicon_features': 0.0,
        'liwc_features' :0.0,
        'pos_features':0.2,
        'readability_features': 0.0},
    {'body_bow': 0.0,
        'surface_features': 0.0,
        'lexicon_features': 0.0,
        'liwc_features' :0.0,
        'pos_features':0.0,
        'readability_features': 0.2}]



texts_snopesChecked, labels_snopesChecked = DataLoading.load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv",CLASSES)
texts_buzzfeedTop, labels_buzzfeedTop = DataLoading.load_data_buzzfeedtop()
texts_perez, labels_perez = DataLoading.load_data_perez("../data/perez/celeb.csv")

for tw in tws:
    print("\n*** Feature weigths *** \n", tw)
    pipeline = Pipeline([
        # Extract the subject & body
        ('subjectbody', SubjectBodyExtractor()),

        # Use ColumnTransformer to combine the features from subject and body
        ('union', ColumnTransformer(
            [
                ('body_bow', TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, ngram_range=(1 ,2),
                                             stop_words='english'), 1),
                ('pos_features', Pipeline([
                    ('stats', PosTagFeatures()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ]), 1),
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
                ('readability_features', Pipeline([
                    ('stats', ReadabilityFeatures()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                    ]), 1)
            ],
            transformer_weights=tw
        )),
        ('svc', LinearSVC(penalty= "l2", dual=False, tol=1e-3)),
    ])


    print("Fitting the model...")

    cross_val = cross_validate(pipeline, texts_train_valid, labels_train_valid, return_train_score=True, cv=3, scoring = 'f1_micro')
    print("********************\n\n")
    print(cross_val, "\n Mean train score: ", np.mean(cross_val['train_score']), "\n Mean test score: ", np.mean(cross_val['test_score']))
    print("********************\n\n")
    grid_search = pipeline.fit(texts_train_valid, labels_train_valid)
    print("Results on training data:")
    y = grid_search.predict(texts_train_valid)
    print(classification_report(labels_train_valid, y))

    print("Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
    texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_snopesChecked, labels_snopesChecked ,sample_size=None , discard_labels=[2,5])
    y = grid_search.predict(texts_test)
    print(classification_report(labels_test, y))
    print("confusion matrix:")
    print(confusion_matrix(labels_test, y))
    print(pd.DataFrame({'Predicted': y, 'Expected': labels_test}))

    print("Test results on data sampled only from buzzfeedTop (mixed claims):")
    texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_buzzfeedTop, labels_buzzfeedTop ,sample_size=None , discard_labels=[])
    y = grid_search.predict(texts_test)
    print(classification_report(labels_test, y))
    print("confusion matrix:")
    print(confusion_matrix(labels_test, y))
    print(pd.DataFrame({'Predicted': y, 'Expected': labels_test}))

    print("Test results on data sampled only from perez (celebrity stories):")
    texts_test, labels_test, texts, labels = DataLoading.balance_data(texts_perez, labels_perez ,sample_size=None , discard_labels=[])
    y = grid_search.predict(texts_test)
    print(classification_report(labels_test, y))
    print("confusion matrix:")
    print(confusion_matrix(labels_test, y))
    print(pd.DataFrame({'Predicted': y, 'Expected': labels_test}))
