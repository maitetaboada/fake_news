
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


#from textutils import DataLoading
import os
import numpy as np
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import string

from bs4 import BeautifulSoup
#from keras.utils.np_utils import to_categorical as to_cat
import random
import time

#################

def clean_str(string):
    string = str(string)
    #string = re.sub(r"\\", "", string.decode("utf-8"))
    #string = re.sub(r"\'", "", string.decode("utf-8"))
    #string = re.sub(r"\"", "", string.decode("utf-8"))
    string = ''.join(e for e in string if (e.isspace() or e.isalnum()))  # comment the if part for Mehvish parser
    return string.strip().lower()

def load_data_buzzfeed(file_name="../data/buzzfeed-facebook/buzzfeed-v02-originalLabels.txt", classes = 2):
    print("Loading data buzzfeed...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["ID", "URL", "label", "data", "domain", "source"],
                               usecols=[2, 3])
    print(data_train.shape)
    '''
    texts = []
    labels = []
    for idx in range(data_train.data.shape[0]):
        #text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(data_train.label[idx])
    '''

    texts = data_train.data
    labels = data_train.label

    if (classes == 2):
        transdict = {
            'mostly true': 0,
            'mostly false': 1,

            'mixture of true and false': 5,
            'no factual content': 5,
        }
    else:
        transdict = {
            'no factual content': 0,
            'mostly true': 1,
            'mixture of true and false': 2,
            'mostly false': 3
        }
    labels = [transdict[i] for i in labels]
    #labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    print(pd.value_counts((labels)))
    return texts, labels

def balance_data(texts, labels, sample_size = None, discard_labels=[]):
    ## sample size is the number of items we want to have from EACH class
    unique, counts = np.unique(labels, return_counts=True)
    print(np.asarray((unique, counts)).T)
    all_index = np.empty([0], dtype=int)
    for l, f in zip(unique, counts):
        if (l in discard_labels):
            print ("Discarding items for label " + str(l))
            continue
        l_index = (np.where(labels == l)[0]).tolist()  ## index of input data with current label
        if (sample_size == None ):
            # print "No up or down sampling")
            l_index = np.asarray(l_index)
        elif (sample_size - f > 0):
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


def load_data_snopes(file_name, classes = 2 ):
    # Useful for reading from the following files:
    # "../data/snopes/snopes_checked_v02_right_forclassificationtest.csv"
    # "../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv"

    print("Loading data snopes...")
    data_train = pd.read_csv(file_name)
    print(data_train.shape)
    print(data_train.label[0:10])
    print(data_train.label.unique())
    # print(data_train[data_train["label"].isnull()])
    texts = data_train.data
    labels = data_train.label

    if (classes == 2):
        transdict = {
            'ftrue': 0,
            'mtrue': 0,
            'mfalse': 1,
            'ffalse': 1,

            'mixture': 5,
        }
    else:
        transdict = {
            'ftrue': 0,
            'mtrue': 1,
            'mixture': 2,
            'mfalse': 3,
            'ffalse': 4,
        }
    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print("Data from Snopes looks like...")
    print(texts[0:10])
    print(labels[0:10])
    print(pd.value_counts((labels)))
    return texts, labels



def load_data_snopes312(file_name="../data/snopes/snopes_checked_v02_forCrowd.csv", classes = 2):
    print("Loading data snopes312...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")

    print(df.shape)
    print(df[0:3])
    df = df[df["assessment"] == "right"]
    print(pd.crosstab(df["assessment"], df["fact_rating_phase1"], margins=True))
    labels = df.fact_rating_phase1
    texts = df.original_article_text_phase2     .apply(lambda x: clean_str(BeautifulSoup(x).encode('ascii', 'ignore')))
    #
    '''
    texts = []
    labels = []
    print(df.original_article_text_phase2.shape[0])
    print(df.original_article_text_phase2[2])

    for idx in range(df.original_article_text_phase2.shape[0]):
        text = BeautifulSoup(df.original_article_text_phase2[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(df.fact_rating_phase1[idx])
    '''

    if (classes == 2):
        transdict = {
            'true': 0,
            'mostly true': 0,
            'mixture': 5,
            'mostly false': 1,
            'false': 1
        }
    else:
        transdict = {
            'true': 1,
            'mostly true': 2,
            'mixture': 3,
            'mostly false': 4,
            'false': 5
        }


    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print("Data from SnopesChecked looks like...")
    print(texts[0:10])
    print(labels[0:10])
    print(pd.value_counts((labels)))


def load_data_emergent(file_name="../data/emergent/url-versions-2015-06-14.csv", classes = 2):
    print("Loading data emergent...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")

    print(df.shape)
    print(df[0:3])
    df = df.drop_duplicates(
        df.columns.difference(['articleVersionId', 'articleVersion', 'articleUrl']),
        keep="last")
    df = df[df["articleBody"] != ""]
    print(pd.crosstab(df["articleStance"], df["claimTruthiness"], margins=True))

    df = df[df["articleStance"] == "for"]

    labels = df.claimTruthiness
    texts = df.articleBody.apply(lambda x: str(x)) #apply(lambda x: clean_str(BeautifulSoup(str(x)).encode('ascii', 'ignore')))

    if (classes == 2):
        transdict = {
            'true': 0,
            'false': 1,
            'unknown': 5
        }
    else:
        transdict = {
            'true': 1,
            'false': 2,
            'unknown': 0
        }

    labels = [transdict[i] for i in labels]
    # labels = to_cat(np.asarray(labels))
    print("Data from Emergent looks like...")
    print(texts[0:10])
    print(labels[0:10])
    print(pd.value_counts((labels)))
    return texts, labels

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

#####################



class PosTagFeatures(BaseEstimator, TransformerMixin):
    pos_family = {}

    def __init__(self):
        print("Inside the init function of PosTagFeatures()")
        self.pos_family = {
            'noun': ['NOUN'],
            'pron': ['PRON'],
            'verb': ['VERB'],
            'adj': ['ADJ'],
            'adv': ['ADV'],
            'content': ['NOUN', 'VERB', 'ADJ', 'ADV'],
            'function': ['ADP', 'PRON', 'NUM', 'PRT', 'DET', 'CONJ'],
            'non': ['X']
        }

    def check_pos_tag(self, text, flag):
        tokenizer = lambda x: x.split()
        tuples = Counter(tag for word, tag in nltk.pos_tag(tokenizer(text), tagset='universal'))
        # print(tuples)
        cnt = 0
        rest = 0
        for ppo in tuples:
            if ppo in self.pos_family[flag]:
                cnt += tuples[ppo]
            else:
                rest += tuples[ppo]
        # print(cnt)
        return cnt * 1.0 / (cnt + rest)

    # fit() doesn't do anything, this is a transformer class
    def fit(self, texts, y=None):
        return self

    # all the work is done here
    def transform(self, texts):
        allTags = ['NOUN', 'PRON', 'ADJ', 'ADV', 'VERB', 'ADP', 'NUM', 'PRT', 'DET', 'X', 'CONJ', '.']
        tokenizer = lambda x: x.split()
        features = [dict(Counter(allTags + [tag for word, tag in nltk.pos_tag(tokenizer(text), tagset='universal')]))
                    for text in texts]
        print(features[0:3])
        features = np.array(features)
        # features = features*1.0 #features.apply(sum, axis=1)
        # features = features.divide(features['n_tokens'], axis=1)

        print(features[0])
        print(type(features))
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
                     'num_punc': len("".join(_ for _ in text if _ in string.punctuation)),
                     'num_upper': len([wrd for wrd in text.split() if wrd.isupper()]),
                     'num_word': len(text.split())
                     }
                    for text in posts]

        '''    
        print("In SurfaceFeatures: avg length and avg num_sentences ")
        avgLength = sum(item['num_char'] for item in features) / len(features)
        avgSentences = sum(item['num_sentence'] for item in features) / len(features)
        # print(avgLength)
        # print(avgSentences)
        # print(features[0:10])
        '''
        print(features[0])
        print(type(features))
        return features


class LiwcFeatures(BaseEstimator, TransformerMixin):
    liwcDic = {}  # = map of text to dataframe row (this dataframe should be read from the file including liwc features)

    def __init__(self):
        print("Inside the init function of LiwcFeatures()")
        self.liwcDic = liwcDic

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        # find the line related to this text in dataframe posts[['feature1','feature2',...,'feature100']].to_dict('records')
        # return [self.getLiwcRow(text).to_dict('records')
        #        for text in posts]
        textvecs = []
        for text in texts:
            # print "*** Current text:\n" + text  + "\n***"
            tokens = nltk.word_tokenize(text)
            textvec = {}
            for category in self.liwcDic.keys():
                lexicon_words = [i for i in tokens if i in self.liwcDic[category]]
                count = len(lexicon_words)
                # print ("Count: " + str(count))
                count = count * 1.0 / len(tokens)  # Continous treatment
                # count = 1 if (count > 0) else 0     #Binary treatment
                textvec[category] = count
            # print(textvec)
            textvecs.append(textvec)

        print(textvecs[0])
        textvecs = np.array(textvecs)
        return textvecs


class LexiconFeatures(BaseEstimator, TransformerMixin):
    lexicons = None
    lexiconNames = None

    def __init__(self):
        print("Inside the init function of LexiconFeatures()")
        # lexicon_directory = "../data/bias_related_lexicons"
        self.lexicons = lexicons
        self.lexiconNames = lexiconNames
        # print("LexiconFeatures() init: loading lexicons")
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
        # print("transforming:...")
        textvecs = []
        for text in texts:
            # print "*** Current text:\n" + text  + "\n***"
            tokens = nltk.word_tokenize(text)
            textvec = {}
            for lexicon, lexiconName in zip(self.lexicons, self.lexiconNames):
                lexicon_words = [i for i in tokens if i in lexicon]
                count = len(lexicon_words)
                # print ("Count: " + str(count))
                count = count * 1.0 / len(tokens)  # Continous treatment
                # count = 1 if (count > 0) else 0     #Binary treatment
                textvec[lexiconName] = count
            # print(textvec)
            textvecs.append(textvec)

        print(textvecs[0])
        textvecs = np.array(textvecs)
        return textvecs


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



#########################
########################


LOAD_DATA_FROM_DISK = True
CLASSES = 2

if LOAD_DATA_FROM_DISK:
    texts_all = np.load("../dump/texts_all")
    texts_all_trans = np.load("../dump/texts_all_trans")
    labels_all = np.load("../dump/labels_all")
    print("Data loaded from disk!")

else:
    # Data sources used for training:
    texts_snopes, labels_snopes = load_data_snopes \
        ("../data/snopes/snopes_leftover_v02_right_forclassificationtrain.csv", CLASSES  )  # load_data_liar("../data/liar_dataset/train.tsv")#load_data_rashkin("../data/r$
    texts_buzzfeed, labels_buzzfeed = load_data_buzzfeed("../data/buzzfeed-facebook/buzzfeed-v02-originalLabels.txt",
                                                                                 CLASSES)
    texts_emergent, labels_emergent = load_data_emergent(
        "../data/emergent/url-versions-2015-06-14.csv", CLASSES)
    len(texts_snopes)
    len(texts_buzzfeed)
    len(texts_emergent)

    texts_all = pd.concat(
        [pd.Series(texts_snopes), pd.Series(texts_buzzfeed), pd.Series(texts_emergent)])
    labels_all = pd.concat(
        [pd.Series(labels_snopes), pd.Series(labels_buzzfeed), pd.Series(labels_emergent)])
    texts_all = np.array(texts_all.values.tolist())
    texts_all = texts_all.reshape((texts_all.shape[0], 1))
    labels_all = np.array(labels_all.values.tolist())
    labels_all = labels_all.reshape((labels_all.shape[0], 1))

    print(texts_all[0:3][0:10])
    print(labels_all[0:3])

    ct = ColumnTransformer(
        [

            ('body_bow', TfidfVectorizer(sublinear_tf=True, max_df=0.3, min_df=5, max_features=200,
                                         stop_words='english', ngram_range=(1, 3)), 0),
            ('pos_features', PosTagFeatures(), 0),
            ('surface_features', Pipeline([
                ('stats', SurfaceFeatures()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]), 0),
            ('lexicon_features', Pipeline([
                ('stats', LexiconFeatures()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]), 0),
            ('liwc_features', Pipeline([
                ('stats', LiwcFeatures()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ]), 0)

        ]
    )
    print(texts_all.shape)

    start = time.time()

    ct.fit(texts_all)
    texts_all_trans = ct.transform(texts_all)
    print(texts_all_trans.shape)

    end = time.time()
    print("Time spent on feature extraction:" , end - start)

    texts_all.dump("../dump/texts_all")
    texts_all_trans.dump("../dump/texts_all_trans")
    labels_all.dump("../dump/labels_all")
    print("Data dumped to disk!")

print("Size of data (Raw, Trans, Labels): " + str(len(texts_all)) + " , " + str(
    len(texts_all_trans)) + " , " + str(len(labels_all)))

print(texts_all[0:3])
print(texts_all_trans[0:3])
print(labels_all[0:3])

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression


def rankFeatures(texts, labels, dicOrDf="dic"):
    # data preparation
    if (dicOrDf == "dic"):
        df = pd.DataFrame(list(texts))
    else:
        df = texts
    print(df)
    X = df.values
    featureNames = list(df.columns.values)
    print(X[0:3, :])
    print(featureNames)
    Y = labels
    print(Y[0:3])

    # feature extraction
    model = LogisticRegression()
    rfe = RFE(model)
    fit = rfe.fit(X, Y)
    print("Num Features: ", fit.n_features_)
    print("Selected Features: ", fit.support_)
    print("Feature Ranking: ", fit.ranking_)

    # feature ranking
    ranks = fit.ranking_
    sortedFeatures = [x for _, x in sorted(zip(ranks, featureNames))]
    # print(sortedFeatures)

    selector = SelectPercentile(f_regression, percentile=10)
    selector.fit(X, Y)
    scores = selector.pvalues_  # -np.log10(selector.pvalues_)
    # scores /= scores.max()
    print(scores)
    sortedFeatures = [x for _, x in sorted(zip(scores, featureNames))]
    featureScores = [(featureNames[i], scores[i]) for i in range(0, len(scores))]
    featureScores = dict(featureScores)
    # print(featureScores)
    dd = pd.DataFrame(data=np.column_stack((Y, X)), columns=['key'] + featureNames)
    # for f in featureNames:
    #    print(dd.groupby('key')[f].mean())#, df.groupby('key')[f].std() )

    featureSplits = [dd.groupby('key')[f].mean() for f in featureNames]
    # print(featureSplits)
    featureRatios = [(x.name, x[0], x[1], featureScores[x.name]) for x in featureSplits]
    # print(featureRatios)
    featureRatios = pd.DataFrame(list(featureRatios), columns=["feature", "true", "false", "p-value"])
    # featureRatios["ratio true/false"] = featureRatios["true"] / featureRatios["false"]
    featureRatios["ratio false/true"] = featureRatios["false"] / featureRatios["true"]
    print(featureRatios.sort_values(by=['ratio false/true']))
    return featureRatios





lexic = LexiconFeatures()
texts_lexic = lexic.transform(texts_all[:,0])
labels = labels_all[:]
print(texts_lexic)
print(labels)
print(type(texts_lexic))
print(type(labels))

rankedFeatures = rankFeatures(texts_lexic, labels)
rankedFeatures.to_csv("../dump/rankedFeatures_lexic")


pos = PosTagFeatures()
texts_pos = pos.transform(texts_all[:,0])
labels = labels_all[:]
print(texts_pos)
print(labels)
print(type(texts_pos))
print(type(labels))

rankedFeatures = rankFeatures(texts_pos, labels)
rankedFeatures.to_csv("../dump/rankedFeatures_pos")


surf = SurfaceFeatures()
texts_surf = surf.transform(texts_all[:,0])
labels = labels_all[:]
print(texts_surf)
print(labels)
print(type(texts_surf))
print(type(labels))

rankedFeatures = rankFeatures(texts_surf, labels)
rankedFeatures.to_csv("../dump/rankedFeatures_surf")


def top_tfidf_feats(row, features, top_n=10):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''

    if grp_ids:
        print("---\n", grp_ids)
        print("***\n", Xtr[grp_ids])
        D = np.asarray(Xtr[grp_ids])  # .toarray()
        print("***\n", D)
    else:
        D = np.asarray(Xtr)  # .toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        print("Finding good features for label ", label)
        ids = np.where(y == label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def rankSparseFeatures(vectorizer, texts, labels, min_tfidf=0.1, top_n=10):
    # data preparation
    X = vectorizer.fit_transform(texts)
    featureNames = vectorizer.get_feature_names()
    featureNames = np.asarray(featureNames)
    indices = np.argsort(vectorizer.idf_)[::-1]
    print(X[0:3])
    print(featureNames)
    print("n_samples: %d, n_features: %d" % X.shape)
    Y = labels

    # top features
    top_features = [featureNames[i] for i in indices[:top_n]]
    print("General top features: ", top_features)

    # top features by class
    print("All feature names:", featureNames)
    dfs = top_feats_by_class(X, Y, featureNames, min_tfidf, top_n)
    print(dfs)



tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, max_features=1000,
                                         stop_words='english', ngram_range=(2, 2))
texts = texts_all[0:100,0]
labels = labels_all[0:100]

texts_tfidf = tfidf.fit_transform(texts)
texts_tfidf = texts_tfidf.toarray()
featureNames = np.asarray(tfidf.get_feature_names())
print(type(texts_tfidf))
print(featureNames)

texts_tfidf = pd.DataFrame(texts_tfidf, columns =  featureNames)

print(texts_tfidf[0:3])

rankedFeatures = rankFeatures(texts_tfidf, labels, dicOrDf = "df")
rankedFeatures.to_csv("../dump/rankedFeatures_tfidf")


