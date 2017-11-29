



from representations import Word2vecFeatures as w2vF
from representations import LexiconFeatures as lexF

import string
import math
#from gensim.models.fastsent import FastSent
from string import punctuation
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from gensim import utils, matutils
import numpy as np
import copy
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization,Conv1D
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers as regs

import itertools
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import resample
from collections import Counter



## This flag is used to mark cases (sentences or sentence pairs) that a model cannot successfully vectorize
errorFlag = ["error flag"] 
        
## lower case and removes punctuation from the input text
#def process(s): return [i.lower().translate(None, punctuation).strip() for i in s]
def process(s):
    replace_punctuation = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
    #return [i.lower().translate(replace_punctuation).strip() for i in s]
    return [i.lower().strip() for i in s]

## find features (a vector) describing the relation between two sentences
def get_features(models, sentences):
    print "using method get_features!"
    result = list()
    for sentence in sentences:
        try:
            vector = list()
            for index , model in enumerate(models):
                part = model.get_features(sentence)
                vector.extend(part)
            result.append(vector)
            #print "\n\n\n***\nText:\n " + sentence
            #print "\n***\nFeatures:\n " + str(vector)
        except:
            #print("ERROR: " + sentenceA + " & " +  sentenceB)
            result.append(errorFlag)
    return result


def train(feature_models, trainSet, devSet, extra_features, nclass ):
   
    print 'Computing feature vectors directly through model.get_features() ...'
    
    trainF = np.asarray(get_features(feature_models, process(trainSet[0])))
    trainY = encode_labels(trainSet[1],nclass)
    index = [i for i, j in enumerate(trainF) if j ==  errorFlag]
    trainF = np.asarray([x for i, x in enumerate(trainF) if i not in index])
    trainY = np.asarray([x for i, x in enumerate(trainY) if i not in index])
    print trainF.shape
    print np.asarray(trainSet[2]).shape
    if (extra_features):
        trainExF = np.asarray([x for i, x in enumerate(trainSet[2]) if i not in index])
        trainF = np.column_stack((trainF, trainExF))
    print trainF.shape

    devF = np.asarray(get_features (feature_models, process(devSet[0])))
    devY = encode_labels(devSet[1],nclass)
    index = [i for i, j in enumerate(devF) if j ==  errorFlag]
    devF = np.asarray([x for i, x in enumerate(devF) if i not in index])
    devY = np.asarray([x for i, x in enumerate(devY) if i not in index])
    devS = np.asarray([x for i, x in enumerate(devSet[1]) if i not in index])
    if (extra_features):
        devExF = np.asarray([x for i, x in enumerate(devSet[2]) if i not in index])
        devF = np.column_stack((devF, devExF))
    print devF.shape

    print 'Compiling model...'
    lrmodel = prepare_nn_model(dim= trainF.shape[1], nclass = nclass)#, ninputs=trainF.shape[0])

    print 'Training...'
    bestlrmodel = train_nn_model(lrmodel, trainF, trainY, devF, devY, devS, nclass)

    return bestlrmodel

    


def test(models, classifier, testSet, extra_features , nclass):
    
    print 'Computing feature vectors directly through model.get_features() ...'
    testF = np.asarray(get_features(models, process(testSet[0])))
    index = [i for i, j in enumerate(testF) if j ==  errorFlag]
    testF = np.asarray([x for i, x in enumerate(testF) if i not in index])
    testS = np.asarray([x for i, x in enumerate(testSet[1]) if i not in index])

    print "Before extra features:" + str(testF.shape)
    #print (np.asarray(testSet[2]).shape)
    print testF
    if(extra_features):
        testExF = np.asarray([x for i, x in enumerate(testSet[2]) if i not in index])
        testF = np.column_stack((testF, testExF))
    #print testF.shape
    print "After extra features:" + str(testF.shape)
    print testF

    p = classifier.predict_proba(testF, verbose=0)
    yhat = decode_labels(p, nclass)
    print(pd.DataFrame({'Predicted': yhat, 'Expected': testS}))

    f1 = f1_score(testS, yhat, average=None)
    pre = precision_score(testS, yhat, average=None)
    rec = recall_score(testS, yhat, average=None)
    accuracy = accuracy_score(testS, yhat)
    print("\n************ SUMMARY ***********")
    print 'Test data size: ' + str(len(testS))
    print 'Test F1-Score: ' + str(f1)
    print 'Test Precision: ' + str(pre)
    print 'Test Recall: ' + str(rec)
    print 'Test Accuracy: ' + str(accuracy)
    print("********************************")
    #***

    pr = pearsonr(yhat, testS)[0]
    #sr = spearmanr(yhat, testS)[0]
    print("\n************ SUMMARY ***********")
    print 'Test data size: ' + str(len(testS))
    print 'Test Pearson: ' + str(pr)
    #print 'Spearman: ' + str(sr)
    print("********************************")

    sentences = np.asarray([x for i, x in enumerate(process(testSet[0])) if i not in index])
    a =  [ (sentences[i], testS[i], yhat[i], testS[i] - yhat[i]) for i,s in enumerate(sentences) ]
    b = pd.DataFrame(a, columns = ['text','score','prediction','error'])
    #print(b.sort(['error', 'score']))
    return b
    



def train_nn_model(lrmodel, X, Y, devX, devY, devscores, nclass):
    """
    Train model, using pearsonr on dev for early stopping
    """
    done = False
    best = -1.0
    while not done:
        # Every 100 epochs, check Pearson on development set
        lrmodel.fit(X, Y, verbose=1, shuffle=False, validation_data=(devX, devY))
        p = lrmodel.predict_proba(devX, verbose=0)
        yhat = decode_labels(p, nclass)
        print(pd.DataFrame({'Predicted': yhat, 'Expected': devscores}))

        #print(f1_score(devscores, yhat, average="micro"))
        #print(precision_score(devscores, yhat, average="micro"))
        #print(recall_score(devscores, yhat, average="micro")))
        #score = f1_score(devscores, yhat, average="micro")
        #print(p)
        score = accuracy_score(devscores, yhat)
        #score = pearsonr(yhat, devscores)[0]
        if score > best:
            print 'Dev score: = ' + str(score) # print 'Dev Pearson: = ' + str(score)
            print 'Dev F1-score: = ' + str(f1_score(devscores, yhat, average=None))  # print 'Dev Pearson: = ' + str(score)
            print 'Dev Precision score: = ' + str(precision_score(devscores, yhat, average=None))  # print 'Dev Pearson: = ' + str(score)
            print 'Dev Recall score: = ' + str(recall_score(devscores, yhat, average=None))  # print 'Dev Pearson: = ' + str(score)
            best = score
            ## FA: commented out the following line because of the new keras version problem with deepcopy
            ## FA: not the model scored right after the best model will be returned (not too bad though, usually the difference is so small)
            #bestlrmodel = copy.deepcopy(lrmodel)
        else:
            done = True
    return lrmodel
    


def prepare_nn_model(dim, nclass):
    lrmodel = Sequential()
    lrmodel.add(Dense(300, input_dim=dim))#, activity_regularizer=regs.l1(0.01))) #set this to twice the size of sentence vector or equal to the final feature vector size
    #lrmodel.add(BatchNormalization())
    #lrmodel.add(Dense(100))
    lrmodel.add(Activation('relu'))
    lrmodel.add(Dropout(0.2))
    #lrmodel.add(Conv1D(filter_length= 10, nb_filter= 10))
    lrmodel.add(Dense(300))
    lrmodel.add(Activation('relu'))
    lrmodel.add(Dense(nclass))
    lrmodel.add(Activation('softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    lrmodel.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    #opt = SGD(lr=0.0001)
    #lrmodel.compile(loss="categorical_crossentropy", optimizer=opt)
    return lrmodel

'''
    lrmodel = Sequential()
    lrmodel.add(Dense(500, input_dim=dim)) #set this to twice the size of sentence vector or equal to the final feature vector size
    #lrmodel.add(BatchNormalization())
    lrmodel.add(Activation('relu'))
    lrmodel.add(Dropout(0.005))
    lrmodel.add(Dense(nclass))
    lrmodel.add(Activation('softmax'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    lrmodel.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    #opt = SGD(lr=0.0001)
    #lrmodel.compile(loss="categorical_crossentropy", optimizer=opt)
    return lrmodel
    
    
    Dev score: = 0.780141843972
    Dev F1-score: = 0.791181873852
    Dev Precision score: = 0.801986343886
    Dev Recall score: = 0.780664652568
    Train on 27926 samples, validate on 3102 samples
        
    Test data size: 1121
    Test F1-Score: 0.761904761905
    Test Precision: 0.68085106383
    Test Recall: 0.864864864865
    Test Accuracy: 0.750223015165
'''

'''
    ## Fa: Binary classification changes are applied using this new architecture
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, activation='relu'))
    model.add(Dense(dim, activation='relu'))
    model.add(Dense(1, input_dim=dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
'''

def encode_labels(labels, nclass):
    ## Fa: Binary classification changes are applied using this new architecture
    #return np.asarray(labels)-1

    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y

def decode_labels(probabilities, nclass, categorical = True):
    ## Fa: Binary classification changes are applied using this new architecture
    #labels = np.copy(probabilities)
    #labels[labels > 0.5] = 1
    #labels[labels <= 0.5] = 2
    #return np.max(labels, axis=1)

    labels = None
    if(categorical):
        labels = np.argmax(probabilities, axis=1) + 1
    else:
        r = np.arange(1, nclass+1)  # [1,2] for the two labels in the text classification  corpus
        labels = np.dot(probabilities, r)  ## this would return a continous score (meaningful for non-categorical class labels, would not work with F-score objectives)
    return labels


def balanced_resample(allS, allA):
    extraS = []
    extraA = []
    print("Resampling...")
    counter = Counter(allS)
    print(counter)
    max_freq = max(counter.values())
    for label in counter.keys():
        index = [i for i, j in enumerate(allS) if j == label]
        labelS = [allS[i] for i in index]
        labelA = [allA[i] for i in index]
        extra = max_freq - counter.get(label)
        print(extra)
        labelS_, labelA_ = resample(labelS, labelA, random_state=0, n_samples=extra)
        extraS = extraS + labelS_
        extraA = extraA + labelA_
    extraS = allS + extraS
    extraA = allA + extraA
    print(Counter(extraS))
    return extraS, extraA


def load_data_rubin(datafile = "/Users/fa/workspace/temp/rubin/data.xlsx", resampling = False  ):
    xl = pd.ExcelFile(datafile)
    df = xl.parse("Sheet1")

    allA = df["Full Text"].tolist()
    allS = df["Satirical"].tolist()
    print len(allA)
    print len(allS)

    ## shuffle the data
    allS, allA = shuffle(allS, allA, random_state=12345)
    allS = [float(s) for s in allS]
    allS = [(x * 4 + 1) for x in allS]

    ## split into 45% train, 5% dev and remaining ~50% test
    trainA, devA, testA = allA[0 : int(math.floor(0.70 * len(allA)))], allA[int(math.floor(0.70 * len(allA))) + 1 : int(math.floor(0.80 * len(allA))) ], allA[int(math.floor(0.80 * len(allA))) + 1 : ]
    trainS, devS, testS = allS[0 : int(math.floor(0.70 * len(allS)))], allS[int(math.floor(0.70 * len(allS))) + 1 : int(math.floor(0.80 * len(allS))) ], allS[int(math.floor(0.80 * len(allS))) + 1 : ]
    if (resampling):
        trainS, trainA = balanced_resample(trainS, trainA)
        testS, testA = balanced_resample(testS, testA)
        devS, devA = balanced_resample(devS, devA)

    print len(allA)
    print len(trainA)+len(devA)+len(testA)
    print len(trainA), len(devA), len(testA)
    return [trainA, trainS], [devA, devS], [testA, testS]


def load_data_constructiveness(datafolder = "/Users/fa/workspace/temp/data/varada_constructiveness/"):
    xl = pd.ExcelFile(datafolder+"train.xlsx")
    df = xl.parse("Sheet1")
    allA = df["Preprocessed"].tolist()
    allS = df["Constructive"].tolist()
    allF = df[["Has_conjunction_or_connectives","Has_stance_adverbials","Has_reasoning_verbs","Has_modals","Has_shell_nouns",
               "Len","Average_word_length", "Readability", "PersonalEXP", "Named_entity_count", "nSents", "Avg_words_per_sent"]].values.tolist()
    nclass = len(Counter(allS).keys())
    print(nclass)
    print len(allA)
    print len(allS)
    print len(allF)


    ## scale the data
    allS = [float(s) for s in allS]
    allS = [(x * (nclass -1) + 1) for x in allS]

    ## shuffle the data
    allS, allA, allF = shuffle(allS, allA, allF, random_state=54321)

    #print(pd.DataFrame({'Label': allA[:10], 'Score': allS[:10], 'Other_features': allF[:10]}))

    ## split into train and dev
    split = 0.80
    trainA, devA = allA[0: int(math.floor(split * len(allA)))], allA[int(math.floor(split * len(allA))) + 1:]
    trainS, devS = allS[0: int(math.floor(split * len(allS)))], allS[int(math.floor(split * len(allS))) + 1:]
    trainF, devF = allF[0: int(math.floor(split * len(allF)))], allF[int(math.floor(split * len(allF))) + 1:]

    # we can later try resampling for this data too

    xl = pd.ExcelFile(datafolder + "test.xlsx")
    df = xl.parse("Sheet1")
    testA = df["Preprocessed"].tolist()
    testS = df["Constructive"].tolist()
    testF = df[["Has_conjunction_or_connectives","Has_stance_adverbials","Has_reasoning_verbs","Has_modals","Has_shell_nouns",
               "Len","Average_word_length", "Readability", "PersonalEXP", "Named_entity_count", "nSents", "Avg_words_per_sent"]].values.tolist()
    print len(testA)
    print len(testS)
    print len(testF)

    ## scale the data
    testS = [float(s) for s in testS]
    testS = [(x * (nclass -1) + 1) for x in testS]

    print len(trainA) + len(devA)
    print len(trainA), len(devA), len(testA)
    print "End of data loading!"


    ## Just call the resample function to count labels, don't modify data:
    #balanced_resample(trainS, trainA)
    #balanced_resample(testS, testA)
    #balanced_resample(devS, devA)

    #print(pd.DataFrame({'Label': trainA[0:10], 'Score': trainS[0:10], 'Other_features': trainF[0:10]}))

    return [trainA, trainS, trainF], [devA, devS, devF], [testA, testS,testF], nclass
    #return [trainA[0:100], trainS[0:100], trainF[0:100]], [devA, devS, devF], [testA, testS, testF], nclass






if __name__ == '__main__':
    ensemble = list()



    word2vec_model = w2vF.Word2vecFeatures("/Users/fa/workspace/repos/_codes/MODELS/Rob/word2vec_100_6/vectorsW.bin")
    #lexicon_model = lexF.LexiconFeatures("/Users/fa/workspace/temp/NPOV/bias_related_lexicons")



    ## Add specific models to ensemble
    #ensemble.append(lexicon_model)
    ensemble.append(word2vec_model)

    ## Load some data for training (standard SICK dataset)
    #trainSet, devSet, testSet = load_data_SICK('../data/SICK/')
    trainSet, devSet, testSet , nclass = load_data_constructiveness()

    # Uncomment till the build_vocab method if using feedback model
    """
    sentences = []
    sentences.extend(trainSet[0])
    sentences.extend(trainSet[1])
    sentences.extend(devSet[0])
    sentences.extend(devSet[1])
    sentences.extend(testSet[0])
    sentences.extend(testSet[1])
    
    sentences = [sent.decode('utf8') for sent in sentences]
    print 'length:', len(sentences)
    feedm.feedback_model.build_vocab(sentences)
    
    """


    ## Train a classifier using train and development subsets
    classifier = train(ensemble, trainSet, devSet, extra_features = True, nclass = nclass)

    #classifier = pickle.load(open('../pretrained/classifiers/feed+fb+sts1214.file', 'rb'))

    ## Test the classifier on test data of the same type (coming from SICK
    test(ensemble, classifier, testSet, extra_features = True , nclass = nclass).to_csv('../data/temp.csv')

    ## FileName to save the trained classifier for later use
    #fileName = '../data/local/SICK-Classifier.h5'

    ## VERSION THREE SAVE / LOAD (the only one that works)
    #classifier.save(fileName)
    #newClassifier = load_model(fileName)

    ## Test the saved and loaded classifier on the testSet again (to make sure the classifier didn't mess up by saving on disk)
    #test(ensemble, newClassifier, testSet)

