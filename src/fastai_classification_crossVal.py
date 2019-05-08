




from fastai import *
from fastai.text import *
#from textutils import DataLoading
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import random
import numpy as np
import torch

#reprudicibility:
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


LOAD_LM_FROM_DISK = True

def load_data_rashkin(file_name, classes = 4):
    print("Loading data...")
    data_train = pd.read_table(file_name, sep='\t', header=None, names=["label", "data"], usecols=[0, 1],
                               dtype={"label": np.str, "data": np.str})
    print(data_train.shape)
    print(data_train[0:6])
    texts = []
    labels = []
    # for i in range(data_train.data.shape[0]):
    #   print(i, type(data_train.data[i]))

    texts = data_train.data
    labels = data_train.label
    '''
    for idx in range(data_train.data.shape[0]):
        text = BeautifulSoup(data_train.data[idx])
        texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        labels.append(str(data_train.label[idx]))
    '''
    if( classes == 2):
        transdict = {
            '1': 1,  # Satire
            '2': 1,  # Hoax
            '3': 1,  # Propaganda
            '4': 0  # Truested
        }
    else:
        transdict = {
            '1': 2,  # Satire
            '2': 3,  # Hoax
            '3': 1,  # Propaganda
            '4': 0  # Truested
        }
    labels = [transdict[i] for i in labels]
    #labels = to_cat(np.asarray(labels))
    print(texts[0:6])
    print(labels[0:6])
    return texts, labels

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
    return texts, labels


def load_data_buzzfeedtop(file_name="../data/buzzfeed-top/buzzfeed-top.csv"):
    print("Loading data buzzfeedtop...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")
    print(df.shape)
    print(df[0:3])
    #texts = df.original_article_text_phase2.apply(lambda x: clean_str(BeautifulSoup(x).encode('ascii', 'ignore')))
    texts = df.original_article_text_phase2
    labels = [1] * len(df.original_article_text_phase2) # all are false news
    #labels = "false" * len(df.original_article_text_phase2)
    print("Data from BuzzFeed looks like...")
    print(texts[0:10])
    print(labels[0:10])
    print(pd.value_counts((labels)))
    return texts, labels




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




def load_data_perez(file_name="../data/perez/celeb.csv"):
    print("Loading data perez...")
    df = pd.read_csv(file_name, encoding="ISO-8859-1")
    print(df.shape)
    print(df[0:3])
    #texts = df.original_article_text_phase2.apply(lambda x: clean_str(BeautifulSoup(x).encode('ascii', 'ignore')))
    texts = df.text
    labels = df.label
    transdict = {
        'legit': 0,
        'fake': 1
    }
    labels = [transdict[i] for i in labels]
    #labels = "false" * len(df.original_article_text_phase2)
    print("Data from perez looks like...")
    print(texts[0:10])
    print(labels[0:10])
    print(pd.value_counts((labels)))
    return texts, labels





#path = untar_data(URLs.IMDB_SAMPLE)
path =  "~/workspace/shared/sfu/fake_news/dump/fastai"

CLASSES = 2

texts_all = np.load("../dump/trainRaw")
labels_all = np.load("../dump/trainlRaw")
texts_train_external = np.load("../dump/trainRaw_external")
texts_valid_external = np.load("../dump/validRaw_external")
texts_test_external = np.load("../dump/testRaw_external")
labels_train_external = np.load("../dump/trainlRaw_external")
labels_valid_external = np.load("../dump/validlRaw_external")
labels_test_external = np.load("../dump/testlRaw_external")

texts_snopesChecked, labels_snopesChecked = load_data_snopes("../data/snopes/snopes_checked_v02_right_forclassificationtest.csv", CLASSES)
texts_buzzfeedTop, labels_buzzfeedTop = load_data_buzzfeedtop()
texts_perez, labels_perez = load_data_perez("../data/perez/celeb.csv")

print("Data loaded from disk!")


# Use training data for language model tuning (or some external big [unlabeled] corpus)
#train_df = pd.DataFrame( {'label':  labels_train_external.astype(str), 'text':  texts_train_external})
#valid_df = pd.DataFrame( { 'label':  labels_valid_external.astype(str), 'text':  texts_valid_external})
print(labels_all.shape, int(labels_all.shape[0]/3))

if(LOAD_LM_FROM_DISK):
    data_lm = TextLMDataBunch.load(path + "/languageModel")
    learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
    learn.load_encoder("../../languageModel/models/"+ 'LM_selfData')
else:
    texts_valid, labels_valid, texts_train, labels_train = balance_data(texts_all, labels_all, int(labels_all.shape[0]/3), [2,3,4,5])
    train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
    valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
    data_lm = TextLMDataBunch.from_df(path + "/languageModel", train_df = train_df, valid_df = valid_df)
    learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
    train_df = pd.DataFrame( {'label':  labels_train.astype(str), 'text':  texts_train})
    valid_df = pd.DataFrame( { 'label':  labels_valid.astype(str), 'text':  texts_valid})
    print(train_df.head(3))
    data_lm = TextLMDataBunch.from_df(path + "/languageModel", train_df = train_df, valid_df = valid_df)
    data_lm.save()
    learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.5)
    # Building a language model
    print("Language model learning")
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, 1e-3/2)
    learn.unfreeze()
    learn.fit(10, 1e-3)
    learn.save_encoder('LM_selfData')


def predict(model, mytexts):
    return [model.predict(x)[0] for x in mytexts]


# ******************************
# ****Classifier model**********
print("Cross validation starts:")
kf = StratifiedKFold(n_splits=3)
for train, valid in kf.split(texts_all, labels_all):
    print("%s %s" % (train, valid))
    texts_train, texts_valid = texts_all[train], texts_all[valid]
    labels_train, labels_valid = labels_all[train], labels_all[valid]
    print(texts_train.shape)
    print(texts_valid.shape)

    train_df = pd.DataFrame({'label': labels_train.astype(str), 'text': texts_train})
    valid_df = pd.DataFrame({'label': labels_valid.astype(str), 'text': texts_valid})
    print("\n\n A sample of classifier training data:")
    print(train_df.head(100))
    data_clas = TextClasDataBunch.from_df(path + "/classification", train_df=train_df, valid_df=valid_df,
                                          vocab=data_lm.train_ds.vocab, bs=32)
    learn = text_classifier_learner(data_clas, drop_mult=0.7)
    learn.load_encoder("../../languageModel/models/" + 'LM_selfData')
    print("Language model loaded!")
    print("Building the text classifier...")
    learn.freeze()
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, 1e-2)
    learn.freeze_to(-2)
    learn.fit(1, 1e-3)
    
    learn.unfreeze()
    learn.fit(1, 1e-3)
    # learn.fit_one_cycle(1, 1e-3)
    # learn.fit_one_cycle(1, 1e-3/2.)
    learn.fit(15, slice(2e-3 / 100, 2e-3))
    print("saving the classifier...")
    learn.save('TC_LM_selfData')

    # Predictions
    sentence = "Hilary Clinton won the 2016 US election"
    p = learn.predict(sentence)
    print("Prediction for sentence \" " + sentence + "\" is " + str(p))

    print("Results on all  data:")
    y = predict(learn, texts_all)
    print(classification_report(labels_all, list(map(int, y))))


    print("Results on validation data:")
    y = predict(learn, texts_valid)
    print(classification_report(labels_valid, list(map(int, y))))
    
    print( "Test results on data sampled only from snopes (snopes312 dataset manually checked right items -- unseen claims):")
    texts_test, labels_test, texts, labels = balance_data(texts_snopesChecked, labels_snopesChecked, None, [2, 5])
    y = predict(learn, texts_test)
    print(classification_report(labels_test, list(map(int, y))))

    print("Test results on data sampled only from buzzfeedTop (mixed claims):")
    texts_test, labels_test, texts, labels = balance_data(texts_buzzfeedTop, labels_buzzfeedTop, sample_size=None,
                                                          discard_labels=[])
    y = predict(learn, texts_test)
    print(classification_report(labels_test, np.asarray(y).astype(int)))

    print("Test results on data sampled only from perez (celebrity stories):")
    texts_test, labels_test, texts, labels = balance_data(texts_perez, labels_perez, sample_size=None,
                                                          discard_labels=[])
    y = predict(learn, texts_test)
    print(classification_report(labels_test,np.asarray(y).astype(int)))








