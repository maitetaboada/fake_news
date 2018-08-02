import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urlparse



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

def news_data_summary_2():
    file_name = "politifact_phase2_clean.csv"
    #file_name = "snopes_phase2_clean_first_paragraph.csv"


    df = pd.read_csv(file_name, encoding="utf-8")
    df.describe()
    df = df[df.error_phase2 == 'No Error']

    df['label'] = df['fact_tag_phase1']
    #df['label'] = df['fact_rating_phase1']


    df['label'].value_counts()
    df.sort_values('label')
    df['category'] = df.article_categories_phase1.apply(lambda x: x[:7])
    pd.crosstab(df.category, df.label)
    fn = df[df.category == 'Fake ne']
    pd.crosstab(fn.category, fn.label)
    pd.crosstab(fn.article_researched_by_phase1, fn.label)
    df['domain'] = df['original_url_phase1'].apply(lambda x: urlparse(x).netloc)
    counts = df['domain'].value_counts()
    topdomains = df[df['domain'].isin(counts[counts > 20].index)]
    pd.crosstab(topdomains.category, topdomains.label)

def news_data_sampler(texts, labels,  train_size, dev_size, test_size, seed = 123):
    print("Sampling data for seed " + str(seed))
    texts_test, labels_test, texts, labels = balance_data(texts, labels, test_size, [6, 7], seed )
    pd.DataFrame(data= [texts_test, labels_test],columns=["text", "label"]).to_pickle("../pickle/test", protocol=2)
    texts_dev, labels_dev, texts, labels = balance_data(texts, labels, dev_size, [6, 7], seed )
    pd.DataFrame(data=[texts_dev, labels_dev], columns=["text", "label"]).to_pickle("../pickle/dev", protocol=2)
    texts_train, labels_train, texts, labels = balance_data(texts, labels, train_size, [6, 7], seed )


def news_data_integration_afterJillAssessment():
    # Prepare a new train/test set for experiments: train set is unchecked and test set is checked by Jill for claim-text alignment
    file_name = "../data/snopes/snopes_leftover_v01.csv"
    df_leftover = pd.read_csv(file_name, encoding="ISO-8859-1")
    df_leftover["checked"] = 0

    file_name = "../data/snopes/snopes_checked_v01.csv"
    df_checked = pd.read_csv(file_name, encoding="ISO-8859-1")
    df_checked["checked"] = 1
    df_checked = df_checked.rename(index=str, columns={"label": "assessment"})
    df_checked = df_checked[df_checked.assessment == "right link"]
    df_checked.fact_rating_phase1.value_counts()

    df = pd.concat([df_leftover, df_checked])
    df = df.sort_values(by=['checked'])

    ## remove duplicate collected links from snope articles (because they can appear in several places withing the snope article)
    df = df.drop_duplicates(
        df.columns.difference(['index_paragraph_phase1', 'page_is_first_citation_phase1', 'article_origin_url_phase1']),
        keep="last")
    df['id'] = df.snopes_url_phase1 + df.article_origin_url_phase1
    df.id.describe()

    ## remove duplicate texts
    df.article_title_phase2.describe()
    df = df.drop_duplicates(subset='original_article_text_phase2', keep="last")

    ## sanity check
    df.checked.value_counts()
    df.fact_rating_phase1.value_counts()
    df.fact_rating_phase1 = df.fact_rating_phase1.str.lower()
    df.fact_rating_phase1.value_counts()

    df = df.loc[df['fact_rating_phase1'].str.lower().isin({'false','true','mixture','mostly false','mostly true'})]
    df['fact_rating_phase1'] = df['fact_rating_phase1'].map({'false': 'ffalse',
                                                             'true': 'ftrue',
                                                             'mostly false': 'mfalse',
                                                             'mixture': 'mixture',
                                                             'mostly true': 'mtrue'})

    df[df.checked == 1].fact_rating_phase1.value_counts()
    df.fact_rating_phase1.value_counts()

    df_checked = df[df.checked == 1]
    df_checked.fact_rating_phase1.value_counts()
    d = {'link': df_checked.article_origin_url_phase1, 'label': df_checked.fact_rating_phase1, 'data': df_checked.original_article_text_phase2}
    df_checked = pd.DataFrame(data=d)
    df_checked.label.value_counts()
    df_checked.to_csv("../data/snopes/snopes_checked_Jill_forClassificatio.csv")

    df_unchecked = df[df.checked == 0]
    df_unchecked.fact_rating_phase1.value_counts()
    d = {'link': df_unchecked.article_origin_url_phase1 ,'label': df_unchecked.fact_rating_phase1, 'data': df_unchecked.original_article_text_phase2}
    df_unchecked = pd.DataFrame(data=d)
    df_unchecked.label.value_counts()
    df_unchecked.to_csv("../data/snopes/snopes_leftover_Jill_forClassificatio.csv")


    file_name = "../data/snopes/snopes_allOld.csv"
    df_old = pd.read_csv(file_name, encoding="ISO-8859-1", sep = "\t")

    df_old["tag"] = "old"
    df_checked["tag"] = "checked"
    df_unchecked["tag"] = "unchecked"

    df_old.label.describe()
    df_checked.label.describe()
    df_unchecked.describe()

    #some extra play with data
    df = pd.concat([df_checked, df_unchecked, df_old])
    df.label.describe()
    df_old.subtract(df)


    #see the domains & topics where the checked right data comes from
    df_checked['domain'] = df_checked['article_origin_url_phase1'].apply(lambda x: urlparse(x).netloc)
    pd.crosstab(df_checked.domain, df_checked.fact_rating_phase1).to_csv("../data/snopes/snopes_checked_Jill_domains.csv")
    df_checked['topic'] = df_checked['article_category_phase1'].apply(lambda x: x[0:7])
    pd.crosstab(df_checked.topic, df_checked.fact_rating_phase1).to_csv("../data/snopes/snopes_checked_Jill_domains.csv")






def replace_unusual_characters(line):
    """replace unusual characters in the word
        1. replace wired "’" apostrophe
        2. replace unusual quotation.
        3. ...

    """
    l = re.sub("’", "'", line)
    l = re.sub("‘", "'", l)
    l = re.sub("“", "\"", l)
    l = re.sub("”", "\"", l)
    l = re.sub("，", ",", l)
    return l


def replace_newlines(text):
    t = re.sub("(?m)(^\\s+|[\\t\\f ](?=[\\t\\f ])|[\\t\\f ]$|\\s+\\z)", "", text)
    t = re.sub("\n", "</p><p>", t)
    t = "<p>" + t + "</p>"
    return t

def news_data_integration_forCrowdSourceAssessment():
# Prepare a new balanced set of data for crowd-source annotation.
# It should not overlap with Jill's data (because we use Jill's as test questions.
file_name = "../data/snopes/snopes_leftover_v01.csv"
df_leftover = pd.read_csv(file_name, encoding="ISO-8859-1")
df_leftover["checked"] = 0
df_leftover.fact_rating_phase1 = df_leftover.fact_rating_phase1.str.lower()
df_leftover.fact_rating_phase1.value_counts()

file_name = "../data/snopes/snopes_checked_v01.csv"
df_checked = pd.read_csv(file_name, encoding="ISO-8859-1")
df_checked["checked"] = 1
df_checked = df_checked.rename(index=str, columns={"label": "assessment"})
#df_checked = df_checked[df_checked.assessment == "right link"]
df_checked.fact_rating_phase1 = df_checked.fact_rating_phase1.str.lower()
df_checked.fact_rating_phase1.value_counts()

df = pd.concat([df_leftover, df_checked])
df = df.sort_values(by=['checked'])

## remove duplicate collected links from snope articles (because they can appear in several places withing the snope article)
df = df.drop_duplicates(
    df.columns.difference(['index_paragraph_phase1', 'page_is_first_citation_phase1', 'article_origin_url_phase1']),
    keep="last")
df['id'] = df.snopes_url_phase1 + df.article_origin_url_phase1
df.id.describe()

## remove duplicate texts
df.article_title_phase2.describe()
df = df.drop_duplicates(subset='original_article_text_phase2', keep="last")

## sanity check
df.checked.value_counts()
df.fact_rating_phase1.value_counts()
df.fact_rating_phase1 = df.fact_rating_phase1.str.lower()
df.fact_rating_phase1.value_counts()

# remove unwanted ratings
df = df.loc[df['fact_rating_phase1'].str.lower().isin({'false','true','mixture','mostly false','mostly true'})]
df.id.describe()

df = df.sort_values(by=['snopes_url_phase1'])
df['article_claim_phase1'] = df['article_claim_phase1'].apply(lambda x: replace_unusual_characters(x))

#final preprocessing for figure8 platform
df['original_article_text_phase2'] = df['original_article_text_phase2'].apply(lambda x: replace_newlines(x))


# split into checked and uncheked
df_ch = df.loc[df['checked'] == 1]
df_un = df.loc[df['checked'] == 0]

df_ch.to_csv("../data/snopes/snopes_checked_v01_forCrowd.csv")
df_un.to_csv("../data/snopes/snopes_leftover_v01_forCrowd.csv")

#news_data_summary()
#texts, labels =  load_data_combined("../data/buzzfeed-debunk-combined/all-v02.txt")
#news_data_sampler(texts, labels,  train_size = 700, dev_size = 200, test_size = 200)
