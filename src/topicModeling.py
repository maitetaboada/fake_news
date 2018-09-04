
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import sys
import io

import gensim
from gensim import corpora
from textutils import DataLoading
from wordcloud import WordCloud
import numpy as np



#reload(sys)
#sys.setdefaultencoding('utf8')


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def printInfo(corpus): # prints general information about a corpus of text (list of strings)
    print("*********** GENERAL INFO ON CORPUS **********")
    lengths = [len(i) for i in corpus]
    print("Number of items in the corpus: " + str(len(corpus)))
    print("Avg length of text in the corpus: " + str(float(sum(lengths)) / len(lengths)))
    docs = [doc.split() for doc in corpus]
    print("Total vocabulary of the corpus: " + str(len(corpora.Dictionary(docs))))



def show_topics(doc_raw, ldamodel, cleaning = False, combine = False): # Working on new corpora or sub-corpora of different types
    if cleaning :
        doc_clean = [clean(doc).split() for doc in doc_raw]
    else:
        doc_clean = doc_raw
    if (combine):
        doc_clean = [[item for sublist in doc_clean for item in sublist]]


    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    print("Working on a test corpus of length " + str(len(corpus)))
    print("****** Topics found in each document of the test corpus ******")
    doc_by_topic = [ldamodel.get_document_topics(bow, minimum_probability=0.1) for bow in corpus]
    for doc in doc_by_topic:
        print(doc)
    print("***************************************************************")



# compile documents
buzzfeed_texts, buzzfeed_labels =  DataLoading.load_data_buzzfeed()
buzztop_texts, buzztop_labels =  DataLoading.load_data_buzzfeedtop()
snopes312_texts, snopes312_labels = DataLoading.load_data_snopes312()
rashkin_texts, rashkin_labels = DataLoading.load_data_rashkin("../data/rashkin/xtrain.txt")

doc_buzzfeed = [clean(doc).split() for doc in buzzfeed_texts]
doc_buzztop = [clean(doc).split() for doc in buzztop_texts]
doc_snopes312 = [clean(doc).split() for doc in snopes312_texts]
doc_rashkin = [clean(doc).split() for doc in rashkin_texts]


print("Test of list item:")
print([item for sublist in doc_buzzfeed for item in sublist])


# either use all test corpora put together or use a reference train corpus
doc_clean = doc_rashkin #doc_buzzfeed + doc_buzztop + doc_snopes312


# Creating the term dictionary of our courpus, where every unique term is assigned an index.
print("Dictionary preparation:")
dictionary = corpora.Dictionary(doc_clean)
print(len(dictionary))
dictionary.filter_n_most_frequent(100)
print(len(dictionary))
dictionary.filter_extremes(no_below=5, no_above=0.5)
print(len(dictionary))
print("Dictionary finalized.")

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
print(len(doc_term_matrix))


# Creating the object for LDA model using gensim library
print("Creating Lda...")
Lda = gensim.models.ldamodel.LdaModel


# Running and Trainign LDA model on the document term matrix.
print("Building model...")
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=5, random_state=1)

# Show topics with some most important words
for i in ldamodel.print_topics(num_topics=10, num_words=10):
    print(i)



print("For buzzfeed data:")
#show_topics(doc_buzzfeed, ldamodel)

print("For buzztop data:")
#show_topics(doc_buzztop, ldamodel)

print("For snopes312 data:")
#show_topics(doc_snopes312, ldamodel)



print("For buzzfeed per label:")
unique, counts = np.unique(buzzfeed_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(buzzfeed_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_buzzfeed)[l_index]
    show_topics(sub_corpus, ldamodel, combine=True)



print("For buzztop per label:")
unique, counts = np.unique(buzztop_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(buzztop_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_buzztop)[l_index]
    show_topics(sub_corpus, ldamodel, combine=True)


print("For snopes312 per label:")
unique, counts = np.unique(snopes312_labels, return_counts=True)
print(np.asarray((unique, counts)).T)
for l in unique:
    print("Label: " + str(l))
    l_index = (np.where(snopes312_labels == l)[0]).tolist()
    sub_corpus = np.asarray(doc_snopes312)[l_index]
    show_topics(sub_corpus, ldamodel, combine=True)



#visulalize topics by wordcloud

from pylab import *
plt.figure(figsize=(30, ldamodel.num_topics))
subplots_adjust(hspace=0.1, wspace=0.1)
plt.axis("off")
for t in range(ldamodel.num_topics):
    print(ldamodel.show_topic(t, 10))

    ax1 = subplot((ldamodel.num_topics/5 +1), 5, t+1)
    ax1.imshow(WordCloud(background_color="white").fit_words(dict(ldamodel.show_topic(t, 10))))
    ax1.set_title("Topic #" + str(t))

plt.savefig('test_all.pdf', format='pdf')



