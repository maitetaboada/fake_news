
import sys
import os
import io

#sys.path = ['../utils'] + sys.path


import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer




class Word2vecFeatures:
    """
    The distributional bag of word model of text meaning:
    Vector representation of a text is obtained by adding up
    the vectors of its constituting words.
    """

    w2vModel = None

    def __init__(self, file):
        """

        :param file: pre-trained vectors file
        """

        print("Word2VecFeatures init: loading word2vec model")
        self.w2vModel = Word2Vec.load_word2vec_format(file, binary=True)  #gensim.models.KeyedVectors.load_word2vec_format(modelFile, binary=True)
        return

    def encode(self, texts):

        textvecs = list()
        textvec = None
        stoplist = stopwords.words('english')
        for index, text in enumerate(texts):
            #print(text)
            #print(type(text))
            text = text.lower().split()
            text = [i for i in text if i not in stoplist]
            wordCount = 0
            for word in text:
                if word in self.w2vModel.vocab:
                    if wordCount == 0:
                        textvec = self.w2vModel[word]
                    else:
                        textvec = np.add(textvec, self.w2vModel[word])
                    wordCount+=1
            if(wordCount == 0):
                #print(str(text))
                raise ValueError("Cannot encode text " + str(index) + " : all words unknown to model!  ::" + str(text))
            else:
                textvecs.append(normalize(textvec[:,np.newaxis], axis=0).ravel())
        return np.array(textvecs)
   
    def get_features(self, text):
        """
        :param text:
        :return: the exact avg vector of words in a text (later try other representations,e.g., mi)
        """

        features = self.encode([text])
        print len(features[0]), " Word2vec features generated "
        return features[0]

    
class TfIdfFeatures():
    """
    The tf/idf representation of text:
    TF/IDF vector for a text is obtained based on the entire reference corpus (collection of documents) given at initialization.
    This corpus can be the entire training corpus.
    """
    model = None

    def __init__(self, corpus):
        """

        :param corpus: list of documents
        """
        print("TfIdfFeatures() init: loading word2vec model")
        print("Number of documents in the tf/idf corpus: " + str(len(corpus)))
        self.model = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=1, stop_words='english')
        self.model.fit_transform(corpus)
        feature_names = self.model.get_feature_names()
        print("Number of terms used in the tf/idf table: " + str(len(feature_names)) )
        print feature_names[100:120]

    def encode(self, texts):
        print("transforming:...")
        #textvecs = []
        textvec = self.model.transform(texts)
        print(textvec)
            #textvecs.append(textvec)
        #print(textvecs)
        #return np.array(textvecs)

    def get_features(self, text):
        # obtains the bow distributional representation of the text
        # currently returns the exact avg vector of words in a text (later try other representations,e.g., mi)
        f = self.encode([text])
        print(f[0]) #" word2vec features generated "
        return f[0]
        
        
        
     




if __name__ == '__main__':

    m = Word2vecFeatures("/Users/fa/workspace/shared/sfu/fake_news/pretrained/embeddings/vectorsW.bin")
    print m.get_features("Greatings my dear lady!")
    print len(m.get_features("Book newspaper hello president cat health !"))



    """
    for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file = os.path.join(directory, filename)
                file_text = io.open(file,encoding = "ISO-8859-1").read()
                #print(file)
                corpus.append(file_text)
                continue
            else:
                continue
    """

xl = pd.ExcelFile("/Users/fa/workspace/temp/rubin/data.xlsx")
df = xl.parse("Sheet1")
corpus = df["Article Headline"]
m = TfIdfFeatures(corpus)
m.encode(corpus[1])
m.encode("doctor hospital bad good newspaper!")
m.encode(["doctor hospital bad good newspaper president health !"])
print m.get_features("doctor hospital bad good newspaper!")
print len(m.get_features("doctor hospital bad good newspaper!"))





    
    

